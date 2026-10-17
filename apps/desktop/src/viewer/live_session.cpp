#include "live_session.h"

#ifdef VG_HAS_GPU
#include "videogammetry/gpu/gpu_orb.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "frame_prefetcher.h"
#include "video_reader.h"

namespace {

using Clock = std::chrono::steady_clock;

// Median distance from `from` to the points (sampled): sets the size of
// camera frustums and the view distance, since monocular maps have no
// absolute scale.
double typicalDistance(const std::vector<vg::Point>& points, const vg::Vec3& from) {
    if (points.empty()) return 1.0;
    const std::size_t step = std::max<std::size_t>(1, points.size() / 2000);
    std::vector<double> d;
    d.reserve(points.size() / step + 1);
    for (std::size_t i = 0; i < points.size(); i += step) {
        const vg::Point& p = points[i];
        d.push_back(vg::norm(vg::Vec3{p.x, p.y, p.z} - from));
    }
    std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
    return std::max(d[d.size() / 2], 1e-6);
}

}  // namespace

LiveSession::LiveSession(std::string path, Settings settings)
    : path_(std::move(path)), settings_(std::move(settings)), realTime_(settings_.realTime) {
    snapshot_ = std::make_shared<LiveSnapshot>();
    thread_ = std::thread([this] { run(); });
}

LiveSession::~LiveSession() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
}

std::shared_ptr<const LiveSnapshot> LiveSession::latest() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
}

bool LiveSession::savePly(const std::string& path, std::size_t& pointsWritten) const {
    if (!done_ || !reconstructor_) return false;
    const vg::PointCloud cloud = reconstructor_->pointCloud();
    pointsWritten = cloud.points.size();
    return vg::writePly(cloud, path);
}

void LiveSession::run() {
    VideoReader reader;
    std::string error;
    if (!reader.open(path_, error)) {
        publish(nullptr, 0.0, true, "error: " + error);
        done_ = true;
        return;
    }
    reader.setMaxSize(settings_.options.processSize);  // downscale while converting to RGB
    frameCount_ = static_cast<long>(reader.info().frameCountEstimate);
    reconstructor_ = std::make_unique<vg::Reconstructor>(settings_.options);

    // Decode thread -> parallel feature extraction -> this thread (tracking
    // and mapping, in order), as in the CLI.
    FramePrefetcher frames(reader);
    vg::FrameExtractorFactory makeExtractor;
    int workers = 0;
    extractor_ = "CPU";
#ifdef VG_HAS_GPU
    if (settings_.gpu) {
        makeExtractor = vg::gpu::makeGpuExtractorFactory(settings_.options.processSize, settings_.options.orb, extractor_);
        if (makeExtractor) workers = 3;  // overlap upload, GPU and host work
    }
#endif
    vg::FeaturePipeline pipeline(settings_.options.processSize, settings_.options.orb, workers,
                                 std::move(makeExtractor));
    std::thread feeder([&] {
        vg::FrameView f;
        while (!stop_ && frames.next(f)) pipeline.push(f);
        pipeline.close();
    });

    // Pacing: frame timestamps are mapped to wall time from an origin that is
    // reset after a pause, a speed change, or when processing falls behind.
    bool resync = true;
    Clock::time_point wallOrigin;
    double videoOrigin = 0.0;

    // Processing rate: frames per second of tracking/mapping time alone.
    double windowCoreSec = 0.0;
    int windowFrames = 0;
    Clock::time_point windowStart = Clock::now();
    double processingFps = 0.0;
    Clock::time_point lastPublish{};

    vg::FrameFeatures features;
    vg::FrameView frame;
    while (pipeline.pop(features, &frame)) {
        if (stop_) continue;  // drain so the feeder can finish

        while (!stop_) {
            if (paused_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                resync = true;
                continue;
            }
            if (!realTime_) {
                resync = true;
                break;
            }
            const Clock::time_point now = Clock::now();
            if (resync) {
                wallOrigin = now;
                videoOrigin = frame.timestamp;
                resync = false;
            }
            const Clock::time_point due =
                wallOrigin + std::chrono::duration_cast<Clock::duration>(
                                 std::chrono::duration<double>(frame.timestamp - videoOrigin));
            if (now >= due) {
                if (now - due > std::chrono::milliseconds(100)) resync = true;  // behind: don't race to catch up
                break;
            }
            std::this_thread::sleep_for(std::min<Clock::duration>(due - now, std::chrono::milliseconds(5)));
        }
        if (stop_) continue;

        const Clock::time_point coreStart = Clock::now();
        reconstructor_->addFeatures(std::move(features));
        windowCoreSec += std::chrono::duration<double>(Clock::now() - coreStart).count();
        ++windowFrames;
        if (Clock::now() - windowStart >= std::chrono::milliseconds(500) && windowCoreSec > 0) {
            processingFps = windowFrames / windowCoreSec;
            windowCoreSec = 0.0;
            windowFrames = 0;
            windowStart = Clock::now();
        }

        frame_ = static_cast<long>(reconstructor_->frameCount());
        timestamp_ = frame.timestamp;
        const Clock::time_point now = Clock::now();
        if (reconstructor_->lastWasKeyframe() || now - lastPublish >= std::chrono::milliseconds(15)) {
            publish(&frame, processingFps, false, {});
            lastPublish = now;
        }
    }
    feeder.join();
    if (stop_) return;

    if (!frames.error().empty()) {
        publish(nullptr, processingFps, false, "decode error: " + frames.error());
    } else {
        publish(nullptr, processingFps, false, "video done, refining the whole map (global bundle adjustment)...");
    }
    reconstructor_->finish();
    const vg::Reconstructor::Stats& st = reconstructor_->stats();
    char summary[256];
    if (st.globalBA.ok) {
        std::snprintf(summary, sizeof(summary), "done: global bundle adjustment %.2f -> %.2f px reprojection error",
                      st.globalBA.initialRmsPx, st.globalBA.finalRmsPx);
    } else {
        std::snprintf(summary, sizeof(summary), "done");
    }
    publish(nullptr, processingFps, true, summary);
    done_ = true;
}

void LiveSession::publish(const vg::FrameView* frame, double processingFps, bool finished,
                          const std::string& message) {
    auto s = std::make_shared<LiveSnapshot>();
    s->finished = finished;
    s->message = message;
    s->extractor = extractor_;
    if (!reconstructor_) {  // the video could not be opened
        std::lock_guard<std::mutex> lock(mutex_);
        s->version = ++version_;
        snapshot_ = std::move(s);
        return;
    }
    const vg::Reconstructor& r = *reconstructor_;

    const std::vector<vg::MapPoint>& mapPoints = r.mapPoints();
    std::vector<std::uint8_t> seen(mapPoints.size(), 0);
    for (int id : r.lastTrackedPoints()) {
        if (id >= 0 && static_cast<std::size_t>(id) < seen.size()) seen[id] = 1;
    }
    s->points.reserve(mapPoints.size());
    s->tracked.reserve(mapPoints.size());
    for (std::size_t i = 0; i < mapPoints.size(); ++i) {
        const vg::MapPoint& p = mapPoints[i];
        if (p.bad) continue;
        vg::Point q;
        q.x = static_cast<float>(p.position.x);
        q.y = static_cast<float>(p.position.y);
        q.z = static_cast<float>(p.position.z);
        q.r = p.color[0];
        q.g = p.color[1];
        q.b = p.color[2];
        s->points.push_back(q);
        s->tracked.push_back(seen[i]);
    }
    for (const vg::Keyframe& k : r.keyframes()) s->keyframes.push_back(k.pose);
    s->currentValid = r.state() == vg::Reconstructor::State::Tracking && !r.keyframes().empty();
    s->current = s->currentValid ? r.lastPose() : (r.keyframes().empty() ? vg::Pose{} : r.keyframes().back().pose);
    s->sceneScale = typicalDistance(s->points, s->current.center());

    const vg::CameraIntrinsics& k = r.intrinsics();
    if (k.fx > 0) {
        s->focal = k.fx;
        s->imageWidth = static_cast<int>(std::lround(2 * k.cx));
        s->imageHeight = static_cast<int>(std::lround(2 * k.cy));
    }

    if (frame && frame->format == vg::FrameView::Format::RGB8 && frame->width > 0) {
        const double shrink =
            std::min(1.0, double(settings_.thumbnailSize) / std::max(frame->width, frame->height));
        const int tw = std::max(1, static_cast<int>(std::lround(frame->width * shrink)));
        const int th = std::max(1, static_cast<int>(std::lround(frame->height * shrink)));
        const RgbImage small = resizeRgb(*frame, tw, th);
        thumbnail_ = drawTracking(small.view(), r.lastKeypoints(), r.lastTrackedPoints(),
                                  r.processScale() * float(tw) / float(frame->width));
    }
    s->thumbnail = thumbnail_;

    s->frame = frame_;
    s->frameCount = frameCount_;
    s->timestamp = timestamp_;
    s->state = r.state();
    s->trackedCount = r.lastTrackedCount();
    s->maps = r.stats().maps;
    s->fovDeg = r.fovDeg();
    s->startFovDeg = settings_.options.fovDeg;
    s->focalUpdates = r.stats().focalUpdates;
    s->k1 = r.distortion();
    s->processingFps = processingFps;

    std::lock_guard<std::mutex> lock(mutex_);
    s->version = ++version_;
    snapshot_ = std::move(s);
}
