#include "live_session.h"

#ifdef VG_HAS_GPU
#include "videogammetry/gpu/gpu_orb.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "camera_reader.h"
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

LiveSession::LiveSession(std::string path, Settings settings, std::wstring cameraId)
    : path_(std::move(path)), settings_(std::move(settings)), cameraId_(std::move(cameraId)),
      realTime_(settings_.realTime && cameraId_.empty()) {
    snapshot_ = std::make_shared<LiveSnapshot>();
    thread_ = std::thread([this] { run(); });
}

LiveSession::~LiveSession() {
    stop_ = true;
    endCapture();
    if (thread_.joinable()) thread_.join();
}

void LiveSession::endCapture() {
    endCapture_ = true;
    std::lock_guard<std::mutex> lock(cameraMutex_);
    if (camera_) camera_->close();
}

std::shared_ptr<const LiveSnapshot> LiveSession::latest() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
}

bool LiveSession::savePly(const std::string& path, std::size_t& pointsWritten) const {
    if (!done_ || !reconstructor_) return false;
    const vg::PointCloud cloud = reconstructor_->pointCloud();
    pointsWritten = cloud.points.size();
    std::shared_ptr<vg::DenseReconstructor> dense;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        dense = dense_;
    }
    if (dense) {
        std::string densePath = path;
        if (densePath.size() > 4 && densePath.ends_with(".ply")) densePath.resize(densePath.size() - 4);
        vg::writePly(dense->pointCloud(), densePath + "_dense.ply");
    }
    return vg::writePly(cloud, path);
}

void LiveSession::run() {
    // Video: decoded ahead on its own thread (FramePrefetcher). Camera: the
    // newest frame each time, so the map follows the camera without lag.
    VideoReader reader;
    CameraReader cam;
    std::unique_ptr<FramePrefetcher> frames;
    std::string error;
    if (camera()) {
        if (!cam.open(cameraId_, settings_.options.processSize, error)) {
            publish(nullptr, 0.0, true, "error: " + error);
            done_ = true;
            return;
        }
        std::lock_guard<std::mutex> lock(cameraMutex_);
        camera_ = &cam;
        if (endCapture_) cam.close();
    } else {
        if (!reader.open(path_, error)) {
            publish(nullptr, 0.0, true, "error: " + error);
            done_ = true;
            return;
        }
        reader.setMaxSize(settings_.options.processSize);  // downscale while converting to RGB
        frameCount_ = static_cast<long>(reader.info().frameCountEstimate);
        frames = std::make_unique<FramePrefetcher>(reader);
    }
    reconstructor_ = std::make_unique<vg::Reconstructor>(settings_.options);

    // Frames -> parallel feature extraction -> this thread (tracking and
    // mapping, in order), as in the CLI.
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
        while (!stop_ && !endCapture_ && (frames ? frames->next(f) : cam.read(f))) {
            while (paused_ && !stop_ && !endCapture_ && !frames) cam.read(f);  // keep only the newest
            pipeline.push(f);
        }
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
        if (settings_.dense && camera() && reconstructor_->lastWasKeyframe()) {  // after publish: the pixels are handed over
            std::vector<std::uint8_t> pixels = pipeline.takePixels();
            if (!pixels.empty()) {
                const vg::FrameView view = frame;  // points into `pixels`, which keeps its buffer when moved
                keyframeImages_[reconstructor_->keyframes().back().frameIndex] = {std::move(pixels), view};
            }
        }
    }
    feeder.join();
    {
        std::lock_guard<std::mutex> lock(cameraMutex_);
        camera_ = nullptr;
    }
    if (stop_) return;

    const std::string readError = frames ? frames->error() : (endCapture_ ? std::string() : cam.lastError());
    if (!readError.empty()) {
        publish(nullptr, processingFps, false, (frames ? "decode error: " : "camera error: ") + readError);
    } else if (camera() || endCapture_) {
        publish(nullptr, processingFps, false, "capture ended, refining the whole map (global bundle adjustment)...");
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
    if (settings_.dense && !stop_) runDense(processingFps, summary);
}

void LiveSession::runDense(double processingFps, const std::string& summary) {
    if (reconstructor_->keyframes().size() < 3) return;
    auto dense = std::make_shared<vg::DenseReconstructor>(*reconstructor_, settings_.denseParams);
    const auto status = [&](const std::string& text) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            denseStatus_ = text;
        }
        publish(nullptr, processingFps, true, summary);
    };
    {
        std::lock_guard<std::mutex> lock(mutex_);
        dense_ = dense;
    }

    // Keyframe images: kept from the capture, or decoded again from the video.
    const std::vector<long>& needed = dense->frames();
    char buf[256];
    if (camera()) {
        for (const auto& [index, image] : keyframeImages_) dense->addImage(index, image.second);
        keyframeImages_.clear();
    } else {
        VideoReader reader;
        std::string error;
        if (!reader.open(path_, error)) {
            status("dense: " + error);
            return;
        }
        reader.setMaxSize(dense->imageLongSide(reader.info().width, reader.info().height));
        std::size_t next = 0;
        vg::FrameView frame;
        Clock::time_point lastStatus;
        for (long index = 0; next < needed.size() && !stop_ && reader.read(frame); ++index) {
            if (index != needed[next]) continue;
            dense->addImage(index, frame);
            ++next;
            if (Clock::now() - lastStatus >= std::chrono::milliseconds(200)) {
                std::snprintf(buf, sizeof(buf), "dense: reading keyframe images %zu / %zu", next, needed.size());
                status(buf);
                lastStatus = Clock::now();
            }
        }
    }
    if (stop_) return;

    Clock::time_point lastStatus;
    const auto update = [&]() {
        const vg::DenseReconstructor::Progress p = dense->progress();
        if (!p.finished && Clock::now() - lastStatus < std::chrono::milliseconds(200)) return;
        lastStatus = Clock::now();
        if (p.finished) {
            std::snprintf(buf, sizeof(buf), "dense: done, %.2f M points from %d keyframes (%dx%d)", p.points / 1e6,
                          p.keyframes, p.width, p.height);
        } else {
            std::snprintf(buf, sizeof(buf), "dense: depth maps %d / %d (last %.1f s)   fused %d   %.2f M points   %dx%d",
                          p.depthMaps, static_cast<int>(needed.size()), p.lastDepthMs / 1000, p.fused, p.points / 1e6,
                          p.width, p.height);
        }
        status(buf);
    };
    status("dense: starting");
    dense->run(stop_, update);
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
    s->noise.reserve(mapPoints.size());
    // Hidden like in the saved cloud (Reconstructor::pointCloud).
    const float maxNoise = static_cast<float>(settings_.options.maxPointNoisePct / 100.0);
    for (std::size_t i = 0; i < mapPoints.size(); ++i) {
        const vg::MapPoint& p = mapPoints[i];
        if (p.bad) continue;
        if (maxNoise > 0 && p.noise > maxNoise) continue;
        vg::Point q;
        q.x = static_cast<float>(p.position.x);
        q.y = static_cast<float>(p.position.y);
        q.z = static_cast<float>(p.position.z);
        q.r = p.color[0];
        q.g = p.color[1];
        q.b = p.color[2];
        s->points.push_back(q);
        s->tracked.push_back(seen[i]);
        s->noise.push_back(p.noise);
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
    if (dense_) s->dense = dense_->cloud();
    s->denseStatus = denseStatus_;
    s->version = ++version_;
    snapshot_ = std::move(s);
}
