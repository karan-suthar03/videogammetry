#include "videogammetry/features.h"

#include <algorithm>
#include <cmath>

namespace vg {

FeatureExtractor::FeatureExtractor(int processSize, const OrbParams& orb, ThreadPool* pool)
    : processSize_(processSize), pool_(pool), orb_(orb, pool) {}

void FeatureExtractor::extract(const FrameView& frame, long index, FrameFeatures& out) {
    out.index = index;
    out.timestamp = frame.timestamp;

    toGray(frame, gray_, pool_);
    const Image* input = &gray_;
    out.processScale = 1.0f;
    const int longSide = std::max(frame.width, frame.height);
    if (processSize_ > 0 && longSide > processSize_) {
        const double s = double(longSide) / processSize_;
        const int w = std::max(1, static_cast<int>(std::lround(frame.width / s)));
        const int h = std::max(1, static_cast<int>(std::lround(frame.height / s)));
        resizeBilinear(gray_, w, h, processed_, pool_);
        out.processScale = float(frame.width) / w;
        input = &processed_;
    }
    out.width = input->width;
    out.height = input->height;

    orb_.extract(*input, out.keypoints, out.descriptors);

    // Colors from the full-resolution frame, for the point cloud.
    out.colors.resize(out.keypoints.size());
    const int bpp = frame.format == FrameView::Format::Gray8 ? 1 : (frame.format == FrameView::Format::RGB8 ? 3 : 4);
    for (std::size_t i = 0; i < out.keypoints.size(); ++i) {
        const Keypoint& k = out.keypoints[i];
        const int x = std::clamp(static_cast<int>(std::lround((k.x + 0.5f) * out.processScale - 0.5f)), 0, frame.width - 1);
        const int y = std::clamp(static_cast<int>(std::lround((k.y + 0.5f) * out.processScale - 0.5f)), 0, frame.height - 1);
        const std::uint8_t* p = frame.data + static_cast<std::size_t>(y) * frame.stride + static_cast<std::size_t>(x) * bpp;
        out.colors[i] = bpp == 1 ? std::array<std::uint8_t, 3>{p[0], p[0], p[0]}
                                 : std::array<std::uint8_t, 3>{p[0], p[1], p[2]};
    }
}

FeaturePipeline::FeaturePipeline(int processSize, const OrbParams& orb, int workers,
                                 FrameExtractorFactory makeExtractor)
    : processSize_(processSize), orb_(orb), makeExtractor_(std::move(makeExtractor)) {
    const int n = workers > 0 ? workers : ThreadPool::hardwareThreads();
    maxInFlight_ = static_cast<std::size_t>(n) + 4;  // keep every worker busy, bound memory
    for (int i = 0; i < n; ++i) threads_.emplace_back([this] { work(); });
}

FeaturePipeline::~FeaturePipeline() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    changed_.notify_all();
    for (std::thread& t : threads_) t.join();
}

void FeaturePipeline::push(const FrameView& frame) {
    auto job = std::make_unique<Job>();
    const int bpp = frame.format == FrameView::Format::Gray8 ? 1 : (frame.format == FrameView::Format::RGB8 ? 3 : 4);
    const std::size_t rowBytes = static_cast<std::size_t>(frame.width) * bpp;
    job->pixels.resize(rowBytes * frame.height);
    for (int y = 0; y < frame.height; ++y) {
        std::copy(frame.data + static_cast<std::size_t>(y) * frame.stride,
                  frame.data + static_cast<std::size_t>(y) * frame.stride + rowBytes,
                  job->pixels.data() + rowBytes * y);
    }
    job->view = frame;
    job->view.data = job->pixels.data();
    job->view.stride = static_cast<int>(rowBytes);

    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [&] { return stopping_ || jobs_.size() < maxInFlight_; });
    if (stopping_) return;
    job->index = nextToPush_++;
    jobs_[job->index] = std::move(job);
    lock.unlock();
    changed_.notify_all();
}

void FeaturePipeline::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
    }
    changed_.notify_all();
}

void FeaturePipeline::work() {
    // Single-threaded each: parallelism is across frames.
    std::unique_ptr<FrameExtractor> extractor =
        makeExtractor_ ? makeExtractor_() : std::make_unique<FeatureExtractor>(processSize_, orb_);
    while (true) {
        Job* job = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait(lock, [&] { return stopping_ || nextToStart_ < nextToPush_; });
            if (stopping_) return;
            job = jobs_.at(nextToStart_++).get();  // stays alive until popped
        }
        extractor->extract(job->view, job->index, job->features);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job->done = true;
        }
        changed_.notify_all();
    }
}

bool FeaturePipeline::pop(FrameFeatures& features, FrameView* frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [&] {
        const auto it = jobs_.find(nextToPop_);
        return (it != jobs_.end() && it->second->done) || (closed_ && nextToPop_ >= nextToPush_);
    });
    const auto it = jobs_.find(nextToPop_);
    if (it == jobs_.end()) return false;
    current_ = std::move(it->second);
    jobs_.erase(it);
    ++nextToPop_;
    lock.unlock();
    changed_.notify_all();

    features = std::move(current_->features);
    if (frame) *frame = current_->view;
    return true;
}

}  // namespace vg
