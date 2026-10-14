#include "videogammetry/reconstructor.h"

#include <algorithm>
#include <cmath>

namespace vg {

namespace {

std::unique_ptr<ThreadPool> makePool(int threads) {
    const int n = threads > 0 ? threads : ThreadPool::hardwareThreads();
    return n > 1 ? std::make_unique<ThreadPool>(n) : nullptr;
}

}  // namespace

Reconstructor::Reconstructor(const Options& options)
    : options_(options), pool_(makePool(options.threads)), orb_(options.orb, pool_.get()) {}

void Reconstructor::addFrame(const FrameView& frame) {
    if (frame.data == nullptr || frame.width <= 0 || frame.height <= 0) return;

    {
        auto t = profiler_.scope("gray");
        toGray(frame, gray_, pool_.get());
    }

    const Image* input = &gray_;
    processScale_ = 1.0f;
    if (options_.processWidth > 0 && frame.width > options_.processWidth) {
        auto t = profiler_.scope("resize");
        const int w = options_.processWidth;
        const int h = static_cast<int>(std::lround(double(frame.height) * w / frame.width));
        resizeBilinear(gray_, w, h, processed_, pool_.get());
        processScale_ = float(frame.width) / w;
        input = &processed_;
    }

    // Reuse the oldest history slot's buffers for the new frame.
    const std::size_t keep = static_cast<std::size_t>(std::max(options_.matchGap, 1)) + 1;
    FrameFeatures current;
    if (history_.size() >= keep) {
        current = std::move(history_.front());
        history_.pop_front();
    }
    {
        auto t = profiler_.scope("orb");
        orb_.extract(*input, current.keypoints, current.descriptors);
    }
    history_.push_back(std::move(current));

    matches_.clear();
    matchStats_ = {};
    if (history_.size() == keep) {
        auto t = profiler_.scope("match");
        const FrameFeatures& query = history_.back();
        const FrameFeatures& train = history_.front();
        matchFeatures(query.keypoints, query.descriptors, train.keypoints, train.descriptors,
                      options_.match, matches_, &matchStats_, pool_.get());
    }

    // TODO(phase 1): pose estimation, triangulation.
    ++frameCount_;
}

}  // namespace vg
