#include "videogammetry/reconstructor.h"

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

    {
        auto t = profiler_.scope("orb");
        orb_.extract(*input, keypoints_, descriptors_);
    }

    // TODO(phase 1): matching, pose estimation, triangulation.
    ++frameCount_;
}

}  // namespace vg
