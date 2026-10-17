// Compares the GPU feature extractor with the CPU one on a video: keypoints
// must be identical (position, level, response, angle); descriptors may
// differ in rare bits (GPU sine/cosine). Also times both, one frame at a time.
//
// Usage: gpu_check <video> [--config viewer.ini] [--frames N] [--every K]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "options_table.h"
#include "video_reader.h"
#include "videogammetry/features.h"
#include "videogammetry/gpu/gpu_orb.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gpu_check <video> [--config file.ini] [--frames N] [--every K]\n");
        return 1;
    }
    vg::Reconstructor::Options options;
    long maxFrames = 100, every = 1;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            if (!loadOptionsFile(argv[++i], options)) {
                std::fprintf(stderr, "cannot read %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            maxFrames = std::strtol(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--every") == 0 && i + 1 < argc) {
            every = std::max(1L, std::strtol(argv[++i], nullptr, 10));
        }
    }

    VideoReader reader;
    std::string error;
    if (!reader.open(argv[1], error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    reader.setMaxSize(options.processSize);

    std::string status;
    const vg::FrameExtractorFactory factory = vg::gpu::makeGpuExtractorFactory(options.processSize, options.orb, status);
    std::printf("extractor: %s\n", status.c_str());
    if (!factory) return 1;
    std::unique_ptr<vg::FrameExtractor> gpu = factory();
    vg::FeatureExtractor cpu(options.processSize, options.orb);

    using Clock = std::chrono::steady_clock;
    double cpuMs = 0, gpuMs = 0, gpuMax = 0;
    long frames = 0, keypoints = 0, keypointMismatch = 0, countMismatch = 0, descriptorsDiffering = 0;
    long long bitsDiffering = 0;
    vg::FrameView frame;
    vg::FrameFeatures a, b;
    for (long index = 0; frames < maxFrames && reader.read(frame); ++index) {
        if (index % every != 0) continue;
        auto t0 = Clock::now();
        cpu.extract(frame, index, a);
        auto t1 = Clock::now();
        gpu->extract(frame, index, b);
        auto t2 = Clock::now();
        if (frames > 0) {  // the first GPU frame includes buffer allocation
            cpuMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double g = std::chrono::duration<double, std::milli>(t2 - t1).count();
            gpuMs += g;
            gpuMax = std::max(gpuMax, g);
        }
        ++frames;

        if (a.keypoints.size() != b.keypoints.size()) ++countMismatch;
        const std::size_t n = std::min(a.keypoints.size(), b.keypoints.size());
        keypoints += static_cast<long>(a.keypoints.size());
        for (std::size_t i = 0; i < n; ++i) {
            const vg::Keypoint& p = a.keypoints[i];
            const vg::Keypoint& q = b.keypoints[i];
            if (p.x != q.x || p.y != q.y || p.level != q.level || p.response != q.response || p.angle != q.angle) {
                if (keypointMismatch < 5) {
                    std::printf("  frame %ld kp %zu: cpu (%.1f,%.1f) L%d r=%g a=%g  gpu (%.1f,%.1f) L%d r=%g a=%g\n",
                                index, i, p.x, p.y, p.level, p.response, p.angle, q.x, q.y, q.level, q.response,
                                q.angle);
                }
                ++keypointMismatch;
                continue;
            }
            int bits = 0;
            for (int w = 0; w < 4; ++w) bits += std::popcount(a.descriptors[i][w] ^ b.descriptors[i][w]);
            if (bits) ++descriptorsDiffering;
            bitsDiffering += bits;
        }
        if (a.colors != b.colors && keypointMismatch == 0 && countMismatch == 0) {
            std::printf("  frame %ld: colors differ\n", index);
        }
    }

    const long timed = std::max(1L, frames - 1);
    std::printf("frames %ld, keypoints %ld (%.0f per frame)\n", frames, keypoints, double(keypoints) / std::max(1L, frames));
    std::printf("keypoint count differs in %ld frames; %ld keypoints differ\n", countMismatch, keypointMismatch);
    std::printf("descriptors with any differing bit: %ld (%.4f%%), bits differing: %lld of %lld\n", descriptorsDiffering,
                100.0 * descriptorsDiffering / std::max(1L, keypoints), bitsDiffering, 256LL * keypoints);
    std::printf("time per frame: CPU %.2f ms (one thread), GPU %.2f ms (max %.2f)\n", cpuMs / timed, gpuMs / timed,
                gpuMax);
    return keypointMismatch == 0 && countMismatch == 0 ? 0 : 2;
}
