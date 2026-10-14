#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "videogammetry/frame.h"
#include "videogammetry/image.h"
#include "videogammetry/orb.h"
#include "videogammetry/thread_pool.h"

namespace vg {

// Everything the reconstruction needs from one video frame. Computed from
// the frame alone (no map state), so frames can be processed in parallel.
struct FrameFeatures {
    long index = -1;
    double timestamp = 0;
    int width = 0, height = 0;  // processed image size
    float processScale = 1.0f;  // input-frame pixels per processed pixel
    std::vector<Keypoint> keypoints;
    std::vector<Descriptor> descriptors;
    std::vector<std::array<std::uint8_t, 3>> colors;  // input-frame RGB at each keypoint
};

// Gray conversion, downscaling, ORB and color sampling for one frame stream.
// Not thread-safe; use one per thread.
class FeatureExtractor {
public:
    // processSize: the long side of the frame is downscaled to this many pixels
    // (0 = full resolution), so portrait and landscape are treated alike.
    FeatureExtractor(int processSize, const OrbParams& orb, ThreadPool* pool = nullptr);

    void extract(const FrameView& frame, long index, FrameFeatures& out);

private:
    int processSize_;
    ThreadPool* pool_;
    OrbExtractor orb_;
    Image gray_, processed_;
};

// Extracts features of many frames at once on `workers` threads (one frame
// per thread, no per-frame synchronization) and returns them in input order.
// push() copies the frame; pop() blocks until the next frame is ready.
class FeaturePipeline {
public:
    // workers <= 0: all hardware threads.
    FeaturePipeline(int processSize, const OrbParams& orb, int workers = 0);
    ~FeaturePipeline();
    FeaturePipeline(const FeaturePipeline&) = delete;
    FeaturePipeline& operator=(const FeaturePipeline&) = delete;

    // Queues a frame (RGB8/RGBA8/Gray8); blocks while too many are in flight.
    void push(const FrameView& frame);
    // No more frames will be pushed.
    void close();
    // Next frame's features, in push order. If `frame` is given it receives a
    // view of the frame's pixels, valid until the next pop(). Returns false
    // once closed and drained.
    bool pop(FrameFeatures& features, FrameView* frame = nullptr);

    int workers() const { return static_cast<int>(threads_.size()); }

private:
    struct Job {
        long index;
        std::vector<std::uint8_t> pixels;
        FrameView view;
        FrameFeatures features;
        bool done = false;
    };

    void work();

    std::vector<std::thread> threads_;
    std::size_t maxInFlight_;
    int processSize_;
    OrbParams orb_;

    std::mutex mutex_;
    std::condition_variable changed_;
    std::map<long, std::unique_ptr<Job>> jobs_;  // in flight, by index
    long nextToStart_ = 0, nextToPush_ = 0, nextToPop_ = 0;
    bool closed_ = false, stopping_ = false;
    std::unique_ptr<Job> current_;  // last popped (owns the returned pixels)
};

}  // namespace vg
