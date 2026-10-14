#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "video_reader.h"

// Decodes frames on a background thread, up to `capacity` frames ahead, so
// video decoding overlaps with processing of the current frame.
class FramePrefetcher {
public:
    FramePrefetcher(VideoReader& reader, int capacity = 4);
    ~FramePrefetcher();
    FramePrefetcher(const FramePrefetcher&) = delete;
    FramePrefetcher& operator=(const FramePrefetcher&) = delete;

    // Next frame in order; the view stays valid until the next call.
    // Returns false at end of stream or on error (see error()).
    bool next(vg::FrameView& frame);

    const std::string& error() const { return error_; }

private:
    struct Slot {
        std::vector<std::uint8_t> pixels;
        vg::FrameView view;
    };

    void produce();

    VideoReader& reader_;
    const std::size_t capacity_;
    std::thread thread_;

    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<Slot> ready_;
    std::vector<std::vector<std::uint8_t>> freeBuffers_;
    bool finished_ = false;  // producer reached end of stream / error
    bool stop_ = false;      // consumer is shutting down
    std::string error_;

    Slot current_;
};
