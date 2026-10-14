#include "frame_prefetcher.h"

#include <algorithm>

FramePrefetcher::FramePrefetcher(VideoReader& reader, int capacity)
    : reader_(reader), capacity_(static_cast<std::size_t>(std::max(capacity, 1))) {
    thread_ = std::thread([this] { produce(); });
}

FramePrefetcher::~FramePrefetcher() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    changed_.notify_all();
    thread_.join();
}

void FramePrefetcher::produce() {
    vg::FrameView frame;
    while (true) {
        // Wait for room in the queue.
        std::vector<std::uint8_t> buffer;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait(lock, [&] { return stop_ || ready_.size() < capacity_; });
            if (stop_) return;
            if (!freeBuffers_.empty()) {
                buffer = std::move(freeBuffers_.back());
                freeBuffers_.pop_back();
            }
        }

        // Decode outside the lock.
        const bool ok = reader_.read(frame);
        if (!ok) {
            std::lock_guard<std::mutex> lock(mutex_);
            finished_ = true;
            error_ = reader_.lastError();
            changed_.notify_all();
            return;
        }

        Slot slot;
        slot.pixels = std::move(buffer);
        const std::size_t rowBytes = static_cast<std::size_t>(frame.width) * 3;
        slot.pixels.resize(rowBytes * frame.height);
        for (int y = 0; y < frame.height; ++y) {
            const std::uint8_t* src = frame.data + static_cast<std::size_t>(y) * frame.stride;
            std::copy(src, src + rowBytes, slot.pixels.data() + rowBytes * y);
        }
        slot.view = frame;
        slot.view.data = slot.pixels.data();
        slot.view.stride = static_cast<int>(rowBytes);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            ready_.push_back(std::move(slot));
        }
        changed_.notify_all();
    }
}

bool FramePrefetcher::next(vg::FrameView& frame) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Hand the previous frame's buffer back to the producer.
    if (!current_.pixels.empty()) {
        freeBuffers_.push_back(std::move(current_.pixels));
        current_.pixels.clear();
    }

    changed_.wait(lock, [&] { return !ready_.empty() || finished_; });
    if (ready_.empty()) return false;

    current_ = std::move(ready_.front());
    ready_.pop_front();
    lock.unlock();
    changed_.notify_all();

    frame = current_.view;
    return true;
}
