#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "videogammetry/frame.h"

// Decodes a video file into RGB8 frames using FFmpeg, optionally downscaled.
// FFmpeg types stay inside video_reader.cpp; nothing else sees them.
class VideoReader {
public:
    struct Info {
        int width = 0;
        int height = 0;
        double fps = 0.0;
        double durationSec = 0.0;
        std::int64_t frameCountEstimate = 0;  // from container metadata, may be 0
        std::string codec;
    };

    VideoReader();
    ~VideoReader();
    VideoReader(const VideoReader&) = delete;
    VideoReader& operator=(const VideoReader&) = delete;

    // Returns false and fills `error` on failure.
    bool open(const std::string& path, std::string& error);

    const Info& info() const;

    // Downscale frames while converting them so the long side is at most
    // `longSide` pixels (0 = native size). Call after open(). Scaling here is
    // cheaper than converting the full frame and shrinking it afterwards.
    void setMaxSize(int longSide);

    // Decodes the next frame. The returned view stays valid until the next call.
    // Returns false at end of stream or on a decode error (see lastError()).
    bool read(vg::FrameView& frame);

    const std::string& lastError() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
