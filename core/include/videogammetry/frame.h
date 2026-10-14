#pragma once

#include <cstdint>

namespace vg {

// A non-owning view of one decoded video frame.
// Each platform app decodes video (or reads the camera) its own way and hands
// frames to the core in this format, so the core never touches codecs or OS APIs.
struct FrameView {
    enum class Format { Gray8, RGB8, RGBA8 };

    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;          // bytes per row
    Format format = Format::Gray8;
    double timestamp = 0.0;  // seconds
};

}  // namespace vg
