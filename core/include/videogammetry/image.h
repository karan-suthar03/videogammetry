#pragma once

#include <cstdint>
#include <vector>

#include "videogammetry/frame.h"

namespace vg {

class ThreadPool;

// Owning 8-bit grayscale image, tightly packed (stride == width).
struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> data;

    void resize(int w, int h) {
        width = w;
        height = h;
        data.resize(static_cast<std::size_t>(w) * h);
    }
    bool empty() const { return width <= 0 || height <= 0; }

    std::uint8_t at(int x, int y) const { return data[static_cast<std::size_t>(y) * width + x]; }
    const std::uint8_t* row(int y) const { return data.data() + static_cast<std::size_t>(y) * width; }
    std::uint8_t* row(int y) { return data.data() + static_cast<std::size_t>(y) * width; }
};

// All operations split work by rows across `pool` when one is given; the
// result is identical with or without a pool.

// Converts any FrameView format to grayscale (BT.601 luma).
void toGray(const FrameView& frame, Image& out, ThreadPool* pool = nullptr);

// Taps of resizeBilinear along one axis: for each output index, the first
// of the two source samples and the weight of the second, in 1/256 units.
void bilinearTaps(int srcLength, int dstLength, std::vector<int>& first, std::vector<int>& weight);

// Bilinear resize to (width, height).
void resizeBilinear(const Image& src, int width, int height, Image& out, ThreadPool* pool = nullptr);

// 7x7 Gaussian blur, sigma = 2 (used to smooth images before sampling descriptors).
void gaussianBlur7(const Image& src, Image& out, ThreadPool* pool = nullptr);

}  // namespace vg
