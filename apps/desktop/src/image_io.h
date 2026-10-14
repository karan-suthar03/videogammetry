#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "videogammetry/frame.h"
#include "videogammetry/matcher.h"
#include "videogammetry/orb.h"

// Owning, tightly packed RGB8 image.
struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;

    vg::FrameView view() const {
        vg::FrameView v;
        v.data = pixels.data();
        v.width = width;
        v.height = height;
        v.stride = width * 3;
        v.format = vg::FrameView::Format::RGB8;
        return v;
    }
};

// Bilinear resize of an RGB8 frame.
RgbImage resizeRgb(const vg::FrameView& frame, int width, int height);

// Stacks `top` above `bottom` and draws a line for every match from the
// train keypoint (top) to the query keypoint (bottom). Keypoints must be in
// the images' pixel coordinates. Line color goes green -> red with distance.
RgbImage drawMatches(const RgbImage& top, const std::vector<vg::Keypoint>& topKeypoints,
                     const RgbImage& bottom, const std::vector<vg::Keypoint>& bottomKeypoints,
                     const std::vector<vg::Match>& matches, int maxDistance);

// Draws each match on `image` as a line from the train keypoint (earlier
// position) to a dot at the query keypoint (current position): optical flow.
RgbImage drawFlow(const RgbImage& image, const std::vector<vg::Keypoint>& trainKeypoints,
                  const std::vector<vg::Keypoint>& queryKeypoints,
                  const std::vector<vg::Match>& matches, int maxDistance);

// Writes an RGB8 frame. Format is chosen by extension: .bmp, otherwise binary .ppm.
bool writeImage(const vg::FrameView& frame, const std::string& path);

// Copies an RGB8 frame and draws keypoints on it: a circle sized by pyramid
// level (color = level) and a line showing the orientation. Keypoint
// coordinates are in processed-image pixels; processScale maps them to the frame.
std::vector<std::uint8_t> drawKeypoints(const vg::FrameView& frame,
                                        const std::vector<vg::Keypoint>& keypoints,
                                        float processScale);
