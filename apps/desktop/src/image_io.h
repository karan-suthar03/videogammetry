#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "videogammetry/frame.h"
#include "videogammetry/orb.h"

// Writes an RGB8 frame. Format is chosen by extension: .bmp, otherwise binary .ppm.
bool writeImage(const vg::FrameView& frame, const std::string& path);

// Copies an RGB8 frame and draws keypoints on it: a circle sized by pyramid
// level (color = level) and a line showing the orientation. Keypoint
// coordinates are in processed-image pixels; processScale maps them to the frame.
std::vector<std::uint8_t> drawKeypoints(const vg::FrameView& frame,
                                        const std::vector<vg::Keypoint>& keypoints,
                                        float processScale);
