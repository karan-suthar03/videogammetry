#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "videogammetry/frame.h"
#include "videogammetry/matcher.h"
#include "videogammetry/mvs.h"
#include "videogammetry/linalg.h"
#include "videogammetry/orb.h"
#include "videogammetry/point_cloud.h"

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
// With `inlier` flags (parallel to matches): inliers green, outliers red;
// otherwise colored by descriptor distance.
RgbImage drawFlow(const RgbImage& image, const std::vector<vg::Keypoint>& trainKeypoints,
                  const std::vector<vg::Keypoint>& queryKeypoints,
                  const std::vector<vg::Match>& matches, int maxDistance,
                  const std::vector<std::uint8_t>* inlier = nullptr);

// Draws the frame's keypoints: green if tracked to a map point (trackedIds[i]
// >= 0), red otherwise. Keypoints are in processed pixels.
RgbImage drawTracking(const vg::FrameView& frame, const std::vector<vg::Keypoint>& keypoints,
                      const std::vector<int>& trackedIds, float processScale);

// Renders the map from above (X right, Z up) and from the side (Z right,
// -Y up), side by side. Points keep their color; the camera path is drawn
// in red with the start marked green. Bounds ignore the farthest 2% of points.
RgbImage renderMapViews(const std::vector<vg::Point>& points, const std::vector<vg::Vec3>& cameraPath,
                        int panelSize);

// Range of a depth map's inverse depth for coloring: its 2nd and 98th
// percentiles. False when it has no depth.
bool inverseDepthRange(const vg::DepthMap& dm, float& lo, float& hi);

// A depth map as colors: inverse depth from far (blue) to near (red) over
// [lo, hi] (inverse depth; both 0 = inverseDepthRange()), no depth black.
RgbImage colorizeDepth(const vg::DepthMap& dm, float lo = 0, float hi = 0);
// Normals as colors (x, y, z mapped to r, g, b), no depth black.
RgbImage colorizeNormals(const vg::DepthMap& dm);
// Matching cost (1 - NCC) from 0 (white) to `maxCost` and over (black).
RgbImage colorizeCost(const vg::DepthMap& dm, float maxCost);

// Writes an RGB8 frame. Format is chosen by extension: .bmp, otherwise binary .ppm.
bool writeImage(const vg::FrameView& frame, const std::string& path);

// Copies an RGB8 frame and draws keypoints on it: a circle sized by pyramid
// level (color = level) and a line showing the orientation. Keypoint
// coordinates are in processed-image pixels; processScale maps them to the frame.
std::vector<std::uint8_t> drawKeypoints(const vg::FrameView& frame,
                                        const std::vector<vg::Keypoint>& keypoints,
                                        float processScale);
