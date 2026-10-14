#pragma once

#include <cstdint>
#include <vector>

#include "videogammetry/linalg.h"
#include "videogammetry/orb.h"

namespace vg {

struct MapPoint {
    Vec3 position;  // world coordinates
    std::uint8_t r = 255, g = 255, b = 255;
    int observations = 0;  // keyframes that see this point
};

struct Keyframe {
    long frameIndex = 0;  // index of the video frame
    Pose pose;
    std::vector<Keypoint> keypoints;
    std::vector<Descriptor> descriptors;
    std::vector<std::array<std::uint8_t, 3>> colors;  // RGB at each keypoint
    std::vector<int> pointIds;                         // map point per keypoint, -1 if none
};

struct MappingParams {
    double minKeyframeParallaxDeg = 1.0;  // new keyframe once median parallax reaches this
    int maxKeyframeGap = 30;              // ... or after this many frames
    double minTriangulationAngleDeg = 1.0;
    double maxReprojectionErrorPx = 2.0;
    int minInitialPoints = 100;           // points needed to accept the initial pair
    int minScalePoints = 15;              // shared map points needed to propagate scale
};

}  // namespace vg
