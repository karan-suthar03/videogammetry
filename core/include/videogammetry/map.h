#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "videogammetry/linalg.h"
#include "videogammetry/orb.h"

namespace vg {

struct MapPoint {
    Vec3 position;          // world coordinates
    Descriptor descriptor;  // from the most recent observation
    int level = 0;          // pyramid level of that observation
    std::array<std::uint8_t, 3> color{255, 255, 255};
    int observations = 0;   // keyframes that see this point
};

struct Keyframe {
    long frameIndex = 0;  // index of the video frame
    Pose pose;            // world -> camera
    std::vector<Keypoint> keypoints;
    std::vector<Descriptor> descriptors;
    std::vector<std::array<std::uint8_t, 3>> colors;  // RGB at each keypoint
    std::vector<int> pointIds;                         // map point per keypoint, -1 if none
};

struct MappingParams {
    // Initialization (needs some parallax; the camera can move in any direction).
    double minInitParallaxDeg = 1.0;
    int minInitialPoints = 100;
    int maxReferenceAge = 60;  // frames before the initialization reference is replaced

    // Tracking against the map (any motion: uses 2D-3D matches, no parallax needed).
    int localKeyframes = 10;         // map points of this many recent keyframes are projected
    float projectionRadiusPx = 15;   // search radius around projected map points
    float wideRadiusPx = 60;         // retry radius when few matches (fast motion)
    int minTrackedPoints = 30;
    int maxLostFrames = 25;          // then start a new map

    // Keyframes: inserted when the view has changed enough.
    double keyframeTrackedRatio = 0.7;  // tracked points below this share of the last keyframe's
    int minKeyframeGap = 2;
    int maxKeyframeGap = 30;

    // Triangulation of new points between keyframes.
    double minTriangulationAngleDeg = 1.0;
    double maxReprojectionErrorPx = 2.0;
    int triangulationKeyframes = 2;  // previous keyframes to triangulate against
};

}  // namespace vg
