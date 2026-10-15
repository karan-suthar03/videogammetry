#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "videogammetry/linalg.h"
#include "videogammetry/orb.h"

namespace vg {

// A keyframe's keypoint that sees a map point.
struct PointObservation {
    int keyframe = -1;
    int keypoint = -1;
};

struct MapPoint {
    Vec3 position;          // world coordinates
    Descriptor descriptor;  // from the most recent observation
    int level = 0;          // pyramid level of that observation
    std::array<std::uint8_t, 3> color{255, 255, 255};
    std::vector<PointObservation> observations;  // one per keyframe that sees it
    int firstKeyframe = 0;  // keyframe whose insertion created it
    bool bad = false;       // rejected; detached from all keyframes and ignored everywhere
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
    int localKeyframes = 20;         // map points of this many recent keyframes are projected
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
    // New points are triangulated against earlier keyframes with enough
    // baseline: among the last `triangulationSearch` keyframes, up to
    // `triangulationKeyframes` most recent ones whose baseline / median scene
    // depth is at least `minBaselineRatio` (enough parallax, most shared view).
    int triangulationKeyframes = 4;
    int triangulationSearch = 15;
    double minBaselineRatio = 0.02;

    // Bundle adjustment.
    int localBAKeyframes = 10;    // keyframes refined after each new keyframe (0 = off)
    int localBAIterations = 10;
    int globalBAIterations = 20;  // full refinement in finish() (0 = off)

    // Culling of new points: a point must be seen by at least
    // `minConfirmingKeyframes` keyframes once `probationKeyframes` more
    // keyframes have been added since its creation, or it is removed (a
    // triangulation from one keyframe pair is unconfirmed: often a mismatch).
    int probationKeyframes = 2;
    int minConfirmingKeyframes = 3;
};

}  // namespace vg
