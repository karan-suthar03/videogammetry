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
    std::vector<Keypoint> keypoints;  // positions undistorted (pinhole) with the current intrinsics
    std::vector<Vec2> rawPositions;   // as detected (distorted), for re-undistortion and BA
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
    // The local map is chosen by covisibility (keyframes sharing points), not
    // by time: the keyframes that see the points tracked in the last frame,
    // then up to `covisibleNeighbours` best covisible keyframes of each, at
    // most `localKeyframes` in all. Their map points are projected.
    int localKeyframes = 30;
    int covisibleNeighbours = 10;
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
    // baseline: among the `triangulationSearch` keyframes sharing the most
    // points with the new one, up to `triangulationKeyframes` (most shared
    // first) whose baseline / median scene depth is at least `minBaselineRatio`.
    int triangulationKeyframes = 4;
    int triangulationSearch = 15;
    double minBaselineRatio = 0.02;
    // Matching for triangulation is guided by the known poses: candidates must
    // lie within `epipolarDistancePx` (times the keypoint's pyramid scale) of
    // the epipolar line, and the best must be unambiguous among them.
    float epipolarDistancePx = 2.0f;
    int triangulationMaxHamming = 50;
    float triangulationRatio = 0.8f;

    // Bundle adjustment.
    int localBAKeyframes = 10;    // refined after each new keyframe: it and its most covisible ones (0 = off)
    int localBAIterations = 10;
    int globalBAIterations = 20;  // full refinement in finish() (0 = off)

    // Focal length self-calibration (the camera is uncalibrated; the field of
    // view option is only a starting guess). Every `focalEveryKeyframes`
    // keyframes, the newest keyframe and its most covisible ones
    // (`focalBAKeyframes` in all) are bundle-adjusted with the focal length
    // free. The result is kept only if the observations alone determine the
    // focal length to `focalMaxStd` (relative); with motion that cannot (e.g.
    // walking straight ahead) the focal length stays as it is. The global BA
    // in finish() refines it the same way.
    bool refineFocal = true;
    int focalEveryKeyframes = 5;
    int focalBAKeyframes = 20;
    double focalMaxStd = 0.01;
    double focalPriorSigma = 0.05;
    // Mapping thread: local and focal bundle adjustment are solved on their
    // own worker (`mappingThreads` threads, 0 = half the hardware threads)
    // while tracking continues, and merged at the first frame boundary after
    // they finish; keyframes added meanwhile wait for the next one. A frame
    // that becomes a keyframe then costs triangulation and fusion only. Off =
    // solved inline on that frame (deterministic, but the frame takes as long
    // as the adjustment).
    bool mappingThread = true;
    int mappingThreads = 0;

    // Background global bundle adjustment: while frames keep being tracked,
    // the whole map (old parts included) is re-optimized on a separate thread
    // with `backgroundBAThreads` workers (0 = half the hardware threads), and
    // the result merged in; keyframes and points created meanwhile move with
    // the keyframe they are attached to. Passes run back to back at low OS
    // priority (only using CPU time tracking leaves over) and refine the focal
    // length and distortion jointly with the whole map (which also repairs
    // parts built with earlier estimates). The fast local refinement runs
    // until a pass agrees with it (both converged), then the passes own the
    // intrinsics.
    // If the motion cannot determine them, the next pass keeps them fixed (and
    // is discarded if the intrinsics changed by > 1% meanwhile). Results are
    // timing dependent; off = fully deterministic.
    bool backgroundBA = true;
    int backgroundBAIterations = 10;
    int backgroundBAMinKeyframes = 10;
    int backgroundBAThreads = 0;

    // Radial distortion (k1) is refined together with the focal length; the
    // result also needs k1 determined to `distortionMaxStd` (absolute).
    bool refineDistortion = true;
    double distortionMaxStd = 0.01;
    double distortionPriorSigma = 0.1;

    // Culling of new points: a point must be seen by at least
    // `minConfirmingKeyframes` keyframes once `probationKeyframes` more
    // keyframes have been added since its creation, or it is removed (a
    // triangulation from one keyframe pair is unconfirmed: often a mismatch).
    int probationKeyframes = 2;
    int minConfirmingKeyframes = 3;

    // Fusion: after triangulation, the new keyframe's points are projected
    // into its `fuseNeighbours` most covisible keyframes (plus up to
    // `fuseSecondOrder` best neighbours of each) and theirs into it. A
    // projection that lands on a matching keypoint becomes an observation, or,
    // if that keypoint already has a point, the two points are merged. This
    // links new keyframes to older points (duplicates and scale drift shrink).
    int fuseNeighbours = 10;
    int fuseSecondOrder = 5;
    float fuseRadiusPx = 3.0f;         // times the predicted pyramid scale
    int fuseMaxHamming = 50;
    float fuseRatio = 0.8f;            // best must beat the second best within the window
    bool fuseMerge = true;             // merge duplicates (otherwise only link free keypoints)
    double maxViewAngleDeg = 60.0;     // from the point's mean viewing direction
};

}  // namespace vg
