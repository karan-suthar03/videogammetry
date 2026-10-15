#pragma once

#include <cstdint>
#include <vector>

#include "videogammetry/linalg.h"
#include "videogammetry/thread_pool.h"

namespace vg {

// Bundle adjustment: jointly refines camera poses and 3D points so that every
// point reprojects onto its observations. Sparse Levenberg-Marquardt with the
// points eliminated by the Schur complement (the reduced system only has the
// cameras), Huber-robust reprojection error.

struct BAObservation {
    int camera = -1;
    int point = -1;
    Vec2 xy;  // normalized image coordinates
    // Expected error scale: the keypoint's pyramid-level scale (coarser levels
    // localize less precisely). Residuals are divided by it, so it weights the
    // observation by 1/sigma^2 and scales the robust loss and outlier test.
    double sigma = 1.0;
};

struct BAProblem {
    std::vector<Pose> poses;                 // world -> camera
    std::vector<std::uint8_t> fixedPose;     // 1 = kept constant (anchors / gauge)
    std::vector<Vec3> points;                // world coordinates
    std::vector<BAObservation> observations;
};

struct BAParams {
    int maxIterations = 10;     // per pass
    double huberPx = 2.0;       // robust loss: quadratic below, linear above (pixels)
    double initialLambda = 1e-3;
    // Two-pass outlier handling (as in ORB-SLAM): after a robust first pass,
    // observations with a larger error are rejected and the problem is solved
    // again without them. The Huber loss alone still lets wrong observations
    // bias the result by a few pixels. <= 0 disables the second pass.
    double outlierThresholdPx = 2.5;
};

struct BAReport {
    bool ok = false;
    int iterations = 0;
    double initialRmsPx = 0;  // RMS reprojection error over all observations, before
    double finalRmsPx = 0;    // ... and after, over the observations kept as inliers
    int outliers = 0;
    std::vector<std::uint8_t> inlier;  // per observation (1 = kept)
};

// Refines the non-fixed poses and all points in place. `focalPx` converts
// normalized units to pixels. Deterministic with or without a pool.
BAReport bundleAdjust(BAProblem& problem, double focalPx, const BAParams& params, ThreadPool* pool = nullptr);

}  // namespace vg
