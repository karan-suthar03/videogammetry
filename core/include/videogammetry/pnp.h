#pragma once

#include <cstdint>
#include <vector>

#include "videogammetry/linalg.h"
#include "videogammetry/thread_pool.h"

namespace vg {

// Absolute camera pose from 2D-3D correspondences (Perspective-n-Point).
// Works for any camera motion: it needs known 3D points, not parallax.

struct PnPParams {
    double inlierThresholdPx = 3.0;  // reprojection error threshold, pixels
    int maxIterations = 300;
    double confidence = 0.999;
    int minInliers = 15;
    std::uint32_t seed = 4242;
};

struct PnPResult {
    bool ok = false;
    Pose pose;                         // world -> camera
    std::vector<std::uint8_t> inlier;  // per correspondence
    int inliers = 0;
    int iterations = 0;
};

// Minimal solver (Grunert's P3P): poses consistent with three world points
// and their unit bearing vectors. Returns the number of solutions (<= 4).
int solveP3P(const Vec3 bearings[3], const Vec3 worldPoints[3], Pose solutions[4]);

// RANSAC over P3P with MSAC scoring, then Gauss-Newton refinement on the
// inliers. `observations` are normalized image coordinates; `focalPx`
// converts thresholds. If `prior` is given it is scored as an extra
// hypothesis (e.g. a constant-velocity prediction). Deterministic.
PnPResult solvePnPRansac(const std::vector<Vec3>& worldPoints, const std::vector<Vec2>& observations,
                         double focalPx, const PnPParams& params, const Pose* prior = nullptr,
                         ThreadPool* pool = nullptr);

// Gauss-Newton (Levenberg-Marquardt damped) minimization of the Huber-robust
// reprojection error over the 6-DOF pose, using correspondences with
// mask[i] != 0 (all if mask is null). Returns false if it could not solve.
bool refinePose(const std::vector<Vec3>& worldPoints, const std::vector<Vec2>& observations,
                const std::vector<std::uint8_t>* mask, double focalPx, double huberPx, int iterations,
                Pose& pose);

// Reprojection error in pixels; infinity if the point is behind the camera.
double reprojectionErrorPx(const Pose& pose, const Vec3& world, const Vec2& observation, double focalPx);

}  // namespace vg
