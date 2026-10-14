#pragma once

#include <cstdint>
#include <vector>

#include "videogammetry/linalg.h"
#include "videogammetry/thread_pool.h"

namespace vg {

// Pinhole camera, in pixels of the image the keypoints live in.
struct CameraIntrinsics {
    double fx = 0, fy = 0, cx = 0, cy = 0;

    // Square pixels, principal point at the image center, given the field of
    // view across the long side (the same for portrait and landscape video
    // from one camera).
    static CameraIntrinsics fromFov(int width, int height, double longSideFovDeg);

    Vec2 normalize(double u, double v) const { return {(u - cx) / fx, (v - cy) / fy}; }
};

struct TwoViewParams {
    double inlierThresholdPx = 1.5;  // Sampson error threshold, in pixels
    int maxIterations = 1000;
    double confidence = 0.999;
    int minInliers = 30;
    std::uint32_t seed = 12345;
};

// Relative pose of view 2 w.r.t. view 1: X2 = R * X1 + t, |t| = 1.
// Epipolar constraint for normalized points: x2^T E x1 = 0, E = [t]x R.
struct TwoViewResult {
    bool ok = false;
    Mat3 E;
    Mat3 R = Mat3::identity();
    Vec3 t;
    std::vector<std::uint8_t> inlier;  // per input correspondence
    int inliers = 0;
    int inFront = 0;                   // inliers triangulated in front of both cameras
    double medianParallaxDeg = 0;      // median triangulation angle of those points
    int iterations = 0;
};

// Estimates the essential matrix with RANSAC (normalized 8-point algorithm,
// MSAC scoring on Sampson error) and recovers (R, t) by cheirality.
// Points are normalized image coordinates (see CameraIntrinsics::normalize);
// `focalPx` converts the pixel threshold. Deterministic for a given seed,
// with or without a pool.
TwoViewResult estimateRelativePose(const std::vector<Vec2>& points1, const std::vector<Vec2>& points2,
                                   double focalPx, const TwoViewParams& params,
                                   ThreadPool* pool = nullptr);

// Linear (DLT) triangulation of normalized points from cameras [I|0] and [R|t].
// Returns false if the point is at infinity.
bool triangulate(const Vec2& x1, const Vec2& x2, const Mat3& R, const Vec3& t, Vec3& X);

// Linear (normalized 8-point) essential matrix from the correspondences at
// `indices` (n >= 8), projected to singular values (1, 1, 0).
bool essentialLinear(const std::vector<Vec2>& points1, const std::vector<Vec2>& points2,
                     const int* indices, int n, Mat3& E);

// Squared Sampson distance of a correspondence to E (normalized units).
double sampsonError(const Mat3& E, const Vec2& x1, const Vec2& x2);

}  // namespace vg
