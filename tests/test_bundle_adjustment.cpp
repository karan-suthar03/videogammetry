// Bundle adjustment on a synthetic scene with known ground truth:
// perturbed cameras and points must return to the truth (up to noise),
// with 5% wrong observations handled by the robust loss, and the
// multi-threaded solve must equal the single-threaded one.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "videogammetry/bundle_adjustment.h"
#include "videogammetry/linalg.h"
#include "videogammetry/thread_pool.h"

namespace {

int failures = 0;
constexpr double kPi = 3.14159265358979323846;

void check(bool ok, const char* what, double value) {
    std::printf("  [%s] %s (%.4g)\n", ok ? "PASS" : "FAIL", what, value);
    if (!ok) ++failures;
}

struct Scene {
    std::vector<vg::Pose> truePoses;
    std::vector<vg::Vec3> truePoints;
    vg::BAProblem problem;
    std::vector<bool> outlier;
};

Scene makeScene(double focal) {
    Scene s;
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u(-1, 1);
    std::normal_distribution<double> noise(0, 0.5 / focal);

    // Cameras on a curved path, turning as they go (any motion is fine here).
    for (int i = 0; i < 15; ++i) {
        vg::Pose p;
        p.R = vg::rotationFromAxisAngle({0.02 * i, 0.06 * i, 0.01 * std::sin(i)});
        const vg::Vec3 center{0.4 * i, 0.05 * std::sin(0.7 * i), 0.15 * i};
        p.t = -(p.R * center);
        s.truePoses.push_back(p);
    }
    for (int j = 0; j < 600; ++j) s.truePoints.push_back({4 * u(rng) + 3, 2 * u(rng), 8 + 4 * u(rng)});

    std::uniform_real_distribution<double> u01(0, 1);
    for (int j = 0; j < static_cast<int>(s.truePoints.size()); ++j) {
        for (int i = 0; i < static_cast<int>(s.truePoses.size()); ++i) {
            const vg::Vec3 c = s.truePoses[i].transform(s.truePoints[j]);
            if (c.z <= 0.5) continue;
            vg::Vec2 xy{c.x / c.z + noise(rng), c.y / c.z + noise(rng)};
            if (std::abs(xy.x) > 0.7 || std::abs(xy.y) > 0.45) continue;  // outside the image
            const bool bad = u01(rng) < 0.05;
            if (bad) xy = {0.7 * u(rng), 0.45 * u(rng)};
            s.problem.observations.push_back({i, j, xy});
            s.outlier.push_back(bad);
        }
    }

    // Start from a perturbed state; the first two cameras are fixed anchors
    // (they fix position, orientation and scale of the reconstruction).
    std::normal_distribution<double> rot(0, 0.01), pos(0, 0.05), pt(0, 0.15);
    s.problem.poses = s.truePoses;
    s.problem.fixedPose.assign(s.truePoses.size(), 0);
    s.problem.fixedPose[0] = s.problem.fixedPose[1] = 1;
    for (std::size_t i = 2; i < s.problem.poses.size(); ++i) {
        vg::Pose& p = s.problem.poses[i];
        p.R = vg::rotationFromAxisAngle({rot(rng), rot(rng), rot(rng)}) * p.R;
        p.t = p.t + vg::Vec3{pos(rng), pos(rng), pos(rng)};
    }
    s.problem.points = s.truePoints;
    for (vg::Vec3& X : s.problem.points) X = X + vg::Vec3{pt(rng), pt(rng), pt(rng)};
    return s;
}

}  // namespace

int main() {
    const double focal = 800;
    Scene s = makeScene(focal);
    std::printf("bundle adjustment: %zu cameras, %zu points, %zu observations\n", s.problem.poses.size(),
                s.problem.points.size(), s.problem.observations.size());

    vg::BAParams params;
    params.maxIterations = 30;
    vg::BAProblem single = s.problem;
    const vg::BAReport r = vg::bundleAdjust(single, focal, params);
    std::printf("  RMS reprojection %.2f px -> %.2f px (inliers) in %d iterations, %d outliers rejected\n",
                r.initialRmsPx, r.finalRmsPx, r.iterations, r.outliers);

    // Error on good observations that were kept (outliers stay large by design).
    double sum = 0;
    int n = 0;
    for (std::size_t o = 0; o < single.observations.size(); ++o) {
        // Good observations the solver kept (rejected ones are checked separately below).
        if (s.outlier[o] || !r.inlier[o]) continue;
        const vg::BAObservation& ob = single.observations[o];
        const vg::Vec3 c = single.poses[ob.camera].transform(single.points[ob.point]);
        sum += std::pow(c.x / c.z - ob.xy.x, 2) + std::pow(c.y / c.z - ob.xy.y, 2);
        ++n;
    }
    const double inlierRms = std::sqrt(sum / n) * focal;
    double worstRot = 0, worstPos = 0, worstPoint = 0;
    for (std::size_t i = 0; i < single.poses.size(); ++i) {
        worstRot = std::max(worstRot, vg::rotationAngle(single.poses[i].R * s.truePoses[i].R.transposed()) * 180 / kPi);
        worstPos = std::max(worstPos, vg::norm(single.poses[i].center() - s.truePoses[i].center()));
    }
    std::vector<double> pointErr;
    for (std::size_t j = 0; j < single.points.size(); ++j) pointErr.push_back(vg::norm(single.points[j] - s.truePoints[j]));
    std::nth_element(pointErr.begin(), pointErr.begin() + pointErr.size() / 2, pointErr.end());
    worstPoint = pointErr[pointErr.size() / 2];

    std::printf("checks:\n");
    check(r.ok && r.finalRmsPx < r.initialRmsPx, "reprojection error decreased", r.finalRmsPx);
    check(inlierRms < 0.9, "inlier RMS at noise level (< 0.9 px for 0.5 px noise)", inlierRms);
    check(worstRot < 0.1, "worst camera rotation error < 0.1 deg (start: ~1 deg)", worstRot);
    check(worstPos < 0.02, "worst camera position error < 0.02 (start: ~0.09)", worstPos);
    check(worstPoint < 0.05, "median point error < 0.05 (start: ~0.26)", worstPoint);
    int caught = 0, wrongly = 0, outliers = 0;
    for (std::size_t o = 0; o < s.outlier.size(); ++o) {
        outliers += s.outlier[o];
        if (!r.inlier[o]) (s.outlier[o] ? caught : wrongly)++;
    }
    check(caught >= 0.9 * outliers, "at least 90% of wrong observations rejected", double(caught) / outliers);
    check(wrongly <= 0.02 * (s.outlier.size() - outliers), "at most 2% of good observations rejected", wrongly);

    vg::ThreadPool pool(8);
    vg::BAProblem parallel = s.problem;
    vg::bundleAdjust(parallel, focal, params, &pool);
    bool identical = true;
    for (std::size_t i = 0; i < parallel.poses.size(); ++i) {
        identical = identical && parallel.poses[i].R.m == single.poses[i].R.m && parallel.poses[i].t.x == single.poses[i].t.x;
    }
    for (std::size_t j = 0; j < parallel.points.size(); ++j) {
        identical = identical && parallel.points[j].x == single.points[j].x;
    }
    check(identical, "8-thread result identical to single-threaded", identical);

    // Focal length: the same scene seen with a focal 20% too short. Normalizing
    // pixels with the wrong focal scales every observation by 800/640.
    const double wrongFocal = 640;
    vg::BAProblem focalProblem = s.problem;
    for (vg::BAObservation& ob : focalProblem.observations) ob.xy = {ob.xy.x * focal / wrongFocal, ob.xy.y * focal / wrongFocal};
    vg::BAParams focalParams = params;
    focalParams.refineFocal = true;
    const vg::BAReport fr = vg::bundleAdjust(focalProblem, wrongFocal, focalParams, &pool);
    const double recovered = fr.focalScale * wrongFocal;
    std::printf("  focal: start %.0f px -> %.1f px (true %.0f), data-only std %.2f%%, RMS %.2f -> %.2f px\n", wrongFocal,
                recovered, focal, 100 * fr.focalStd, fr.initialRmsPx, fr.finalRmsPx);
    check(std::abs(recovered - focal) < 0.01 * focal, "focal recovered within 1% (start: 20% off)", recovered);
    check(fr.focalStd < 0.02, "focal well determined by this motion (std < 2%)", fr.focalStd);

    std::printf("%s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
