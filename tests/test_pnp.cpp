// PnP checks with arbitrary camera orientations (any rotation about any
// axis, any translation direction): no assumption about how the camera moves.
//  1. P3P is exact on noise-free random triples.
//  2. RANSAC + refinement recovers random poses under noise and 40% outliers.
//  3. Multi-threaded result identical to single-threaded.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "videogammetry/linalg.h"
#include "videogammetry/pnp.h"
#include "videogammetry/thread_pool.h"

namespace {

int failures = 0;
constexpr double kPi = 3.14159265358979323846;

void check(bool ok, const char* what, double value) {
    std::printf("  [%s] %s (%.4g)\n", ok ? "PASS" : "FAIL", what, value);
    if (!ok) ++failures;
}

vg::Mat3 randomRotation(std::mt19937& rng) {
    std::normal_distribution<double> n(0, 1);
    const vg::Vec3 axis = vg::normalized({n(rng), n(rng), n(rng)});
    std::uniform_real_distribution<double> angle(0, kPi);
    return vg::rotationFromAxisAngle(axis * angle(rng));
}

// Random camera pose looking at random points: points are generated in the
// camera frame (in front, inside a 70 deg field of view) and moved to world.
struct Scene {
    vg::Pose pose;
    std::vector<vg::Vec3> world;
    std::vector<vg::Vec2> obs;
};

Scene makeScene(std::mt19937& rng, int n, double noisePx, double outlierRate, double focal) {
    Scene s;
    s.pose.R = randomRotation(rng);
    std::uniform_real_distribution<double> ut(-10, 10), depth(2, 30), fov(-0.7, 0.7), u01(0, 1);
    s.pose.t = {ut(rng), ut(rng), ut(rng)};
    std::normal_distribution<double> noise(0, noisePx / focal);
    const vg::Pose inv = s.pose.inverse();
    for (int i = 0; i < n; ++i) {
        const double z = depth(rng);
        const vg::Vec3 c{fov(rng) * z, fov(rng) * 0.6 * z, z};
        s.world.push_back(inv.transform(c));
        vg::Vec2 o{c.x / z + noise(rng), c.y / z + noise(rng)};
        if (u01(rng) < outlierRate) o = {fov(rng), fov(rng) * 0.6};
        s.obs.push_back(o);
    }
    return s;
}

double rotationErrorDeg(const vg::Pose& a, const vg::Pose& b) {
    return vg::rotationAngle(a.R * b.R.transposed()) * 180 / kPi;
}

}  // namespace

int main() {
    std::mt19937 rng(2024);
    const double focal = 800;

    std::printf("P3P exactness (500 random noise-free triples, random poses):\n");
    {
        int solved = 0;
        double worst = 0;
        for (int k = 0; k < 500; ++k) {
            const Scene s = makeScene(rng, 3, 0, 0, focal);
            vg::Vec3 bearings[3], points[3];
            for (int i = 0; i < 3; ++i) {
                bearings[i] = {s.obs[i].x, s.obs[i].y, 1};
                points[i] = s.world[i];
            }
            vg::Pose sols[4];
            const int ns = vg::solveP3P(bearings, points, sols);
            double bestErr = 1e9;
            for (int i = 0; i < ns; ++i) {
                const double e = rotationErrorDeg(sols[i], s.pose) + vg::norm(sols[i].t - s.pose.t);
                bestErr = std::min(bestErr, e);
            }
            if (bestErr < 1e-4) ++solved;
            else worst = std::max(worst, bestErr);
        }
        check(solved >= 495, "true pose among P3P solutions for >= 99% of triples", solved);
    }

    std::printf("RANSAC PnP (50 random poses, 300 points, 0.5 px noise, 40%% outliers):\n");
    vg::ThreadPool pool(8);
    {
        int ok = 0, identical = 0;
        double worstRot = 0, worstPos = 0;
        for (int k = 0; k < 50; ++k) {
            const Scene s = makeScene(rng, 300, 0.5, 0.4, focal);
            vg::PnPParams params;
            params.seed = 100 + k;
            const vg::PnPResult r = vg::solvePnPRansac(s.world, s.obs, focal, params);
            if (!r.ok) continue;
            ++ok;
            worstRot = std::max(worstRot, rotationErrorDeg(r.pose, s.pose));
            worstPos = std::max(worstPos, vg::norm(r.pose.center() - s.pose.center()));
            const vg::PnPResult rp = vg::solvePnPRansac(s.world, s.obs, focal, params, nullptr, &pool);
            if (rp.inlier == r.inlier && rp.pose.R.m == r.pose.R.m && rp.pose.t.x == r.pose.t.x) ++identical;
        }
        check(ok == 50, "all 50 poses solved", ok);
        check(worstRot < 0.2, "worst rotation error < 0.2 deg", worstRot);
        check(worstPos < 0.1, "worst camera position error < 0.1 (scene depth 2-30)", worstPos);
        check(identical == ok, "8-thread result identical to single-threaded", identical);
    }

    std::printf("%s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
