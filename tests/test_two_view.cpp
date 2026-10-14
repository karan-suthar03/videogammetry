// Two-view geometry on a synthetic scene with known ground truth:
//  - linear algebra building blocks (eigen solver, SVD),
//  - relative pose recovery under noise and 30% outliers, for sideways and
//    forward motion (forward = our walking video),
//  - multi-threaded RANSAC gives exactly the single-threaded result.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "videogammetry/linalg.h"
#include "videogammetry/thread_pool.h"
#include "videogammetry/two_view.h"

namespace {

int failures = 0;
constexpr double kPi = 3.14159265358979323846;

void check(bool ok, const char* what, double value) {
    std::printf("  [%s] %s (%.4g)\n", ok ? "PASS" : "FAIL", what, value);
    if (!ok) ++failures;
}

vg::Mat3 rotationXYZ(double ax, double ay, double az) {
    const auto rx = [](double a) { vg::Mat3 r; r.m = {1, 0, 0, 0, std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a)}; return r; };
    const auto ry = [](double a) { vg::Mat3 r; r.m = {std::cos(a), 0, std::sin(a), 0, 1, 0, -std::sin(a), 0, std::cos(a)}; return r; };
    const auto rz = [](double a) { vg::Mat3 r; r.m = {std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a), 0, 0, 0, 1}; return r; };
    return rz(az) * ry(ay) * rx(ax);
}

void testLinalg() {
    std::printf("linalg:\n");
    const vg::Mat3 a = rotationXYZ(0.3, -0.2, 0.9) * [] { vg::Mat3 d; d.m = {5, 0, 0, 0, 2, 0, 0, 0, 0.5}; return d; }() *
                       rotationXYZ(-0.7, 0.4, 0.1).transposed();
    vg::Mat3 u, v;
    vg::Vec3 s;
    vg::svd3(a, u, s, v);
    vg::Mat3 d;
    d.m = {s.x, 0, 0, 0, s.y, 0, 0, 0, s.z};
    const vg::Mat3 back = u * d * v.transposed();
    double err = 0;
    for (int i = 0; i < 9; ++i) err = std::max(err, std::abs(back.m[i] - a.m[i]));
    check(err < 1e-10, "svd3 reconstructs A", err);
    check(std::abs(s.x - 5) < 1e-10 && std::abs(s.y - 2) < 1e-10 && std::abs(s.z - 0.5) < 1e-10,
          "svd3 singular values 5, 2, 0.5", s.x + s.y + s.z);
}

struct Scenario {
    const char* name;
    vg::Mat3 R;
    vg::Vec3 t;
};

void testPose(const Scenario& sc, vg::ThreadPool& pool) {
    std::printf("%s:\n", sc.name);
    const double focal = 800, width = 960, height = 540;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> ux(-4, 4), uy(-2, 2), uz(4, 20);
    std::normal_distribution<double> noise(0, 0.5);  // pixels
    std::uniform_real_distribution<double> anyX(0, width), anyY(0, height);
    std::uniform_real_distribution<double> unit(0, 1);

    std::vector<vg::Vec2> p1, p2;
    std::vector<bool> isOutlier;
    while (p1.size() < 800) {
        const vg::Vec3 X{ux(rng), uy(rng), uz(rng)};
        const vg::Vec3 X2 = sc.R * X + sc.t;
        if (X2.z <= 0.5) continue;
        const double u1 = focal * X.x / X.z + width / 2, v1 = focal * X.y / X.z + height / 2;
        double u2 = focal * X2.x / X2.z + width / 2, v2 = focal * X2.y / X2.z + height / 2;
        if (u1 < 0 || u1 >= width || v1 < 0 || v1 >= height || u2 < 0 || u2 >= width || v2 < 0 || v2 >= height) continue;
        const bool outlier = unit(rng) < 0.3;
        if (outlier) {
            u2 = anyX(rng);
            v2 = anyY(rng);
        }
        p1.push_back({(u1 + noise(rng) - width / 2) / focal, (v1 + noise(rng) - height / 2) / focal});
        p2.push_back({(u2 + noise(rng) - width / 2) / focal, (v2 + noise(rng) - height / 2) / focal});
        isOutlier.push_back(outlier);
    }

    vg::TwoViewParams params;
    const vg::TwoViewResult r = vg::estimateRelativePose(p1, p2, focal, params);

    int truePos = 0, falsePos = 0, outliers = 0;
    for (std::size_t i = 0; i < p1.size(); ++i) {
        outliers += isOutlier[i];
        if (r.inlier[i]) (isOutlier[i] ? falsePos : truePos)++;
    }
    const double rotErrDeg = vg::rotationAngle(r.R * sc.R.transposed()) * 180 / kPi;
    const double tErrDeg = std::acos(std::min(1.0, vg::dot(vg::normalized(r.t), vg::normalized(sc.t)))) * 180 / kPi;
    std::printf("  inliers %d (true %d, false %d of %d outliers), iterations %d, parallax %.2f deg\n",
                r.inliers, truePos, falsePos, outliers, r.iterations, r.medianParallaxDeg);

    check(r.ok, "pose recovered", r.ok);
    check(rotErrDeg < 0.2, "rotation error < 0.2 deg", rotErrDeg);
    check(tErrDeg < 2.0, "translation direction error < 2 deg", tErrDeg);
    check(falsePos <= outliers / 50, "at most 2% of outliers accepted", falsePos);
    check(truePos >= 0.9 * (p1.size() - outliers), "at least 90% of true matches kept", truePos);

    bool identical = true;
    for (int run = 0; run < 3; ++run) {
        const vg::TwoViewResult rp = vg::estimateRelativePose(p1, p2, focal, params, &pool);
        identical = identical && rp.inlier == r.inlier && rp.R.m == r.R.m && rp.iterations == r.iterations;
    }
    check(identical, "8-thread RANSAC identical to single-threaded (3 runs)", identical);
}

}  // namespace

int main() {
    testLinalg();
    vg::ThreadPool pool(8);
    testPose({"sideways motion", rotationXYZ(0.02, -0.05, 0.01), vg::normalized({1, 0.1, 0.05})}, pool);
    testPose({"forward motion (walking)", rotationXYZ(0.01, 0.03, -0.005), vg::normalized({0.05, -0.02, 1})}, pool);
    std::printf("%s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
