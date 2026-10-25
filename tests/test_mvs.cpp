// PatchMatch depth maps on synthetic images: a textured plane, slanted 50
// degrees away from the reference camera (plane sweep with fronto-parallel
// patches does badly there), rendered from five cameras.
//  1. Most pixels get a depth, and nearly all of them within 1% of the truth.
//  2. The normals match the plane's.
//  3. Multi-threaded result identical to single-threaded.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "videogammetry/linalg.h"
#include "videogammetry/mvs.h"
#include "videogammetry/thread_pool.h"

namespace {

int failures = 0;
constexpr double kPi = 3.14159265358979323846;

void check(bool ok, const char* what, double value) {
    std::printf("  [%s] %s (%.4g)\n", ok ? "PASS" : "FAIL", what, value);
    if (!ok) ++failures;
}

constexpr int kWidth = 320, kHeight = 240;
const vg::PinholeCamera kCamera{300, 300, 159.5, 119.5};

// Plane n . X = c through (0, 0, 6), turned 50 degrees about the y axis.
struct Plane {
    vg::Vec3 n, u, v;  // normal and in-plane axes
    double c;
};

Plane makePlane() {
    const double a = 50 * kPi / 180;
    Plane p;
    p.n = {std::sin(a), 0, -std::cos(a)};
    p.u = {std::cos(a), 0, std::sin(a)};
    p.v = {0, 1, 0};
    p.c = vg::dot(p.n, vg::Vec3{0, 0, 6});
    return p;
}

// Value noise on a lattice, smoothly interpolated, two octaves.
struct Texture {
    std::vector<double> lattice;
    static constexpr int kSize = 256;
    Texture() {
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> u(0, 1);
        lattice.resize(kSize * kSize);
        for (double& v : lattice) v = u(rng);
    }
    double noise(double x, double y) const {
        const double fx = std::floor(x), fy = std::floor(y);
        const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
        const double ax = x - fx, ay = y - fy;
        const double sx = ax * ax * (3 - 2 * ax), sy = ay * ay * (3 - 2 * ay);
        const auto at = [&](int i, int j) { return lattice[((j & (kSize - 1)) * kSize) + (i & (kSize - 1))]; };
        return (at(ix, iy) * (1 - sx) + at(ix + 1, iy) * sx) * (1 - sy) + (at(ix, iy + 1) * (1 - sx) + at(ix + 1, iy + 1) * sx) * sy;
    }
    double operator()(double s, double t) const {
        return 255 * (0.6 * noise(s * 12 + 100, t * 12 + 100) + 0.4 * noise(s * 31, t * 31));
    }
};

vg::Pose lookingForward(const vg::Vec3& center, double yawDeg) {
    vg::Pose p;
    p.R = vg::rotationFromAxisAngle({0, yawDeg * kPi / 180, 0});
    p.t = -(p.R * center);
    return p;
}

vg::MvsView render(const vg::Pose& pose, const Plane& plane, const Texture& tex) {
    vg::MvsView view;
    view.pose = pose;
    view.gray.resize(kWidth, kHeight);
    const vg::Mat3 Rt = pose.R.transposed();
    const vg::Vec3 c = pose.center();
    for (int y = 0; y < kHeight; ++y)
        for (int x = 0; x < kWidth; ++x) {
            double sum = 0;
            for (int s = 0; s < 4; ++s) {  // 2x2 supersampling
                const double px = x - 0.25 + 0.5 * (s & 1), py = y - 0.25 + 0.5 * (s >> 1);
                const vg::Vec3 d = Rt * vg::Vec3{(px - kCamera.cx) / kCamera.fx, (py - kCamera.cy) / kCamera.fy, 1};
                const double lambda = (plane.c - vg::dot(plane.n, c)) / vg::dot(plane.n, d);
                const vg::Vec3 X = c + d * lambda;
                sum += tex(vg::dot(X, plane.u), vg::dot(X, plane.v));
            }
            view.gray.data[static_cast<std::size_t>(y) * kWidth + x] = static_cast<std::uint8_t>(std::clamp(sum / 4, 0.0, 255.0));
        }
    return view;
}

}  // namespace

int main() {
    const Plane plane = makePlane();
    const Texture tex;
    const vg::MvsView ref = render(lookingForward({0, 0, 0}, 0), plane, tex);
    std::vector<vg::MvsView> others;
    others.push_back(render(lookingForward({-0.6, 0, 0}, 3), plane, tex));
    others.push_back(render(lookingForward({0.6, 0, 0}, -3), plane, tex));
    others.push_back(render(lookingForward({0, -0.4, 0.2}, 0), plane, tex));
    others.push_back(render(lookingForward({0, 0.4, -0.2}, 0), plane, tex));
    std::vector<const vg::MvsView*> sources;
    for (const vg::MvsView& v : others) sources.push_back(&v);

    // A few seeds on the plane, as map points would be.
    std::vector<vg::Vec3> seeds;
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> uni(-1.5, 1.5);
    const vg::Vec3 origin = plane.n * plane.c;
    for (int i = 0; i < 30; ++i) seeds.push_back(origin + plane.u * uni(rng) + plane.v * uni(rng));

    vg::DenseParams params;
    vg::ThreadPool pool(0);
    std::printf("PatchMatch on a plane slanted 50 degrees (%dx%d, %zu sources):\n", kWidth, kHeight, sources.size());
    const vg::DepthMap dm = vg::patchMatch(ref, sources, kCamera, 2.0, 20.0, seeds, params, &pool);

    std::vector<double> errors, angles;
    const vg::Vec3 nCam = ref.pose.R * plane.n;  // facing the camera: n . ray < 0
    const vg::Vec3 nTrue = nCam.z < 0 ? nCam : -nCam;
    for (int y = 0; y < kHeight; ++y)
        for (int x = 0; x < kWidth; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * kWidth + x;
            if (dm.depth[i] <= 0) continue;
            const vg::Vec3 r{(x - kCamera.cx) / kCamera.fx, (y - kCamera.cy) / kCamera.fy, 1};
            const double truth = plane.c / vg::dot(plane.n, r);  // reference camera at the origin, no rotation
            errors.push_back(std::abs(dm.depth[i] - truth) / truth);
            angles.push_back(std::acos(std::clamp(vg::dot(dm.normal[i], nTrue), -1.0, 1.0)) * 180 / kPi);
        }
    const double valid = double(errors.size()) / (kWidth * kHeight);
    check(valid > 0.7, "fraction of pixels with a depth", valid);
    const double within1 = errors.empty() ? 0 : double(std::count_if(errors.begin(), errors.end(), [](double e) { return e < 0.01; })) / errors.size();
    check(within1 > 0.95, "fraction of those within 1% of the true depth", within1);
    std::vector<double> sorted = errors;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted.empty() ? 1 : sorted[sorted.size() / 2];
    check(median < 0.003, "median relative depth error", median);
    std::sort(angles.begin(), angles.end());
    const double medianAngle = angles.empty() ? 180 : angles[angles.size() / 2];
    check(medianAngle < 8, "median normal error, degrees", medianAngle);

    const vg::DepthMap single = vg::patchMatch(ref, sources, kCamera, 2.0, 20.0, seeds, params, nullptr);
    check(single.depth == dm.depth, "single-threaded result identical", 0);

    std::printf("%s\n", failures == 0 ? "all passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
