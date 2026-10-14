// ORB sanity checks on a synthetic image:
//  1. Rotating the image by 90 degrees finds the same corners, with
//     orientation shifted by 90 degrees and near-identical descriptors.
//  2. Descriptors of unrelated keypoints differ by roughly half their bits.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "videogammetry/image.h"
#include "videogammetry/orb.h"

namespace {

int failures = 0;

void check(bool ok, const char* what, double value) {
    std::printf("  [%s] %s (%.3f)\n", ok ? "PASS" : "FAIL", what, value);
    if (!ok) ++failures;
}

// Natural-looking texture: rectangles and discs filled with smoothed noise
// (flat-colored shapes make every corner look alike after steering).
vg::Image makeTexture(int size) {
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> intensity(0, 255);
    std::uniform_int_distribution<int> pos(0, size - 1);
    std::uniform_int_distribution<int> extent(6, 40);

    // Smoothed noise used as the fill texture.
    vg::Image noise, smooth;
    noise.resize(size, size);
    for (auto& p : noise.data) p = static_cast<std::uint8_t>(intensity(rng));
    vg::gaussianBlur7(noise, smooth);

    vg::Image img;
    img.resize(size, size);
    for (std::size_t i = 0; i < img.data.size(); ++i) {
        img.data[i] = static_cast<std::uint8_t>(std::clamp(128 + 3 * (smooth.data[i] - 128), 0, 255));
    }
    for (int i = 0; i < 400; ++i) {
        const int cx = pos(rng), cy = pos(rng), r = extent(rng);
        const int base = intensity(rng);
        const bool disc = i % 2 == 0;
        for (int y = std::max(0, cy - r); y < std::min(size, cy + r); ++y) {
            for (int x = std::max(0, cx - r); x < std::min(size, cx + r); ++x) {
                if (!disc || (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) {
                    const int t = 2 * (smooth.at((x * 7) % size, (y * 7) % size) - 128);
                    img.row(y)[x] = static_cast<std::uint8_t>(std::clamp(base + t, 0, 255));
                }
            }
        }
    }
    vg::Image blurred;
    vg::gaussianBlur7(img, blurred);
    return blurred;
}

// Rotates 90 degrees clockwise: (x, y) -> (H - 1 - y, x).
vg::Image rotate90(const vg::Image& src) {
    vg::Image dst;
    dst.resize(src.height, src.width);
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            dst.row(x)[src.height - 1 - y] = src.at(x, y);
        }
    }
    return dst;
}

double angleDiff(double a, double b) {
    double d = std::fmod(a - b, 2 * 3.14159265358979);
    if (d > 3.14159265358979) d -= 2 * 3.14159265358979;
    if (d < -3.14159265358979) d += 2 * 3.14159265358979;
    return d;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

}  // namespace

int main() {
    const vg::Image a = makeTexture(640);
    const vg::Image b = rotate90(a);

    vg::OrbParams params;
    params.maxFeatures = 1000;
    vg::OrbExtractor orb(params);

    std::vector<vg::Keypoint> ka, kb;
    std::vector<vg::Descriptor> da, db;
    orb.extract(a, ka, da);
    orb.extract(b, kb, db);
    std::printf("keypoints: %zu original, %zu rotated\n", ka.size(), kb.size());

    // Pair each original keypoint with the rotated keypoint at its expected position.
    std::vector<double> distances, angleErrors;
    for (std::size_t i = 0; i < ka.size(); ++i) {
        const float ex = (a.height - 1) - ka[i].y;
        const float ey = ka[i].x;
        int best = -1;
        float bestD2 = 1.5f * 1.5f * ka[i].scale * ka[i].scale;
        for (std::size_t j = 0; j < kb.size(); ++j) {
            if (kb[j].level != ka[i].level) continue;
            const float dx = kb[j].x - ex, dy = kb[j].y - ey;
            const float d2 = dx * dx + dy * dy;
            if (d2 <= bestD2) {
                bestD2 = d2;
                best = static_cast<int>(j);
            }
        }
        if (best < 0) continue;
        distances.push_back(vg::hammingDistance(da[i], db[best]));
        angleErrors.push_back(std::abs(angleDiff(kb[best].angle - ka[i].angle, 3.14159265358979 / 2)));
    }

    // Unrelated pairs: keypoints at least 100 px apart (neighbors share texture).
    std::vector<double> randomDistances;
    std::mt19937 rng(7);
    for (int n = 0; n < 5000 && !ka.empty(); ++n) {
        const std::size_t i = rng() % ka.size(), j = rng() % ka.size();
        const float dx = ka[i].x - ka[j].x, dy = ka[i].y - ka[j].y;
        if (dx * dx + dy * dy < 100.0f * 100.0f) continue;
        randomDistances.push_back(vg::hammingDistance(da[i], da[j]));
    }

    const double repeat = ka.empty() ? 0 : double(distances.size()) / ka.size();
    std::printf("checks:\n");
    check(ka.size() >= 500, "found at least 500 keypoints", double(ka.size()));
    check(repeat >= 0.6, "at least 60% of keypoints re-detected after rotation", repeat);
    check(median(angleErrors) < 0.1, "median orientation error < 0.1 rad", median(angleErrors));
    check(median(distances) < 30, "median matched descriptor distance < 30 / 256 bits", median(distances));
    // Smoke check only (catches constant/degenerate descriptors). Synthetic patches are
    // smoother than real ones; real-footage quality is measured separately (~116 median).
    check(median(randomDistances) > 60, "median unrelated descriptor distance > 60 / 256 bits", median(randomDistances));

    std::printf("%s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
