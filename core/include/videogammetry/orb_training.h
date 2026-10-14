#pragma once

// Building blocks for learning the BRIEF test pattern offline
// (tools/train_orb_pattern). Not needed by normal pipeline code.

#include <cstdint>
#include <vector>

#include "videogammetry/image.h"

namespace vg {

// One binary test: compare the smoothed intensity at (x1, y1) and (x2, y2),
// offsets from the keypoint before rotation.
struct BriefTest {
    std::int8_t x1, y1, x2, y2;
};

// Maximum offset radius of a test point; rotated points stay inside the
// 15 px orientation patch.
constexpr int kBriefTestRadius = 13;

// `count` candidate tests with both points drawn from an isotropic Gaussian
// (sigma = 31/5), clipped to kBriefTestRadius. Deterministic for a given seed
// on every platform.
std::vector<BriefTest> randomBriefTests(int count, std::uint32_t seed);

// Evaluates one test on a blurred pyramid level at integer keypoint (x, y)
// with orientation given by (cosA, sinA). Returns I(p1) < I(p2).
inline bool evaluateBriefTest(const Image& blurred, int x, int y, float cosA, float sinA,
                              const BriefTest& t) {
    const auto sample = [&](int px, int py) {
        const float rx = cosA * px - sinA * py;
        const float ry = sinA * px + cosA * py;
        const int ix = x + static_cast<int>(rx + (rx >= 0 ? 0.5f : -0.5f));
        const int iy = y + static_cast<int>(ry + (ry >= 0 ? 0.5f : -0.5f));
        return blurred.at(ix, iy);
    };
    return sample(t.x1, t.y1) < sample(t.x2, t.y2);
}

}  // namespace vg
