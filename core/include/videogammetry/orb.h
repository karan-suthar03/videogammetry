#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

#include "videogammetry/image.h"
#include "videogammetry/thread_pool.h"

namespace vg {

struct Keypoint {
    float x = 0, y = 0;    // position in level-0 (input image) pixels
    float angle = 0;       // orientation in radians, (-pi, pi]
    float response = 0;    // Harris corner score (higher is stronger)
    int level = 0;         // pyramid level it was detected on
    float scale = 1;       // level scale relative to level 0
};

// 256-bit binary descriptor.
using Descriptor = std::array<std::uint64_t, 4>;

inline int hammingDistance(const Descriptor& a, const Descriptor& b) {
    return std::popcount(a[0] ^ b[0]) + std::popcount(a[1] ^ b[1]) +
           std::popcount(a[2] ^ b[2]) + std::popcount(a[3] ^ b[3]);
}

struct OrbParams {
    int maxFeatures = 2000;      // total across all levels
    float scaleFactor = 1.2f;    // between consecutive pyramid levels
    int levels = 8;
    int fastThreshold = 20;      // FAST intensity threshold
    int minFastThreshold = 7;    // retried in grid cells where nothing was found
    // Extra share of features for the middle of the image: a grid cell at the
    // center gets (1 + centerBoost) times the features of a cell at the edge,
    // falling off smoothly (Gaussian). 0 = spread evenly.
    float centerBoost = 0.0f;
};

// Building blocks shared with other implementations of the same extractor
// (the Vulkan one in gpu/), so that they choose exactly the same keypoints.

// Keypoints keep this distance from the level border; levels whose width or
// height is at most twice this hold no keypoints.
inline constexpr int kOrbBorder = 18;

struct OrbCorner {
    int x, y;     // level pixels
    float score;  // Harris response
};

// Detection grid of one pyramid level: roughly one cell per wanted feature.
struct OrbGrid {
    int x0, y0, x1, y1;  // detection area (inside the border), x1/y1 exclusive
    int cell, cols, rows;
    int cellOf(int x, int y) const { return ((y - y0) / cell) * cols + (x - x0) / cell; }
};
OrbGrid orbGrid(int width, int height, int budget);

// Picks up to `budget` of a level's corners, spread over the grid: weighted
// round-robin over cells, strongest first. `corners` are the level's 3x3
// local maxima in detection order (cell row, then cell, then y, then x).
void distributeOrbCorners(const OrbGrid& grid, const std::vector<OrbCorner>& corners, int budget,
                          float centerBoost, std::vector<OrbCorner>& out);

// Oriented FAST + rotated BRIEF (Rublee et al., ICCV 2011), with features
// spread over the image using a per-level grid.
class OrbExtractor {
public:
    // With a pool, every stage runs in parallel; output is identical either way.
    explicit OrbExtractor(const OrbParams& params = {}, ThreadPool* pool = nullptr);

    // Detects keypoints on `gray` and computes one descriptor per keypoint.
    void extract(const Image& gray, std::vector<Keypoint>& keypoints,
                 std::vector<Descriptor>& descriptors);

    const OrbParams& params() const { return params_; }
    // Per pyramid level: scale relative to level 0 and feature budget.
    const std::vector<float>& scales() const { return scales_; }
    const std::vector<int>& budgets() const { return budgets_; }
    // Size of a level of a width x height input; 0 x 0 if too small to hold
    // any keypoint (and then so are all coarser levels).
    static void levelSize(int width, int height, float scale, int& levelWidth, int& levelHeight);

    // Smoothed pyramid level from the last extract() call (used for training).
    // A keypoint's level coordinates are (x + 0.5) / scale - 0.5.
    const Image& blurredLevel(int level) const { return blurred_[level]; }

private:
    using Corner = OrbCorner;

    void buildPyramid(const Image& gray);
    void detectLevel(int level, int budget, std::vector<Corner>& out);

    OrbParams params_;
    ThreadPool* pool_ = nullptr;
    std::vector<float> scales_;
    std::vector<int> budgets_;
    std::vector<Image> pyramid_;
    std::vector<Image> blurred_;
    std::vector<float> scoreMap_;   // reused NMS buffer
    std::vector<Corner> candidates_;
    std::vector<Corner> maxima_;
    std::vector<std::vector<Corner>> rowCandidates_;
    std::vector<std::uint8_t> isLocalMax_;
};

}  // namespace vg
