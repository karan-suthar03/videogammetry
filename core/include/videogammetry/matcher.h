#pragma once

#include <algorithm>
#include <vector>

#include "videogammetry/orb.h"
#include "videogammetry/thread_pool.h"

namespace vg {

// Uniform grid over keypoint positions for radius queries.
class KeypointGrid {
public:
    KeypointGrid(const std::vector<Keypoint>& keypoints, float cellSize) : cell_(cellSize) {
        if (keypoints.empty()) return;
        float maxX = 0, maxY = 0;
        for (const Keypoint& k : keypoints) {
            maxX = std::max(maxX, k.x);
            maxY = std::max(maxY, k.y);
        }
        cols_ = static_cast<int>(maxX / cell_) + 1;
        rows_ = static_cast<int>(maxY / cell_) + 1;
        cells_.resize(static_cast<std::size_t>(cols_) * rows_);
        for (int i = 0; i < static_cast<int>(keypoints.size()); ++i) {
            const int c = static_cast<int>(std::max(keypoints[i].x, 0.0f) / cell_);
            const int r = static_cast<int>(std::max(keypoints[i].y, 0.0f) / cell_);
            cells_[static_cast<std::size_t>(r) * cols_ + c].push_back(i);
        }
    }

    // Calls fn(index) for every keypoint whose cell overlaps the square around (x, y).
    template <typename Fn>
    void forEachNear(float x, float y, float radius, Fn&& fn) const {
        if (cells_.empty()) return;
        const int c0 = std::max(0, static_cast<int>((x - radius) / cell_));
        const int c1 = std::min(cols_ - 1, static_cast<int>((x + radius) / cell_));
        const int r0 = std::max(0, static_cast<int>((y - radius) / cell_));
        const int r1 = std::min(rows_ - 1, static_cast<int>((y + radius) / cell_));
        for (int r = r0; r <= r1; ++r) {
            for (int c = c0; c <= c1; ++c) {
                for (int i : cells_[static_cast<std::size_t>(r) * cols_ + c]) fn(i);
            }
        }
    }

private:
    float cell_;
    int cols_ = 0, rows_ = 0;
    std::vector<std::vector<int>> cells_;
};

struct Match {
    int query = -1;    // index into the query (current) features
    int train = -1;    // index into the train (earlier) features
    int distance = 0;  // Hamming distance in bits
};

struct MatchParams {
    float searchRadius = 100.0f;  // px (processed image); <= 0 compares every pair
    int maxLevelDiff = 2;         // max pyramid level difference between matched keypoints
    int maxDistance = 64;         // max Hamming distance (of 256 bits)
    float ratio = 0.8f;           // best must be < ratio * second best (Lowe's test); >= 1 disables
    bool crossCheck = true;       // keep only mutual best matches
    bool orientationCheck = true; // drop matches whose rotation disagrees with the majority
};

// Statistics of one matching call, for diagnostics.
struct MatchStats {
    int candidates = 0;        // queries with at least one candidate in range
    int afterDistance = 0;     // passed max distance + ratio test
    int afterCrossCheck = 0;
    int afterOrientation = 0;  // final
};

// Matches `query` features against `train` features. Output is sorted by
// query index and identical with or without a pool.
void matchFeatures(const std::vector<Keypoint>& queryKeypoints,
                   const std::vector<Descriptor>& queryDescriptors,
                   const std::vector<Keypoint>& trainKeypoints,
                   const std::vector<Descriptor>& trainDescriptors,
                   const MatchParams& params, std::vector<Match>& matches,
                   MatchStats* stats = nullptr, ThreadPool* pool = nullptr);

}  // namespace vg
