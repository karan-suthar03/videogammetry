#pragma once

#include <vector>

#include "videogammetry/orb.h"
#include "videogammetry/thread_pool.h"

namespace vg {

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
