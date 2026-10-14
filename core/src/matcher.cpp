#include "videogammetry/matcher.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <memory>

namespace vg {

namespace {

struct Best {
    int index = -1;
    int distance = INT_MAX;
    int second = INT_MAX;
};

// For every feature in `from`, finds the best and second-best candidate in `to`.
void bestMatches(const std::vector<Keypoint>& fromK, const std::vector<Descriptor>& fromD,
                 const std::vector<Keypoint>& toK, const std::vector<Descriptor>& toD,
                 const KeypointGrid* toGrid, const MatchParams& p, std::vector<Best>& out,
                 ThreadPool* pool) {
    out.assign(fromD.size(), Best{});
    const float r2 = p.searchRadius * p.searchRadius;

    parallelFor(pool, 0, static_cast<int>(fromD.size()), 64, [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            const Keypoint& a = fromK[i];
            Best& b = out[i];
            const auto consider = [&](int j) {
                const Keypoint& c = toK[j];
                if (std::abs(c.level - a.level) > p.maxLevelDiff) return;
                if (toGrid) {
                    const float dx = c.x - a.x, dy = c.y - a.y;
                    if (dx * dx + dy * dy > r2) return;
                }
                const int d = hammingDistance(fromD[i], toD[j]);
                // Ties go to the lower index so results never depend on visit order.
                if (d < b.distance || (d == b.distance && j < b.index)) {
                    b.second = b.distance;
                    b.distance = d;
                    b.index = j;
                } else if (d < b.second) {
                    b.second = d;
                }
            };
            if (toGrid) {
                toGrid->forEachNear(a.x, a.y, p.searchRadius, consider);
            } else {
                for (int j = 0; j < static_cast<int>(toD.size()); ++j) consider(j);
            }
        }
    });
}

bool passesDistance(const Best& b, const MatchParams& p) {
    if (b.index < 0 || b.distance > p.maxDistance) return false;
    if (p.ratio < 1.0f && b.second != INT_MAX && b.distance >= p.ratio * b.second) return false;
    return true;
}

}  // namespace

void matchFeatures(const std::vector<Keypoint>& queryKeypoints,
                   const std::vector<Descriptor>& queryDescriptors,
                   const std::vector<Keypoint>& trainKeypoints,
                   const std::vector<Descriptor>& trainDescriptors,
                   const MatchParams& params, std::vector<Match>& matches,
                   MatchStats* stats, ThreadPool* pool) {
    matches.clear();
    MatchStats s;
    if (queryDescriptors.empty() || trainDescriptors.empty()) {
        if (stats) *stats = s;
        return;
    }

    const bool windowed = params.searchRadius > 0;
    const float cellSize = windowed ? std::max(params.searchRadius / 2, 8.0f) : 1.0f;
    std::unique_ptr<KeypointGrid> trainGrid, queryGrid;
    if (windowed) {
        trainGrid = std::make_unique<KeypointGrid>(trainKeypoints, cellSize);
        if (params.crossCheck) queryGrid = std::make_unique<KeypointGrid>(queryKeypoints, cellSize);
    }

    std::vector<Best> forward;
    bestMatches(queryKeypoints, queryDescriptors, trainKeypoints, trainDescriptors,
                trainGrid.get(), params, forward, pool);

    std::vector<Best> backward;
    if (params.crossCheck) {
        bestMatches(trainKeypoints, trainDescriptors, queryKeypoints, queryDescriptors,
                    queryGrid.get(), params, backward, pool);
    }

    for (int i = 0; i < static_cast<int>(forward.size()); ++i) {
        const Best& b = forward[i];
        if (b.index >= 0) ++s.candidates;
        if (!passesDistance(b, params)) continue;
        ++s.afterDistance;
        if (params.crossCheck && backward[b.index].index != i) continue;
        ++s.afterCrossCheck;
        matches.push_back({i, b.index, b.distance});
    }

    // Orientation consistency: between nearby frames all features rotate by
    // about the same angle. Histogram the rotation of every match and keep
    // only the dominant bins (as in ORB-SLAM).
    if (params.orientationCheck && !matches.empty()) {
        constexpr int kBins = 30;
        constexpr float kTwoPi = 6.2831853f;
        std::array<int, kBins> histogram{};
        std::vector<int> binOf(matches.size());
        for (std::size_t m = 0; m < matches.size(); ++m) {
            float d = queryKeypoints[matches[m].query].angle - trainKeypoints[matches[m].train].angle;
            d = std::fmod(d + 2 * kTwoPi, kTwoPi);  // [0, 2pi)
            const int bin = std::min(static_cast<int>(d / kTwoPi * kBins), kBins - 1);
            binOf[m] = bin;
            ++histogram[bin];
        }
        std::array<int, kBins> order{};
        for (int i = 0; i < kBins; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return histogram[a] > histogram[b]; });
        std::array<bool, kBins> keep{};
        const int top = histogram[order[0]];
        for (int k = 0; k < 3; ++k) {
            if (histogram[order[k]] > 0 && histogram[order[k]] >= top / 10) keep[order[k]] = true;
        }
        std::size_t w = 0;
        for (std::size_t m = 0; m < matches.size(); ++m) {
            if (keep[binOf[m]]) matches[w++] = matches[m];
        }
        matches.resize(w);
    }
    s.afterOrientation = static_cast<int>(matches.size());
    if (stats) *stats = s;
}

}  // namespace vg
