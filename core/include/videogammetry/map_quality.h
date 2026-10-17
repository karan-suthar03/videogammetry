#pragma once

#include <vector>

#include "videogammetry/map.h"
#include "videogammetry/thread_pool.h"

namespace vg {

// Ground-truth-free measures of map noise, for comparing mapping changes.
struct MapQuality {
    int points = 0;  // good (non-rejected) points

    // Observations: keyframes that see each point.
    double meanObservations = 0;
    double twoViewShare = 0;       // seen by only 2 keyframes (never confirmed)

    // Largest triangulation angle among each point's keyframes. Depth
    // uncertainty grows as this shrinks.
    double medianParallaxDeg = 0;
    double lowParallaxShare = 0;   // below 2 degrees

    // Local shape of the cloud: for each point, the share of the smallest
    // eigenvalue of its k nearest neighbours' covariance ("surface
    // variation": 0 on a plane, 1/3 in an isotropic blob). Street scenes are
    // mostly walls and ground, so noise raises it.
    double medianSurfaceVariation = 0;
    double scatteredShare = 0;     // surface variation above 0.1

    // MapPoint::noise (depth uncertainty for 1 px keypoint noise, percent of
    // the distance) over the points where it is known.
    double medianNoisePct = 0;
    double p90NoisePct = 0;
};

// Points marked bad are ignored. kNeighbours >= 3.
MapQuality evaluateMap(const std::vector<MapPoint>& points, const std::vector<Keyframe>& keyframes,
                       ThreadPool* pool = nullptr, int kNeighbours = 12);

}  // namespace vg
