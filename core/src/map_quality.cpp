#include "videogammetry/map_quality.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <utility>

namespace vg {

namespace {

constexpr double kRadToDeg = 57.29577951308232;

// Static 3D k-d tree over point positions, for k-nearest-neighbour queries.
class KdTree {
public:
    explicit KdTree(const std::vector<Vec3>& points) : points_(points), order_(points.size()) {
        for (std::size_t i = 0; i < order_.size(); ++i) order_[i] = static_cast<int>(i);
        nodes_.reserve(points.size());
        if (!points.empty()) build(0, static_cast<int>(points.size()));
    }

    // Indices of the k nearest points to `q` (including a point at q itself).
    void nearest(const Vec3& q, int k, std::vector<int>& out) const {
        std::priority_queue<std::pair<double, int>> heap;  // (squared distance, index), farthest on top
        if (!nodes_.empty()) search(0, q, k, heap);
        out.clear();
        while (!heap.empty()) {
            out.push_back(heap.top().second);
            heap.pop();
        }
    }

private:
    struct Node {
        int point;        // index into points_ (the median)
        int axis;
        int left = -1, right = -1;
    };

    int build(int begin, int end) {
        if (begin >= end) return -1;
        Vec3 lo = points_[order_[begin]], hi = lo;
        for (int i = begin; i < end; ++i) {
            const Vec3& p = points_[order_[i]];
            lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
            hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
        }
        const Vec3 extent = hi - lo;
        const int axis = extent.x >= extent.y && extent.x >= extent.z ? 0 : (extent.y >= extent.z ? 1 : 2);
        const int mid = (begin + end) / 2;
        std::nth_element(order_.begin() + begin, order_.begin() + mid, order_.begin() + end,
                         [&](int a, int b) { return points_[a][axis] < points_[b][axis]; });
        const int node = static_cast<int>(nodes_.size());
        nodes_.push_back({order_[mid], axis});
        const int left = build(begin, mid);
        const int right = build(mid + 1, end);
        nodes_[node].left = left;
        nodes_[node].right = right;
        return node;
    }

    void search(int node, const Vec3& q, int k, std::priority_queue<std::pair<double, int>>& heap) const {
        const Node& n = nodes_[node];
        const Vec3& p = points_[n.point];
        const Vec3 d = p - q;
        const double d2 = dot(d, d);
        if (static_cast<int>(heap.size()) < k) {
            heap.push({d2, n.point});
        } else if (d2 < heap.top().first) {
            heap.pop();
            heap.push({d2, n.point});
        }
        const double diff = q[n.axis] - p[n.axis];
        const int nearSide = diff < 0 ? n.left : n.right;
        const int farSide = diff < 0 ? n.right : n.left;
        if (nearSide >= 0) search(nearSide, q, k, heap);
        if (farSide >= 0 && (static_cast<int>(heap.size()) < k || diff * diff < heap.top().first)) {
            search(farSide, q, k, heap);
        }
    }

    const std::vector<Vec3>& points_;
    std::vector<int> order_;
    std::vector<Node> nodes_;
};

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
    std::nth_element(v.begin(), mid, v.end());
    return *mid;
}

}  // namespace

MapQuality evaluateMap(const std::vector<MapPoint>& points, const std::vector<Keyframe>& keyframes, ThreadPool* pool,
                       int kNeighbours) {
    MapQuality q;

    // Observing keyframes of every point, from the keyframes' side.
    std::vector<std::vector<int>> seenBy(points.size());
    for (std::size_t k = 0; k < keyframes.size(); ++k) {
        for (int id : keyframes[k].pointIds) {
            if (id >= 0 && static_cast<std::size_t>(id) < points.size()) seenBy[id].push_back(static_cast<int>(k));
        }
    }

    std::vector<int> good;
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (!points[i].bad) good.push_back(static_cast<int>(i));
    }
    q.points = static_cast<int>(good.size());
    if (good.empty()) return q;

    std::vector<double> parallax(good.size(), 0.0);
    long long observations = 0;
    int twoView = 0, lowParallax = 0;
    for (std::size_t j = 0; j < good.size(); ++j) {
        const std::vector<int>& kfs = seenBy[good[j]];
        observations += static_cast<long long>(kfs.size());
        if (kfs.size() <= 2) ++twoView;
        const Vec3& X = points[good[j]].position;
        double best = 0.0;
        for (std::size_t a = 0; a < kfs.size(); ++a) {
            const Vec3 ra = normalized(X - keyframes[kfs[a]].pose.center());
            for (std::size_t b = a + 1; b < kfs.size(); ++b) {
                const Vec3 rb = normalized(X - keyframes[kfs[b]].pose.center());
                best = std::max(best, std::acos(std::clamp(dot(ra, rb), -1.0, 1.0)) * kRadToDeg);
            }
        }
        parallax[j] = best;
        if (best < 2.0) ++lowParallax;
    }
    q.meanObservations = double(observations) / good.size();
    q.twoViewShare = double(twoView) / good.size();
    q.medianParallaxDeg = median(parallax);
    std::vector<double> noise;
    noise.reserve(good.size());
    for (int id : good) {
        if (points[id].noise >= 0) noise.push_back(100.0 * points[id].noise);
    }
    if (!noise.empty()) {
        std::sort(noise.begin(), noise.end());
        q.medianNoisePct = noise[noise.size() / 2];
        q.p90NoisePct = noise[std::min(noise.size() - 1, noise.size() * 9 / 10)];
    }
    q.lowParallaxShare = double(lowParallax) / good.size();

    // Surface variation of each point's neighbourhood.
    std::vector<Vec3> positions(good.size());
    for (std::size_t j = 0; j < good.size(); ++j) positions[j] = points[good[j]].position;
    const KdTree tree(positions);
    const int k = std::max(3, std::min(kNeighbours, static_cast<int>(positions.size())));
    std::vector<double> variation(positions.size(), 0.0);
    parallelFor(pool, 0, static_cast<int>(positions.size()), 256, [&](int j0, int j1) {
        std::vector<int> nn;
        for (int j = j0; j < j1; ++j) {
            tree.nearest(positions[j], k, nn);
            Vec3 mean;
            for (int i : nn) mean = mean + positions[i];
            mean = mean * (1.0 / nn.size());
            double c[9] = {};
            for (int i : nn) {
                const Vec3 d = positions[i] - mean;
                const double v[3] = {d.x, d.y, d.z};
                for (int r = 0; r < 3; ++r)
                    for (int s = 0; s < 3; ++s) c[r * 3 + s] += v[r] * v[s];
            }
            double values[3], vectors[9];
            symmetricEigen(c, 3, values, vectors);
            const double sum = values[0] + values[1] + values[2];
            variation[j] = sum > 0 ? std::max(0.0, values[0]) / sum : 0.0;
        }
    });
    q.medianSurfaceVariation = median(variation);
    q.scatteredShare =
        double(std::count_if(variation.begin(), variation.end(), [](double v) { return v > 0.1; })) / variation.size();
    return q;
}

}  // namespace vg
