#include "videogammetry/mvs.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_map>

#include "videogammetry/reconstructor.h"
#include "videogammetry/thread_pool.h"

namespace vg {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr float kBadCost = 2.0f;

// Patch: 11x11 pixels, every second one (6 x 6 samples), weighted by color
// similarity to the center pixel and by distance from it.
constexpr int kPatchRadius = 5;
constexpr int kPatchSide = 6;
constexpr int kPatchSamples = kPatchSide * kPatchSide;
constexpr float kSigmaColor = 8.0f;   // gray levels
constexpr float kSigmaSpatial = 5.0f; // pixels
constexpr int kMaxSources = 16;

// Propagation: neighbours of the other checkerboard color (odd offsets).
constexpr int kNeighbours[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-5, 0}, {5, 0}, {0, -5}, {0, 5}};

struct Level {
    int width = 0, height = 0;
    std::vector<float> px;
};

Level toLevel(const Image& img) {
    Level l;
    l.width = img.width;
    l.height = img.height;
    l.px.assign(img.data.begin(), img.data.end());
    return l;
}

// Halves a level by averaging 2x2 blocks.
Level halve(const Level& src) {
    Level dst;
    dst.width = std::max(1, src.width / 2);
    dst.height = std::max(1, src.height / 2);
    dst.px.resize(static_cast<std::size_t>(dst.width) * dst.height);
    for (int y = 0; y < dst.height; ++y) {
        const int y0 = std::min(2 * y, src.height - 1), y1 = std::min(2 * y + 1, src.height - 1);
        for (int x = 0; x < dst.width; ++x) {
            const int x0 = std::min(2 * x, src.width - 1), x1 = std::min(2 * x + 1, src.width - 1);
            dst.px[static_cast<std::size_t>(y) * dst.width + x] =
                0.25f * (src.px[static_cast<std::size_t>(y0) * src.width + x0] + src.px[static_cast<std::size_t>(y0) * src.width + x1] +
                         src.px[static_cast<std::size_t>(y1) * src.width + x0] + src.px[static_cast<std::size_t>(y1) * src.width + x1]);
        }
    }
    return dst;
}

inline bool sample(const Level& l, float u, float v, float& out) {
    if (!(u >= 0 && v >= 0 && u < l.width - 1 && v < l.height - 1)) return false;
    const int x = static_cast<int>(u), y = static_cast<int>(v);
    const float ax = u - x, ay = v - y;
    const float* p = &l.px[static_cast<std::size_t>(y) * l.width + x];
    out = (p[0] * (1 - ax) + p[1] * ax) * (1 - ay) + (p[l.width] * (1 - ax) + p[l.width + 1] * ax) * ay;
    return true;
}

// Deterministic per-pixel random numbers in [0, 1).
inline double random01(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
}

Vec3 randomUnit(std::uint64_t& rng) {
    const double z = 2 * random01(rng) - 1, phi = 2 * kPi * random01(rng);
    const double r = std::sqrt(std::max(0.0, 1 - z * z));
    return {r * std::cos(phi), r * std::sin(phi), z};
}

// A normal facing the camera along `ray` (unit), at most ~80 degrees off it.
Vec3 facing(Vec3 n, const Vec3& ray) {
    if (dot(n, ray) > 0) n = -n;
    for (int i = 0; i < 4 && dot(n, ray) > -0.17; ++i) n = normalized(n - ray * 0.5);
    return n;
}

// Homography of one source for a plane: H = A + b m^T with A = Ks R Kr^-1,
// b = Ks t (relative pose reference -> source) and m = Kr^-T n / (n . X).
struct SourceGeometry {
    float A[9];
    float b[3];
};

SourceGeometry sourceGeometry(const Pose& ref, const Pose& src, const PinholeCamera& k) {
    const Mat3 R = src.R * ref.R.transposed();
    const Vec3 t = src.t - R * ref.t;
    Mat3 K, Kinv;
    K.m = {k.fx, 0, k.cx, 0, k.fy, k.cy, 0, 0, 1};
    Kinv.m = {1 / k.fx, 0, -k.cx / k.fx, 0, 1 / k.fy, -k.cy / k.fy, 0, 0, 1};
    const Mat3 A = K * R * Kinv;
    const Vec3 b = K * t;
    SourceGeometry g;
    for (int i = 0; i < 9; ++i) g.A[i] = static_cast<float>(A.m[i]);
    g.b[0] = static_cast<float>(b.x);
    g.b[1] = static_cast<float>(b.y);
    g.b[2] = static_cast<float>(b.z);
    return g;
}

struct RefPatch {
    float a[kPatchSamples];
    float w[kPatchSamples];
    float sumW = 0, mean = 0, var = 0;
    bool textured = false;
};

// PatchMatch on one image level.
class LevelSolver {
public:
    LevelSolver(const Level& ref, const std::vector<Level>& sources, const std::vector<SourceGeometry>& geometry,
                const PinholeCamera& camera, double minDepth, double maxDepth, const DenseParams& params)
        : ref_(ref), sources_(sources), geometry_(geometry), k_(camera), minDepth_(minDepth), maxDepth_(maxDepth),
          params_(params) {
        const std::size_t n = static_cast<std::size_t>(ref.width) * ref.height;
        depth.assign(n, 0.0f);
        normal.assign(n, Vec3{0, 0, -1});
        cost.assign(n, kBadCost);
    }

    std::vector<float> depth;
    std::vector<Vec3> normal;
    std::vector<float> cost;

    int width() const { return ref_.width; }
    int height() const { return ref_.height; }

    Vec3 ray(int x, int y) const { return {(x - k_.cx) / k_.fx, (y - k_.cy) / k_.fy, 1.0}; }

    void patch(int x, int y, RefPatch& p) const {
        const float center = ref_.px[static_cast<std::size_t>(y) * ref_.width + x];
        double sw = 0, sa = 0, saa = 0;
        int i = 0;
        for (int dy = -kPatchRadius; dy <= kPatchRadius; dy += 2) {
            const int yy = std::clamp(y + dy, 0, ref_.height - 1);
            for (int dx = -kPatchRadius; dx <= kPatchRadius; dx += 2, ++i) {
                const int xx = std::clamp(x + dx, 0, ref_.width - 1);
                const float a = ref_.px[static_cast<std::size_t>(yy) * ref_.width + xx];
                const float w = std::exp(-std::abs(a - center) / kSigmaColor -
                                         float(dx * dx + dy * dy) / (2 * kSigmaSpatial * kSigmaSpatial));
                p.a[i] = a;
                p.w[i] = w;
                sw += w;
                sa += w * a;
                saa += w * a * a;
            }
        }
        p.sumW = static_cast<float>(sw);
        p.mean = static_cast<float>(sa / sw);
        p.var = static_cast<float>(std::max(0.0, saa / sw - (sa / sw) * (sa / sw)));
        p.textured = std::sqrt(p.var) >= params_.minPatchStd;
    }

    // 1 - NCC of the plane (depth d, normal n) at pixel (x, y), averaged
    // over the best views.
    float evaluate(int x, int y, const RefPatch& p, double d, const Vec3& n) const {
        const Vec3 r = ray(x, y);
        const double denom = dot(n, r);
        if (denom >= -1e-6 || !(d > 0)) return kBadCost;
        const double q = d * denom;  // n . X
        const float m0 = static_cast<float>(n.x / k_.fx / q), m1 = static_cast<float>(n.y / k_.fy / q);
        const float m2 = static_cast<float>((n.z - k_.cx * n.x / k_.fx - k_.cy * n.y / k_.fy) / q);
        float costs[kMaxSources];
        int count = 0;
        for (std::size_t s = 0; s < sources_.size(); ++s) {
            const SourceGeometry& g = geometry_[s];
            const Level& img = sources_[s];
            double sb = 0, sbb = 0, sab = 0;
            bool inside = true;
            int i = 0;
            for (int dy = -kPatchRadius; dy <= kPatchRadius && inside; dy += 2) {
                const float v = static_cast<float>(y + dy);
                for (int dx = -kPatchRadius; dx <= kPatchRadius; dx += 2, ++i) {
                    const float u = static_cast<float>(x + dx);
                    const float mp = m0 * u + m1 * v + m2;
                    const float hz = g.A[6] * u + g.A[7] * v + g.A[8] + g.b[2] * mp;
                    if (hz <= 1e-6f) {
                        inside = false;
                        break;
                    }
                    const float hx = g.A[0] * u + g.A[1] * v + g.A[2] + g.b[0] * mp;
                    const float hy = g.A[3] * u + g.A[4] * v + g.A[5] + g.b[1] * mp;
                    float b;
                    if (!sample(img, hx / hz, hy / hz, b)) {
                        inside = false;
                        break;
                    }
                    const float w = p.w[i];
                    sb += w * b;
                    sbb += w * b * b;
                    sab += w * p.a[i] * b;
                }
            }
            if (!inside) continue;
            const double mb = sb / p.sumW;
            const double varB = sbb / p.sumW - mb * mb;
            const double cov = sab / p.sumW - p.mean * mb;
            if (varB < 1e-3 || p.var < 1e-3) {
                costs[count++] = 1.0f;
                continue;
            }
            const double ncc = cov / std::sqrt(p.var * varB);
            costs[count++] = static_cast<float>(1.0 - std::clamp(ncc, -1.0, 1.0));
        }
        if (count == 0) return kBadCost;
        const int k = std::min(count, std::max(1, params_.bestViews));
        std::partial_sort(costs, costs + k, costs + count);
        float sum = 0;
        for (int i = 0; i < k; ++i) sum += costs[i];
        return sum / k;
    }

    double randomDepth(std::uint64_t& rng) const {
        const double a = 1 / maxDepth_, b = 1 / minDepth_;
        return 1 / (a + (b - a) * random01(rng));
    }

    std::uint64_t seed(int x, int y, int round) const {
        return (static_cast<std::uint64_t>(y) * 73856093u) ^ (static_cast<std::uint64_t>(x) * 19349663u) ^
               (static_cast<std::uint64_t>(round + 1) * 83492791u) ^ (static_cast<std::uint64_t>(ref_.width) << 40);
    }

    void randomInit(ThreadPool* pool) {
        parallelFor(pool, 0, height(), 4, [&](int y0, int y1) {
            RefPatch p;
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < width(); ++x) {
                    std::uint64_t rng = seed(x, y, -1);
                    const std::size_t i = static_cast<std::size_t>(y) * width() + x;
                    depth[i] = static_cast<float>(randomDepth(rng));
                    normal[i] = facing(randomUnit(rng), normalized(ray(x, y)));
                    patch(x, y, p);
                    cost[i] = evaluate(x, y, p, depth[i], normal[i]);
                }
        });
    }

    // Starts from a coarser level's result (depth and normal of the parent pixel).
    void initFrom(const LevelSolver& coarse, ThreadPool* pool) {
        parallelFor(pool, 0, height(), 4, [&](int y0, int y1) {
            RefPatch p;
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < width(); ++x) {
                    const int px = std::min(x / 2, coarse.width() - 1), py = std::min(y / 2, coarse.height() - 1);
                    const std::size_t j = static_cast<std::size_t>(py) * coarse.width() + px;
                    const std::size_t i = static_cast<std::size_t>(y) * width() + x;
                    std::uint64_t rng = seed(x, y, -2);
                    depth[i] = coarse.depth[j] > 0 ? coarse.depth[j] : static_cast<float>(randomDepth(rng));
                    normal[i] = coarse.depth[j] > 0 ? coarse.normal[j] : facing(randomUnit(rng), normalized(ray(x, y)));
                    patch(x, y, p);
                    cost[i] = evaluate(x, y, p, depth[i], normal[i]);
                }
        });
    }

    // Seeds (camera-frame points) replace worse hypotheses at their pixels.
    void addSeeds(const std::vector<Vec3>& points) {
        RefPatch p;
        for (const Vec3& c : points) {
            if (c.z <= 0) continue;
            const int x = static_cast<int>(std::lround(k_.fx * c.x / c.z + k_.cx));
            const int y = static_cast<int>(std::lround(k_.fy * c.y / c.z + k_.cy));
            if (x < 0 || y < 0 || x >= width() || y >= height()) continue;
            const std::size_t i = static_cast<std::size_t>(y) * width() + x;
            const Vec3 n = -normalized(ray(x, y));
            patch(x, y, p);
            const float c2 = evaluate(x, y, p, c.z, n);
            if (c2 < cost[i]) {
                depth[i] = static_cast<float>(c.z);
                normal[i] = n;
                cost[i] = c2;
            }
        }
    }

    // One red-black round: every pixel tries its neighbours' planes and a
    // few random changes of its own.
    void iterate(int round, double scale, ThreadPool* pool) {
        for (int color = 0; color < 2; ++color) {
            parallelFor(pool, 0, height(), 2, [&](int y0, int y1) {
                RefPatch p;
                for (int y = y0; y < y1; ++y)
                    for (int x = (y + color) & 1; x < width(); x += 2) step(x, y, round, scale, p);
            });
        }
    }

    // Final test: enough NCC and texture, depth in range.
    void filter(ThreadPool* pool) {
        const float maxCost = static_cast<float>(1.0 - params_.minNcc);
        parallelFor(pool, 0, height(), 8, [&](int y0, int y1) {
            RefPatch p;
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < width(); ++x) {
                    const std::size_t i = static_cast<std::size_t>(y) * width() + x;
                    patch(x, y, p);
                    if (!p.textured || cost[i] > maxCost || depth[i] < minDepth_ || depth[i] > maxDepth_) depth[i] = 0;
                }
        });
    }

private:
    void step(int x, int y, int round, double scale, RefPatch& p) {
        const std::size_t i = static_cast<std::size_t>(y) * width() + x;
        patch(x, y, p);
        if (!p.textured) return;
        double bestD = depth[i];
        Vec3 bestN = normal[i];
        float best = cost[i];
        const Vec3 r = ray(x, y);
        const auto tryPlane = [&](double d, const Vec3& n) {
            if (!(d >= minDepth_ && d <= maxDepth_)) return;
            const float c = evaluate(x, y, p, d, n);
            if (c < best) {
                best = c;
                bestD = d;
                bestN = n;
            }
        };
        for (const auto& o : kNeighbours) {
            const int xn = x + o[0], yn = y + o[1];
            if (xn < 0 || yn < 0 || xn >= width() || yn >= height()) continue;
            const std::size_t j = static_cast<std::size_t>(yn) * width() + xn;
            if (depth[j] <= 0 || cost[j] >= kBadCost) continue;
            const Vec3& n = normal[j];
            const double denom = dot(n, r);
            if (denom >= -1e-6) continue;
            tryPlane(dot(n, ray(xn, yn) * depth[j]) / denom, n);  // the neighbour's plane at this pixel
        }
        std::uint64_t rng = seed(x, y, round);
        const Vec3 rn = normalized(r);
        tryPlane(randomDepth(rng), facing(randomUnit(rng), rn));
        tryPlane(bestD, facing(normalized(bestN + randomUnit(rng) * (0.6 * scale)), rn));
        tryPlane(bestD * (1 + 0.1 * scale * (2 * random01(rng) - 1)), facing(normalized(bestN + randomUnit(rng) * (0.2 * scale)), rn));
        depth[i] = static_cast<float>(bestD);
        normal[i] = bestN;
        cost[i] = best;
    }

    const Level& ref_;
    const std::vector<Level>& sources_;
    const std::vector<SourceGeometry>& geometry_;
    PinholeCamera k_;
    double minDepth_, maxDepth_;
    const DenseParams& params_;
};

// Octahedral normal encoding in 16 bits.
std::uint16_t encodeNormal(const Vec3& n) {
    const double l1 = std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
    double u = n.x / l1, v = n.y / l1;
    if (n.z < 0) {
        const double ou = u, ov = v;
        u = (1 - std::abs(ov)) * (ou >= 0 ? 1 : -1);
        v = (1 - std::abs(ou)) * (ov >= 0 ? 1 : -1);
    }
    const int a = static_cast<int>(std::lround((u * 0.5 + 0.5) * 255)), b = static_cast<int>(std::lround((v * 0.5 + 0.5) * 255));
    return static_cast<std::uint16_t>(std::clamp(a, 0, 255) | (std::clamp(b, 0, 255) << 8));
}

Vec3 decodeNormal(std::uint16_t e) {
    double u = (e & 255) / 255.0 * 2 - 1, v = (e >> 8) / 255.0 * 2 - 1;
    const double z = 1 - std::abs(u) - std::abs(v);
    if (z < 0) {
        const double ou = u, ov = v;
        u = (1 - std::abs(ov)) * (ou >= 0 ? 1 : -1);
        v = (1 - std::abs(ou)) * (ov >= 0 ? 1 : -1);
    }
    return normalized({u, v, z});
}

// Of keyframes seeing the scene from nearly the same place and direction,
// only the one seeing the most map points (spacing relative to the median
// depth of their map points).
std::vector<const Keyframe*> spreadKeyframes(const std::vector<const Keyframe*>& kfs, const std::vector<MapPoint>& points,
                                             double spacing, double angleDeg) {
    struct Candidate {
        const Keyframe* kf;
        int points;
        double depth;
    };
    std::vector<Candidate> candidates;
    for (const Keyframe* k : kfs) {
        std::vector<double> z;
        for (int id : k->pointIds) {
            if (id < 0 || static_cast<std::size_t>(id) >= points.size() || points[static_cast<std::size_t>(id)].bad) continue;
            const double d = k->pose.transform(points[static_cast<std::size_t>(id)].position).z;
            if (d > 0) z.push_back(d);
        }
        if (z.empty()) continue;
        std::nth_element(z.begin(), z.begin() + z.size() / 2, z.end());
        candidates.push_back({k, static_cast<int>(z.size()), z[z.size() / 2]});
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.points > b.points; });
    const double maxAngle = angleDeg * kPi / 180;
    std::vector<const Candidate*> kept;
    for (const Candidate& a : candidates) {
        const Vec3 c = a.kf->pose.center();
        const bool close = std::any_of(kept.begin(), kept.end(), [&](const Candidate* b) {
            return norm(b->kf->pose.center() - c) < spacing * std::min(a.depth, b->depth) &&
                   rotationAngle(a.kf->pose.R * b->kf->pose.R.transposed()) < maxAngle;
        });
        if (!close) kept.push_back(&a);
    }
    std::vector<const Keyframe*> out;
    for (const Candidate* k : kept) out.push_back(k->kf);
    return out;
}

// Bytes kept per pixel of every keyframe: gray, half-size RGB, depth, normal, fused flag.
constexpr double kBytesPerPixel = 1 + 0.75 + 2 + 2 + 1;

}  // namespace

DepthMap patchMatch(const MvsView& ref, const std::vector<const MvsView*>& sources, const PinholeCamera& camera,
                    double minDepth, double maxDepth, const std::vector<Vec3>& seeds, const DenseParams& params,
                    ThreadPool* pool, const std::atomic<bool>* cancel) {
    DepthMap out;
    out.width = ref.gray.width;
    out.height = ref.gray.height;
    const std::size_t n = static_cast<std::size_t>(out.width) * out.height;
    out.depth.assign(n, 0.0f);
    out.normal.assign(n, Vec3{0, 0, -1});
    out.cost.assign(n, kBadCost);
    if (sources.empty() || n == 0 || !(minDepth > 0) || !(maxDepth > minDepth)) return out;

    const int nsrc = std::min(static_cast<int>(sources.size()), kMaxSources);
    const int scales = std::max(1, params.scales);
    // Image pyramids, level 0 = full size.
    std::vector<std::vector<Level>> refLevels(1), srcLevels(static_cast<std::size_t>(nsrc));
    refLevels[0].push_back(toLevel(ref.gray));
    for (int s = 0; s < nsrc; ++s) srcLevels[static_cast<std::size_t>(s)].push_back(toLevel(sources[static_cast<std::size_t>(s)]->gray));
    for (int l = 1; l < scales; ++l) {
        refLevels[0].push_back(halve(refLevels[0].back()));
        for (auto& levels : srcLevels) levels.push_back(halve(levels.back()));
    }

    std::vector<Vec3> seedsCam;
    seedsCam.reserve(seeds.size());
    for (const Vec3& w : seeds) seedsCam.push_back(ref.pose.transform(w));

    std::unique_ptr<LevelSolver> previous;
    int round = 0;
    for (int l = scales - 1; l >= 0; --l) {
        const PinholeCamera k = camera.scaled(std::pow(2.0, l));
        std::vector<Level> levelSources;
        std::vector<SourceGeometry> geometry;
        for (int s = 0; s < nsrc; ++s) {
            levelSources.push_back(srcLevels[static_cast<std::size_t>(s)][static_cast<std::size_t>(l)]);
            geometry.push_back(sourceGeometry(ref.pose, sources[static_cast<std::size_t>(s)]->pose, k));
        }
        auto solver = std::make_unique<LevelSolver>(refLevels[0][static_cast<std::size_t>(l)], levelSources, geometry,
                                                    k, minDepth, maxDepth, params);
        if (previous) {
            solver->initFrom(*previous, pool);
        } else {
            solver->randomInit(pool);
            solver->addSeeds(seedsCam);
        }
        const int iterations = previous ? params.fineIterations : params.iterations;
        for (int it = 0; it < iterations; ++it, ++round) {
            if (cancel && *cancel) return out;
            solver->iterate(round, std::pow(0.5, it), pool);
        }
        if (l == 0) {
            solver->filter(pool);
            out.depth = std::move(solver->depth);
            out.normal = std::move(solver->normal);
            out.cost = std::move(solver->cost);
        }
        previous = std::move(solver);
    }
    return out;
}

struct DenseReconstructor::View {
    long frame = 0;
    MvsView mvs;
    std::vector<int> points;   // indices into mapPoints_
    std::vector<int> sources;  // indices into views_
    double minDepth = 0, maxDepth = 0;
    bool reference = false;  // gets a depth map
    std::vector<std::uint8_t> rgb;  // half size
    int rgbWidth = 0, rgbHeight = 0;
    std::vector<std::uint16_t> depth;  // quantized inverse depth, 0 = none
    std::vector<std::uint16_t> normal; // world frame, octahedral
    std::vector<std::uint8_t> used;    // pixel already part of a fused point
    bool hasImage = false, depthDone = false, hasDepth = false, fused = false;

    double invLo() const { return 1 / maxDepth; }
    double invHi() const { return 1 / minDepth; }
    std::uint16_t quantize(double d) const {
        const double t = (1 / d - invLo()) / (invHi() - invLo());
        return static_cast<std::uint16_t>(std::clamp(1 + std::lround(t * 65534), 1L, 65535L));
    }
    double dequantize(std::uint16_t q) const { return 1 / (invLo() + (q - 1) / 65534.0 * (invHi() - invLo())); }
};

DenseReconstructor::DenseReconstructor(const Reconstructor& r, const DenseParams& params) : params_(params) {
    const std::vector<MapPoint>& points = r.mapPoints();
    mapPoints_.reserve(points.size());
    for (const MapPoint& p : points) mapPoints_.push_back(p.position);
    const CameraIntrinsics& K = r.intrinsics();
    fxP_ = K.fx, fyP_ = K.fy, cxP_ = K.cx, cyP_ = K.cy, k1_ = K.k1;
    processedWidth_ = r.imageWidth();
    processedHeight_ = r.imageHeight();
    cloud_ = std::make_shared<DenseCloud>();

    std::vector<const Keyframe*> kfs;
    for (const Keyframe& k : r.keyframes()) kfs.push_back(&k);
    progress_.totalKeyframes = static_cast<int>(kfs.size());
    if (params_.selectKeyframes) kfs = spreadKeyframes(kfs, points, params_.keyframeSpacing, params_.keyframeAngleDeg);
    progress_.selected = static_cast<int>(kfs.size());
    std::sort(kfs.begin(), kfs.end(), [](auto* a, auto* b) { return a->frameIndex < b->frameIndex; });
    views_.resize(kfs.size());
    std::vector<std::vector<int>> observers(points.size());
    for (std::size_t v = 0; v < kfs.size(); ++v) {
        View& view = views_[v];
        view.frame = kfs[v]->frameIndex;
        view.mvs.pose = kfs[v]->pose;
        for (int id : kfs[v]->pointIds) {
            if (id < 0 || static_cast<std::size_t>(id) >= points.size() || points[static_cast<std::size_t>(id)].bad) continue;
            std::vector<int>& obs = observers[static_cast<std::size_t>(id)];
            if (!obs.empty() && obs.back() == static_cast<int>(v)) continue;
            obs.push_back(static_cast<int>(v));
            view.points.push_back(id);
        }
        // Depth range: the map points it sees, with a margin.
        std::vector<double> z;
        for (int id : view.points) {
            const double d = view.mvs.pose.transform(mapPoints_[static_cast<std::size_t>(id)]).z;
            if (d > 0) z.push_back(d);
        }
        if (z.size() >= 10) {
            std::sort(z.begin(), z.end());
            view.minDepth = 0.6 * z[z.size() / 100];
            view.maxDepth = 1.6 * z[z.size() - 1 - z.size() / 100];
        }
        frames_.push_back(view.frame);
    }

    // Source views: shared map points, each weighted by how well its
    // triangulation angle between the two cameras suits matching.
    const double minAngle = params_.minAngleDeg * kPi / 180, maxAngle = params_.maxAngleDeg * kPi / 180;
    for (std::size_t v = 0; v < views_.size(); ++v) {
        View& view = views_[v];
        const Vec3 c = view.mvs.pose.center();
        std::unordered_map<int, double> score;
        for (int id : view.points) {
            const Vec3& X = mapPoints_[static_cast<std::size_t>(id)];
            const Vec3 a = c - X;
            const double la = norm(a);
            for (int o : observers[static_cast<std::size_t>(id)]) {
                if (o == static_cast<int>(v)) continue;
                const Vec3 b = views_[static_cast<std::size_t>(o)].mvs.pose.center() - X;
                const double lb = norm(b);
                if (la <= 0 || lb <= 0) continue;
                const double angle = std::acos(std::clamp(dot(a, b) / (la * lb), -1.0, 1.0));
                double w = angle < minAngle ? (angle / minAngle) * (angle / minAngle)
                         : angle > maxAngle ? (maxAngle / angle) * (maxAngle / angle) : 1.0;
                const double ratio = la / lb;
                if (ratio > 2 || ratio < 0.5) w *= 0.25;  // very different scale
                score[o] += w;
            }
        }
        std::vector<std::pair<double, int>> ranked;
        for (const auto& [o, s] : score)
            if (s >= 5) ranked.push_back({s, o});
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
        const std::size_t keep = std::min<std::size_t>(ranked.size(), static_cast<std::size_t>(std::clamp(params_.sourceViews, 1, kMaxSources)));
        for (std::size_t i = 0; i < keep; ++i) view.sources.push_back(ranked[i].second);
    }
}

DenseReconstructor::~DenseReconstructor() = default;

void DenseReconstructor::setDepthSolver(DepthMapSolver solver, std::string name) {
    solver_ = std::move(solver);
    std::lock_guard<std::mutex> lock(mutex_);
    progress_.solver = name.empty() ? "CPU" : std::move(name);
}

int DenseReconstructor::imageLongSide(int videoWidth, int videoHeight) const {
    const int native = std::max(videoWidth, videoHeight);
    double side = params_.imageSize > 0 ? std::min(params_.imageSize, native) : native;
    const double aspect = double(std::min(videoWidth, videoHeight)) / std::max(1, native);
    const double bytes = side * side * aspect * kBytesPerPixel * std::max<std::size_t>(1, views_.size());
    const double limit = std::max(64, params_.maxMemoryMB) * 1048576.0;
    if (bytes > limit) side *= std::sqrt(limit / bytes);
    return std::max(64, static_cast<int>(side));
}

void DenseReconstructor::addImage(long frameIndex, const FrameView& frame) {
    const auto it = std::lower_bound(frames_.begin(), frames_.end(), frameIndex);
    if (it == frames_.end() || *it != frameIndex || frame.format != FrameView::Format::RGB8 || frame.width < 2) return;
    View& view = views_[static_cast<std::size_t>(it - frames_.begin())];
    if (view.hasImage) return;
    if (mapX_.empty()) {
        // Processed pixel x (integer centers) is (x + 0.5) * s - 0.5 here.
        width_ = frame.width;
        height_ = frame.height;
        const double s = width_ / std::max(1.0, processedWidth_);
        camera_ = {fxP_ * s, fyP_ * s, (cxP_ + 0.5) * s - 0.5, (cyP_ + 0.5) * s - 0.5};
        mapX_.resize(static_cast<std::size_t>(width_) * height_);
        mapY_.resize(mapX_.size());
        for (int y = 0; y < height_; ++y)
            for (int x = 0; x < width_; ++x) {
                const double nx = (x - camera_.cx) / camera_.fx, ny = (y - camera_.cy) / camera_.fy;
                const double d = 1 + k1_ * (nx * nx + ny * ny);
                mapX_[static_cast<std::size_t>(y) * width_ + x] = static_cast<float>(camera_.fx * nx * d + camera_.cx);
                mapY_[static_cast<std::size_t>(y) * width_ + x] = static_cast<float>(camera_.fy * ny * d + camera_.cy);
            }
        std::lock_guard<std::mutex> lock(mutex_);
        progress_.width = width_;
        progress_.height = height_;
    }
    if (frame.width != width_ || frame.height != height_) return;

    // Undistorted RGB (bilinear), then gray and half-size color.
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width_) * height_ * 3);
    view.mvs.gray.resize(width_, height_);
    for (int y = 0; y < height_; ++y)
        for (int x = 0; x < width_; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * width_ + x;
            const float sx = std::clamp(mapX_[i], 0.0f, float(width_ - 1)), sy = std::clamp(mapY_[i], 0.0f, float(height_ - 1));
            const int x0 = std::min(static_cast<int>(sx), width_ - 2), y0 = std::min(static_cast<int>(sy), height_ - 2);
            const float ax = sx - x0, ay = sy - y0;
            const std::uint8_t* p = frame.data + static_cast<std::size_t>(y0) * frame.stride + x0 * 3;
            std::uint8_t* o = &rgb[i * 3];
            for (int c = 0; c < 3; ++c) {
                const float top = p[c] * (1 - ax) + p[3 + c] * ax;
                const float bottom = p[frame.stride + c] * (1 - ax) + p[frame.stride + 3 + c] * ax;
                o[c] = static_cast<std::uint8_t>(top * (1 - ay) + bottom * ay + 0.5f);
            }
            view.mvs.gray.data[i] = static_cast<std::uint8_t>((77 * o[0] + 150 * o[1] + 29 * o[2] + 128) >> 8);
        }
    view.rgbWidth = std::max(1, width_ / 2);
    view.rgbHeight = std::max(1, height_ / 2);
    view.rgb.resize(static_cast<std::size_t>(view.rgbWidth) * view.rgbHeight * 3);
    for (int y = 0; y < view.rgbHeight; ++y)
        for (int x = 0; x < view.rgbWidth; ++x)
            for (int c = 0; c < 3; ++c) {
                const int x1 = std::min(2 * x + 1, width_ - 1), y1 = std::min(2 * y + 1, height_ - 1);
                const auto at = [&](int xx, int yy) { return rgb[(static_cast<std::size_t>(yy) * width_ + xx) * 3 + c]; };
                view.rgb[(static_cast<std::size_t>(y) * view.rgbWidth + x) * 3 + c] =
                    static_cast<std::uint8_t>((at(2 * x, 2 * y) + at(x1, 2 * y) + at(2 * x, y1) + at(x1, y1) + 2) / 4);
            }
    view.hasImage = true;
    std::lock_guard<std::mutex> lock(mutex_);
    ++progress_.keyframes;
}

// Views a depth map can be computed for: an image, a depth range and at
// least two source views with images.
void DenseReconstructor::checkViews() {
    int references = 0;
    for (View& view : views_) {
        std::vector<int> kept;
        for (int s : view.sources)
            if (views_[static_cast<std::size_t>(s)].hasImage) kept.push_back(s);
        view.sources = std::move(kept);
        view.reference = view.hasImage && view.minDepth > 0 && view.sources.size() >= 2;
        references += view.reference;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    progress_.references = references;
}

void DenseReconstructor::computeDepth(int index, ThreadPool& pool, const std::atomic<bool>& cancel) {
    View& view = views_[static_cast<std::size_t>(index)];
    view.depthDone = true;
    if (!view.reference) return;
    std::vector<const MvsView*> sources;
    for (int s : view.sources) sources.push_back(&views_[static_cast<std::size_t>(s)].mvs);
    if (sources.size() < 2) return;
    std::vector<Vec3> seeds;
    for (int id : view.points) seeds.push_back(mapPoints_[static_cast<std::size_t>(id)]);
    DepthMap dm;
    if (solver_ && !solver_(view.mvs, sources, camera_, view.minDepth, view.maxDepth, seeds, params_, &cancel, dm)) {
        if (cancel) return;
        solver_ = {};
        std::lock_guard<std::mutex> lock(mutex_);
        progress_.solver += " failed, now CPU";
    }
    if (!solver_) dm = patchMatch(view.mvs, sources, camera_, view.minDepth, view.maxDepth, seeds, params_, &pool, &cancel);
    if (cancel) return;
    const std::size_t n = dm.depth.size();
    view.depth.assign(n, 0);
    view.normal.assign(n, 0);
    view.used.assign(n, 0);
    const Mat3 Rt = view.mvs.pose.R.transposed();
    for (std::size_t i = 0; i < n; ++i) {
        if (dm.depth[i] <= 0) continue;
        view.depth[i] = view.quantize(dm.depth[i]);
        view.normal[i] = encodeNormal(Rt * dm.normal[i]);
    }
    view.hasDepth = true;
    std::lock_guard<std::mutex> lock(mutex_);
    lastDepth_ = std::make_shared<const DepthMap>(std::move(dm));
}

void DenseReconstructor::fuse(int index) {
    View& view = views_[static_cast<std::size_t>(index)];
    view.fused = true;
    auto points = std::make_shared<std::vector<Point>>();
    auto normals = std::make_shared<std::vector<Vec3>>();
    if (view.hasDepth) {
        const PinholeCamera& k = camera_;
        const double cosNormal = std::cos(params_.maxNormalDeg * kPi / 180);
        const Pose& pose = view.mvs.pose;
        std::vector<int> sources;
        for (int s : view.sources)
            if (views_[static_cast<std::size_t>(s)].hasDepth) sources.push_back(s);
        const auto unproject = [&](const View& v, int x, int y, double d) {
            const Vec3 c{(x - k.cx) / k.fx * d, (y - k.cy) / k.fy * d, d};
            return v.mvs.pose.R.transposed() * (c - v.mvs.pose.t);
        };
        const auto color = [](const View& v, int x, int y, double* sum) {
            const int hx = std::clamp(x / 2, 0, v.rgbWidth - 1), hy = std::clamp(y / 2, 0, v.rgbHeight - 1);
            const std::uint8_t* c = &v.rgb[(static_cast<std::size_t>(hy) * v.rgbWidth + hx) * 3];
            sum[0] += c[0], sum[1] += c[1], sum[2] += c[2];
        };
        struct Match {
            int view;
            std::size_t pixel;
        };
        std::vector<Match> matches;
        for (int y = 0; y < height_; ++y)
            for (int x = 0; x < width_; ++x) {
                const std::size_t i = static_cast<std::size_t>(y) * width_ + x;
                if (!view.depth[i] || view.used[i]) continue;
                const Vec3 X = unproject(view, x, y, view.dequantize(view.depth[i]));
                const Vec3 nX = decodeNormal(view.normal[i]);
                Vec3 sumX = X, sumN = nX;
                double rgb[3] = {0, 0, 0};
                color(view, x, y, rgb);
                matches.clear();
                bool duplicate = false;
                for (int s : sources) {
                    const View& src = views_[static_cast<std::size_t>(s)];
                    const Vec3 c = src.mvs.pose.transform(X);
                    if (c.z <= 0) continue;
                    const int u = static_cast<int>(std::lround(k.fx * c.x / c.z + k.cx));
                    const int v = static_cast<int>(std::lround(k.fy * c.y / c.z + k.cy));
                    if (u < 0 || v < 0 || u >= width_ || v >= height_) continue;
                    const std::size_t j = static_cast<std::size_t>(v) * width_ + u;
                    if (!src.depth[j]) continue;
                    const double dj = src.dequantize(src.depth[j]);
                    if (std::abs(c.z - dj) > params_.maxDepthDiff * dj) continue;
                    const Vec3 Y = unproject(src, u, v, dj);
                    const Vec3 back = pose.transform(Y);
                    if (back.z <= 0) continue;
                    const double ex = k.fx * back.x / back.z + k.cx - x, ey = k.fy * back.y / back.z + k.cy - y;
                    if (ex * ex + ey * ey > params_.maxReprojPx * params_.maxReprojPx) continue;
                    const Vec3 nY = decodeNormal(src.normal[j]);
                    if (dot(nX, nY) < cosNormal) continue;
                    if (src.used[j]) {  // this surface spot is already a point
                        duplicate = true;
                        break;
                    }
                    matches.push_back({s, j});
                    sumX = sumX + Y;
                    sumN = sumN + nY;
                    color(src, u, v, rgb);
                }
                if (duplicate) {
                    view.used[i] = 1;
                    continue;
                }
                if (static_cast<int>(matches.size()) < params_.minConsistentViews) continue;
                view.used[i] = 1;
                for (const Match& m : matches) views_[static_cast<std::size_t>(m.view)].used[m.pixel] = 1;
                const double count = 1.0 + matches.size();
                const Vec3 P = sumX * (1 / count);
                Point q;
                q.x = static_cast<float>(P.x);
                q.y = static_cast<float>(P.y);
                q.z = static_cast<float>(P.z);
                q.r = static_cast<std::uint8_t>(rgb[0] / count + 0.5);
                q.g = static_cast<std::uint8_t>(rgb[1] / count + 0.5);
                q.b = static_cast<std::uint8_t>(rgb[2] / count + 0.5);
                points->push_back(q);
                normals->push_back(normalized(sumN));
            }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    ++progress_.fused;
    if (points->empty()) return;
    progress_.points += static_cast<long>(points->size());
    auto next = std::make_shared<DenseCloud>(*cloud_);
    next->push_back(std::move(points));
    cloud_ = std::move(next);
    normals_.push_back(std::move(normals));
}

bool DenseReconstructor::run(const std::atomic<bool>& cancel, const std::function<void()>& onUpdate) {
    ThreadPool pool(params_.threads);
    checkViews();
    const int n = static_cast<int>(views_.size());
    const auto ready = [&](int v) {
        if (views_[static_cast<std::size_t>(v)].fused || !views_[static_cast<std::size_t>(v)].depthDone) return false;
        for (int s : views_[static_cast<std::size_t>(v)].sources)
            if (!views_[static_cast<std::size_t>(s)].depthDone) return false;
        return true;
    };
    for (int v = 0; v < n; ++v) {
        if (cancel) return false;
        if (!views_[static_cast<std::size_t>(v)].reference) {
            views_[static_cast<std::size_t>(v)].depthDone = true;
            continue;
        }
        const auto start = std::chrono::steady_clock::now();
        computeDepth(v, pool, cancel);
        if (cancel) return false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++progress_.depthMaps;
            progress_.lastDepthMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        if (onUpdate) onUpdate();
        for (int f = 0; f < n; ++f) {
            if (!ready(f)) continue;
            fuse(f);
            if (onUpdate) onUpdate();
        }
    }
    for (int f = 0; f < n; ++f)
        if (!views_[static_cast<std::size_t>(f)].fused) fuse(f);
    for (View& v : views_) {  // only the cloud is needed now
        v.mvs.gray = Image{};
        v.rgb = {};
        v.depth = {};
        v.normal = {};
        v.used = {};
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        progress_.finished = true;
    }
    if (onUpdate) onUpdate();
    return true;
}

DenseReconstructor::Progress DenseReconstructor::progress() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return progress_;
}

std::shared_ptr<const DenseCloud> DenseReconstructor::cloud() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cloud_;
}

bool DenseReconstructor::depthInputs(long frameIndex, DepthInputs& out) const {
    const auto it = std::lower_bound(frames_.begin(), frames_.end(), frameIndex);
    if (it == frames_.end() || *it != frameIndex) return false;
    const View& view = views_[static_cast<std::size_t>(it - frames_.begin())];
    out = {};
    for (int s : view.sources)
        if (views_[static_cast<std::size_t>(s)].hasImage) out.sources.push_back(&views_[static_cast<std::size_t>(s)].mvs);
    if (!view.hasImage || !(view.minDepth > 0) || out.sources.size() < 2) return false;
    out.reference = &view.mvs;
    out.camera = camera_;
    out.minDepth = view.minDepth;
    out.maxDepth = view.maxDepth;
    for (int id : view.points) out.seeds.push_back(mapPoints_[static_cast<std::size_t>(id)]);
    out.rgb = view.rgb.data();
    out.rgbWidth = view.rgbWidth;
    out.rgbHeight = view.rgbHeight;
    return true;
}

std::shared_ptr<const DepthMap> DenseReconstructor::lastDepthMap() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lastDepth_;
}

PointCloud DenseReconstructor::pointCloud() const {
    std::lock_guard<std::mutex> lock(mutex_);
    PointCloud out;
    for (std::size_t c = 0; c < cloud_->size(); ++c) {
        const std::vector<Point>& pts = *(*cloud_)[c];
        const std::vector<Vec3>& nrm = *normals_[c];
        out.points.insert(out.points.end(), pts.begin(), pts.end());
        for (const Vec3& n : nrm)
            out.normals.push_back({static_cast<float>(n.x), static_cast<float>(n.y), static_cast<float>(n.z)});
    }
    return out;
}

}  // namespace vg
