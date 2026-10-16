#include "videogammetry/orb.h"

#include "videogammetry/orb_training.h"
#include "videogammetry/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace vg {

namespace {

// Keypoints keep this distance from the level border so that the orientation
// patch (radius 15), the rotated BRIEF pattern and the Harris window all fit.
constexpr int kBorder = 18;
constexpr int kPatchRadius = 15;
constexpr int kHarrisRadius = 3;

// Bresenham circle of radius 3 used by FAST, clockwise from 12 o'clock.
constexpr int kCircle[16][2] = {
    {0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
    {0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}};

// The 256 tests used by the descriptor, learned offline by tools/train_orb_pattern.
#include "orb_pattern.inc"
const std::array<BriefTest, 256>& briefPattern() { return kLearnedBriefPattern; }

// Half-width of the circular orientation patch for each row offset v.
const std::array<int, kPatchRadius + 1>& patchRowExtent() {
    static const std::array<int, kPatchRadius + 1> extent = [] {
        std::array<int, kPatchRadius + 1> e{};
        for (int v = 0; v <= kPatchRadius; ++v) {
            e[v] = static_cast<int>(std::floor(std::sqrt(double(kPatchRadius * kPatchRadius - v * v)) + 1e-9));
        }
        return e;
    }();
    return extent;
}

// FAST-9 segment test: 9 contiguous circle pixels all brighter than p + t or all darker than p - t.
bool isFastCorner(const Image& img, int x, int y, int t) {
    const std::uint8_t* c = img.row(y) + x;
    const int stride = img.width;
    const int p = *c;
    const int hi = p + t;
    const int lo = p - t;

    const auto px = [&](int i) { return c[kCircle[i][1] * stride + kCircle[i][0]]; };

    // Any arc of 9 covers at least two of the four compass points.
    const int n = px(0), e = px(4), s = px(8), w = px(12);
    const int brightCompass = (n > hi) + (e > hi) + (s > hi) + (w > hi);
    const int darkCompass = (n < lo) + (e < lo) + (s < lo) + (w < lo);
    if (brightCompass < 2 && darkCompass < 2) return false;

    int v[16];
    for (int i = 0; i < 16; ++i) v[i] = px(i);

    const auto hasArc = [&](auto pred) {
        int run = 0;
        for (int i = 0; i < 16 + 8; ++i) {
            if (pred(v[i & 15])) {
                if (++run >= 9) return true;
            } else {
                run = 0;
            }
        }
        return false;
    };
    return (brightCompass >= 2 && hasArc([hi](int q) { return q > hi; })) ||
           (darkCompass >= 2 && hasArc([lo](int q) { return q < lo; }));
}

float harrisResponse(const Image& img, int x, int y) {
    float a = 0, b = 0, c = 0;
    for (int dy = -kHarrisRadius; dy <= kHarrisRadius; ++dy) {
        const std::uint8_t* r = img.row(y + dy) + x;
        const std::uint8_t* up = img.row(y + dy - 1) + x;
        const std::uint8_t* dn = img.row(y + dy + 1) + x;
        for (int dx = -kHarrisRadius; dx <= kHarrisRadius; ++dx) {
            const float ix = float(r[dx + 1]) - float(r[dx - 1]);
            const float iy = float(dn[dx]) - float(up[dx]);
            a += ix * ix;
            b += iy * iy;
            c += ix * iy;
        }
    }
    constexpr float k = 0.04f;
    return a * b - c * c - k * (a + b) * (a + b);
}

// Orientation from the intensity centroid of the circular patch.
float intensityCentroidAngle(const Image& img, int x, int y) {
    const auto& extent = patchRowExtent();
    int m01 = 0, m10 = 0;

    const std::uint8_t* center = img.row(y) + x;
    for (int u = -kPatchRadius; u <= kPatchRadius; ++u) m10 += u * center[u];

    for (int v = 1; v <= kPatchRadius; ++v) {
        const std::uint8_t* below = img.row(y + v) + x;
        const std::uint8_t* above = img.row(y - v) + x;
        int vSum = 0;
        for (int u = -extent[v]; u <= extent[v]; ++u) {
            const int b = below[u];
            const int a = above[u];
            vSum += b - a;
            m10 += u * (b + a);
        }
        m01 += v * vSum;
    }
    return std::atan2(float(m01), float(m10));
}

Descriptor computeDescriptor(const Image& blurred, int x, int y, float angle) {
    const float ca = std::cos(angle);
    const float sa = std::sin(angle);
    const auto& pattern = briefPattern();

    Descriptor d{};
    for (int i = 0; i < 256; ++i) {
        if (evaluateBriefTest(blurred, x, y, ca, sa, pattern[i])) {
            d[i >> 6] |= std::uint64_t{1} << (i & 63);
        }
    }
    return d;
}

}  // namespace

std::vector<BriefTest> randomBriefTests(int count, std::uint32_t seed) {
    // Own Box-Muller on raw mt19937 output: std::normal_distribution differs
    // between standard libraries, this does not.
    std::mt19937 rng(seed);
    const auto uniform = [&rng] { return (rng() >> 8) * (1.0 / 16777216.0); };  // [0, 1)
    const auto gaussianPoint = [&](std::int8_t& x, std::int8_t& y) {
        const double sigma = 31.0 / 5.0;
        while (true) {
            const double u1 = 1.0 - uniform();  // (0, 1]
            const double u2 = uniform();
            const double r = std::sqrt(-2.0 * std::log(u1)) * sigma;
            const int px = static_cast<int>(std::lround(r * std::cos(6.283185307179586 * u2)));
            const int py = static_cast<int>(std::lround(r * std::sin(6.283185307179586 * u2)));
            if (px * px + py * py <= kBriefTestRadius * kBriefTestRadius) {
                x = static_cast<std::int8_t>(px);
                y = static_cast<std::int8_t>(py);
                return;
            }
        }
    };

    std::vector<BriefTest> tests(static_cast<std::size_t>(count));
    for (BriefTest& t : tests) {
        do {
            gaussianPoint(t.x1, t.y1);
            gaussianPoint(t.x2, t.y2);
        } while (t.x1 == t.x2 && t.y1 == t.y2);
    }
    return tests;
}

OrbExtractor::OrbExtractor(const OrbParams& params, ThreadPool* pool) : params_(params), pool_(pool) {
    // Levels past a 2^16 downscale are empty for any real image (see
    // buildPyramid), so a huge level count does not allocate them.
    int levels = std::max(params_.levels, 1);
    if (params_.scaleFactor > 1.0f) {
        const double useful = 1.0 + std::floor(std::log(65536.0) / std::log(double(params_.scaleFactor)));
        levels = static_cast<int>(std::min<double>(levels, useful));
    }
    scales_.resize(levels);
    budgets_.resize(levels);
    pyramid_.resize(levels);
    blurred_.resize(levels);

    scales_[0] = 1.0f;
    for (int i = 1; i < levels; ++i) scales_[i] = scales_[i - 1] * params_.scaleFactor;

    // Geometric split of the feature budget: coarser levels get fewer features.
    const float f = 1.0f / params_.scaleFactor;
    const float first = params_.maxFeatures * (1 - f) / (1 - std::pow(f, float(levels)));
    int assigned = 0;
    for (int i = 0; i < levels - 1; ++i) {
        budgets_[i] = static_cast<int>(std::round(first * std::pow(f, float(i))));
        assigned += budgets_[i];
    }
    budgets_[levels - 1] = std::max(params_.maxFeatures - assigned, 0);
}

void OrbExtractor::buildPyramid(const Image& gray) {
    pyramid_[0] = gray;
    for (std::size_t i = 1; i < pyramid_.size(); ++i) {
        const int w = static_cast<int>(std::round(gray.width / scales_[i]));
        const int h = static_cast<int>(std::round(gray.height / scales_[i]));
        if (w <= 2 * kBorder || h <= 2 * kBorder) {
            pyramid_[i].resize(0, 0);
            continue;
        }
        resizeBilinear(pyramid_[i - 1], w, h, pyramid_[i], pool_);
    }
}

void OrbExtractor::detectLevel(int level, int budget, std::vector<Corner>& out) {
    const Image& img = pyramid_[level];
    if (img.width <= 2 * kBorder || img.height <= 2 * kBorder || budget <= 0) return;  // too small for any corner

    const int x0 = kBorder, x1 = img.width - kBorder;
    const int y0 = kBorder, y1 = img.height - kBorder;

    // Grid with roughly one cell per wanted feature, so features spread evenly.
    const float area = float(x1 - x0) * float(y1 - y0);
    const int cell = std::clamp(static_cast<int>(std::sqrt(area / budget)), 16, 128);
    const int cols = (x1 - x0 + cell - 1) / cell;
    const int rows = (y1 - y0 + cell - 1) / cell;

    // 1. FAST per cell, retrying with a lower threshold where nothing was found.
    //    Each grid row is one parallel task with its own output list.
    rowCandidates_.resize(static_cast<std::size_t>(rows));
    parallelFor(pool_, 0, rows, 1, [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            std::vector<Corner>& found = rowCandidates_[r];
            found.clear();
            for (int c = 0; c < cols; ++c) {
                const int cx0 = x0 + c * cell, cx1 = std::min(cx0 + cell, x1);
                const int cy0 = y0 + r * cell, cy1 = std::min(cy0 + cell, y1);
                const std::size_t before = found.size();
                for (int threshold : {params_.fastThreshold, params_.minFastThreshold}) {
                    for (int y = cy0; y < cy1; ++y) {
                        for (int x = cx0; x < cx1; ++x) {
                            if (isFastCorner(img, x, y, threshold)) found.push_back({x, y, 0.0f});
                        }
                    }
                    if (found.size() > before) break;
                }
            }
        }
    });
    candidates_.clear();
    for (const auto& found : rowCandidates_) candidates_.insert(candidates_.end(), found.begin(), found.end());
    if (candidates_.empty()) return;

    // 2. Harris score + 3x3 non-maximum suppression (each corner owns its pixel).
    const int n = static_cast<int>(candidates_.size());
    constexpr int kCornerGrain = 256;
    scoreMap_.resize(static_cast<std::size_t>(img.width) * img.height, 0.0f);
    parallelFor(pool_, 0, n, kCornerGrain, [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            Corner& k = candidates_[i];
            k.score = harrisResponse(img, k.x, k.y);
            scoreMap_[static_cast<std::size_t>(k.y) * img.width + k.x] = k.score;
        }
    });
    isLocalMax_.assign(static_cast<std::size_t>(n), 0);
    parallelFor(pool_, 0, n, kCornerGrain, [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            const Corner& k = candidates_[i];
            bool isMax = true;
            for (int dy = -1; dy <= 1 && isMax; ++dy) {
                const float* s = scoreMap_.data() + static_cast<std::size_t>(k.y + dy) * img.width + k.x;
                for (int dx = -1; dx <= 1; ++dx) {
                    if ((dx || dy) && s[dx] != 0.0f &&
                        (s[dx] > k.score || (s[dx] == k.score && (dy < 0 || (dy == 0 && dx < 0))))) {
                        isMax = false;
                        break;
                    }
                }
            }
            isLocalMax_[i] = isMax ? 1 : 0;
        }
    });
    std::vector<std::vector<Corner>> buckets(static_cast<std::size_t>(cols) * rows);
    for (int i = 0; i < n; ++i) {
        const Corner& k = candidates_[i];
        scoreMap_[static_cast<std::size_t>(k.y) * img.width + k.x] = 0.0f;
        if (isLocalMax_[i]) {
            const int c = (k.x - x0) / cell, r = (k.y - y0) / cell;
            buckets[static_cast<std::size_t>(r) * cols + c].push_back(k);
        }
    }

    // 3. Weighted round-robin over cells, strongest first, until the level
    //    budget is filled: after round r a cell has taken up to r * weight
    //    corners (weight 1 everywhere unless the center is boosted).
    for (auto& b : buckets) {
        std::sort(b.begin(), b.end(), [](const Corner& a, const Corner& c) { return a.score > c.score; });
    }
    std::vector<float> weight(buckets.size(), 1.0f);
    if (params_.centerBoost > 0) {
        const float mx = 0.5f * (x0 + x1), my = 0.5f * (y0 + y1);
        const float halfDiagonal = 0.5f * std::hypot(float(x1 - x0), float(y1 - y0));
        constexpr float kSigma = 0.35f;  // of the half diagonal
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const float dx = (x0 + (c + 0.5f) * cell - mx) / halfDiagonal;
                const float dy = (y0 + (r + 0.5f) * cell - my) / halfDiagonal;
                weight[static_cast<std::size_t>(r) * cols + c] =
                    1.0f + params_.centerBoost * std::exp(-(dx * dx + dy * dy) / (2 * kSigma * kSigma));
            }
        }
    }
    std::vector<std::size_t> taken(buckets.size(), 0);
    for (int round = 1; static_cast<int>(out.size()) < budget; ++round) {
        bool any = false;
        for (std::size_t i = 0; i < buckets.size() && static_cast<int>(out.size()) < budget; ++i) {
            const std::size_t allowed = std::min(buckets[i].size(), static_cast<std::size_t>(round * weight[i]));
            while (taken[i] < allowed && static_cast<int>(out.size()) < budget) out.push_back(buckets[i][taken[i]++]);
            if (taken[i] < buckets[i].size()) any = true;
        }
        if (!any) break;
    }
}

void OrbExtractor::extract(const Image& gray, std::vector<Keypoint>& keypoints,
                           std::vector<Descriptor>& descriptors) {
    keypoints.clear();
    descriptors.clear();
    if (gray.empty()) return;

    buildPyramid(gray);

    // Detect on every level (each level is parallel internally), remembering
    // which level each corner came from.
    struct LevelCorner {
        Corner corner;
        int level;
    };
    std::vector<LevelCorner> all;
    std::vector<Corner> corners;
    for (int level = 0; level < static_cast<int>(pyramid_.size()); ++level) {
        if (pyramid_[level].empty()) continue;
        corners.clear();
        detectLevel(level, budgets_[level], corners);
        if (corners.empty()) continue;
        gaussianBlur7(pyramid_[level], blurred_[level], pool_);
        for (const Corner& c : corners) all.push_back({c, level});
    }

    // Orientation + descriptor for all keypoints in one parallel pass.
    const int n = static_cast<int>(all.size());
    keypoints.resize(static_cast<std::size_t>(n));
    descriptors.resize(static_cast<std::size_t>(n));
    parallelFor(pool_, 0, n, 64, [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            const Corner& c = all[i].corner;
            const int level = all[i].level;
            const float scale = scales_[level];
            Keypoint& kp = keypoints[i];
            kp.angle = intensityCentroidAngle(pyramid_[level], c.x, c.y);
            kp.x = (c.x + 0.5f) * scale - 0.5f;  // pixel centers map between levels
            kp.y = (c.y + 0.5f) * scale - 0.5f;
            kp.response = c.score;
            kp.level = level;
            kp.scale = scale;
            descriptors[i] = computeDescriptor(blurred_[level], c.x, c.y, kp.angle);
        }
    });
}

}  // namespace vg
