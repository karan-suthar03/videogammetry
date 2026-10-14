#include "videogammetry/two_view.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace vg {

CameraIntrinsics CameraIntrinsics::fromFov(int width, int height, double horizontalFovDeg) {
    CameraIntrinsics k;
    const double halfFov = horizontalFovDeg * 3.14159265358979323846 / 360.0;
    k.fx = k.fy = 0.5 * width / std::tan(halfFov);
    k.cx = 0.5 * width - 0.5;  // pixel centers are at integer coordinates
    k.cy = 0.5 * height - 0.5;
    return k;
}

double sampsonError(const Mat3& E, const Vec2& x1, const Vec2& x2) {
    const Vec3 p1{x1.x, x1.y, 1}, p2{x2.x, x2.y, 1};
    const Vec3 Ex1 = E * p1;
    const Vec3 Etx2 = E.transposed() * p2;
    const double num = dot(p2, Ex1);
    const double den = Ex1.x * Ex1.x + Ex1.y * Ex1.y + Etx2.x * Etx2.x + Etx2.y * Etx2.y;
    return den > 0 ? num * num / den : std::numeric_limits<double>::infinity();
}

namespace {

// Projects a 3x3 matrix to the nearest essential matrix (singular values 1, 1, 0).
Mat3 enforceEssential(const Mat3& F) {
    Mat3 u, v;
    Vec3 s;
    svd3(F, u, s, v);
    Mat3 d;
    d.m = {1, 0, 0, 0, 1, 0, 0, 0, 0};
    return u * d * v.transposed();
}

// Projects a 3x3 matrix to the nearest rank-2 matrix.
Mat3 enforceRank2(const Mat3& F) {
    Mat3 u, v;
    Vec3 s;
    svd3(F, u, s, v);
    Mat3 d;
    d.m = {s.x, 0, 0, 0, s.y, 0, 0, 0, 0};
    return u * d * v.transposed();
}

// Which constraint the 8-point solution is projected onto.
//  - Fundamental (rank 2): used for RANSAC hypotheses. Projecting a minimal
//    8-point solution straight onto the essential manifold distorts it badly
//    (on sideways motion < 1% of clean samples stay usable vs ~15% as rank 2).
//  - Essential (1, 1, 0): used once many inliers are known; accurate then.
enum class Model { Fundamental, Essential };

// Normalized 8-point algorithm on the given correspondence indices.
// Points are re-centered and scaled (Hartley) before solving.
bool eightPoint(const std::vector<Vec2>& p1, const std::vector<Vec2>& p2, const int* idx, int n, Mat3& E,
                Model model = Model::Essential) {
    const auto conditioning = [&](const std::vector<Vec2>& p, Mat3& T) {
        double mx = 0, my = 0;
        for (int i = 0; i < n; ++i) {
            mx += p[idx[i]].x;
            my += p[idx[i]].y;
        }
        mx /= n;
        my /= n;
        double d = 0;
        for (int i = 0; i < n; ++i) d += std::hypot(p[idx[i]].x - mx, p[idx[i]].y - my);
        d /= n;
        if (d <= 0) return false;
        const double s = std::sqrt(2.0) / d;
        T.m = {s, 0, -s * mx, 0, s, -s * my, 0, 0, 1};
        return true;
    };
    Mat3 T1, T2;
    if (!conditioning(p1, T1) || !conditioning(p2, T2)) return false;

    // Accumulate A^T A (9x9) directly; each row is x2 (x) x1 in conditioned coordinates.
    double ata[81] = {};
    for (int i = 0; i < n; ++i) {
        const Vec3 a = T1 * Vec3{p1[idx[i]].x, p1[idx[i]].y, 1};
        const Vec3 b = T2 * Vec3{p2[idx[i]].x, p2[idx[i]].y, 1};
        const double row[9] = {b.x * a.x, b.x * a.y, b.x, b.y * a.x, b.y * a.y, b.y, a.x, a.y, 1};
        for (int r = 0; r < 9; ++r)
            for (int c = r; c < 9; ++c) ata[r * 9 + c] += row[r] * row[c];
    }
    for (int r = 0; r < 9; ++r)
        for (int c = 0; c < r; ++c) ata[r * 9 + c] = ata[c * 9 + r];

    double values[9], vectors[81];
    symmetricEigen(ata, 9, values, vectors);
    Mat3 Fc;
    for (int k = 0; k < 9; ++k) Fc.m[k] = vectors[k * 9 + 0];  // smallest eigenvalue

    // Rank 2 in conditioned coordinates, undo conditioning, then (optionally)
    // project to an essential matrix.
    const Mat3 F = T2.transposed() * enforceRank2(Fc) * T1;
    E = model == Model::Essential ? enforceEssential(F) : F;
    return true;
}

struct Score {
    double cost = std::numeric_limits<double>::infinity();
    int inliers = 0;
};

Score scoreModel(const Mat3& E, const std::vector<Vec2>& p1, const std::vector<Vec2>& p2, double th2) {
    Score s{0, 0};
    for (std::size_t i = 0; i < p1.size(); ++i) {
        const double e = sampsonError(E, p1[i], p2[i]);
        if (e < th2) {
            s.cost += e;
            ++s.inliers;
        } else {
            s.cost += th2;
        }
    }
    return s;
}

// LO-RANSAC local optimization: refit E on inliers selected with a shrinking
// threshold (3x .. 1x). A minimal 8-point model is noisy; refitting only on
// its own 1x inliers would just reproduce its bias, the wider first passes
// let it move toward the true solution. Returns the refined model and score.
Mat3 localOptimize(const Mat3& E0, const std::vector<Vec2>& p1, const std::vector<Vec2>& p2,
                   double th2, Model model, Score& score) {
    Mat3 best = E0;
    score = scoreModel(E0, p1, p2, th2);
    std::vector<int> idx;
    for (int pass = 0; pass < 2; ++pass) {
        for (double factor : {3.0, 2.0, 1.5, 1.0}) {
            const double t2 = th2 * factor * factor;
            idx.clear();
            for (int i = 0; i < static_cast<int>(p1.size()); ++i) {
                if (sampsonError(best, p1[i], p2[i]) < t2) idx.push_back(i);
            }
            if (idx.size() < 8) continue;
            Mat3 refined;
            if (!eightPoint(p1, p2, idx.data(), static_cast<int>(idx.size()), refined, model)) continue;
            const Score s = scoreModel(refined, p1, p2, th2);
            if (s.cost < score.cost) {
                score = s;
                best = refined;
            }
        }
    }
    return best;
}

// Picks the (R, t) among the four decompositions of E with most points in front.
void recoverPose(const Mat3& E, const std::vector<Vec2>& p1, const std::vector<Vec2>& p2,
                 TwoViewResult& result) {
    Mat3 u, v;
    Vec3 s;
    svd3(E, u, s, v);
    if (u.determinant() < 0) u = u * -1.0;
    if (v.determinant() < 0) v = v * -1.0;
    Mat3 w;
    w.m = {0, -1, 0, 1, 0, 0, 0, 0, 1};
    const Mat3 Rs[2] = {u * w * v.transposed(), u * w.transposed() * v.transposed()};
    const Vec3 tu = u.column(2);
    const Vec3 ts[2] = {tu, -tu};

    int bestFront = -1;
    std::vector<double> bestParallax;
    for (const Mat3& R : Rs) {
        for (const Vec3& t : ts) {
            int front = 0;
            std::vector<double> parallax;
            const Vec3 c2 = -(R.transposed() * t);  // center of camera 2 in camera-1 frame
            for (std::size_t i = 0; i < p1.size(); ++i) {
                if (!result.inlier[i]) continue;
                Vec3 X;
                if (!triangulate(p1[i], p2[i], R, t, X)) continue;
                const Vec3 X2 = R * X + t;
                if (X.z <= 0 || X2.z <= 0) continue;
                ++front;
                const double c = dot(normalized(X), normalized(X - c2));
                parallax.push_back(std::acos(std::clamp(c, -1.0, 1.0)) * 180.0 / 3.14159265358979323846);
            }
            if (front > bestFront) {
                bestFront = front;
                result.R = R;
                result.t = t;
                bestParallax = std::move(parallax);
            }
        }
    }
    result.inFront = std::max(bestFront, 0);
    if (!bestParallax.empty()) {
        auto mid = bestParallax.begin() + static_cast<std::ptrdiff_t>(bestParallax.size() / 2);
        std::nth_element(bestParallax.begin(), mid, bestParallax.end());
        result.medianParallaxDeg = *mid;
    }
}

}  // namespace

bool essentialLinear(const std::vector<Vec2>& points1, const std::vector<Vec2>& points2,
                     const int* indices, int n, Mat3& E) {
    return n >= 8 && eightPoint(points1, points2, indices, n, E);
}

bool triangulate(const Vec2& x1, const Vec2& x2, const Mat3& R, const Vec3& t, Vec3& X) {
    // Rows of A (4x4): x * P3 - P1, y * P3 - P2 for both cameras; solve A X = 0.
    const double P1[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
    const double P2[3][4] = {{R(0, 0), R(0, 1), R(0, 2), t.x},
                             {R(1, 0), R(1, 1), R(1, 2), t.y},
                             {R(2, 0), R(2, 1), R(2, 2), t.z}};
    double A[4][4];
    for (int c = 0; c < 4; ++c) {
        A[0][c] = x1.x * P1[2][c] - P1[0][c];
        A[1][c] = x1.y * P1[2][c] - P1[1][c];
        A[2][c] = x2.x * P2[2][c] - P2[0][c];
        A[3][c] = x2.y * P2[2][c] - P2[1][c];
    }
    double ata[16] = {};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            for (int k = 0; k < 4; ++k) ata[r * 4 + c] += A[k][r] * A[k][c];
    double values[4], vectors[16];
    symmetricEigen(ata, 4, values, vectors);
    const double w = vectors[3 * 4 + 0];
    if (std::abs(w) < 1e-12) return false;
    X = {vectors[0] / w, vectors[4] / w, vectors[8] / w};
    return true;
}

TwoViewResult estimateRelativePose(const std::vector<Vec2>& points1, const std::vector<Vec2>& points2,
                                   double focalPx, const TwoViewParams& params, ThreadPool* pool) {
    TwoViewResult result;
    const int n = static_cast<int>(points1.size());
    result.inlier.assign(static_cast<std::size_t>(n), 0);
    if (n < 8 || points2.size() != points1.size()) return result;

    const double th = params.inlierThresholdPx / focalPx;
    const double th2 = th * th;

    // RANSAC in batches: samples come from one seeded RNG in order, models are
    // evaluated in parallel, and the best is chosen by (cost, index) so the
    // outcome never depends on thread timing.
    constexpr int kBatch = 64;
    std::mt19937 rng(params.seed);
    Mat3 bestE;
    Score best;
    int maxIterations = params.maxIterations;
    int iteration = 0;

    std::vector<std::array<int, 8>> samples(kBatch);
    std::vector<Mat3> models(kBatch);
    std::vector<Score> scores(kBatch);
    std::vector<std::uint8_t> valid(kBatch);

    while (iteration < maxIterations) {
        const int batch = std::min(kBatch, maxIterations - iteration);
        for (int b = 0; b < batch; ++b) {
            auto& s = samples[b];
            for (int k = 0; k < 8; ++k) {
                int candidate;
                bool duplicate;
                do {
                    candidate = static_cast<int>(rng() % static_cast<std::uint32_t>(n));
                    duplicate = std::find(s.begin(), s.begin() + k, candidate) != s.begin() + k;
                } while (duplicate);
                s[k] = candidate;
            }
        }
        parallelFor(pool, 0, batch, 1, [&](int b0, int b1) {
            for (int b = b0; b < b1; ++b) {
                valid[b] = eightPoint(points1, points2, samples[b].data(), 8, models[b], Model::Fundamental) ? 1 : 0;
                scores[b] = valid[b] ? scoreModel(models[b], points1, points2, th2) : Score{};
            }
        });
        // Best of this batch (lowest cost, then lowest index); locally optimize
        // it if it beats the current best.
        int batchBest = -1;
        for (int b = 0; b < batch; ++b) {
            if (valid[b] && (batchBest < 0 || scores[b].cost < scores[batchBest].cost)) batchBest = b;
        }
        if (batchBest >= 0 && scores[batchBest].cost < best.cost) {
            Score optimized;
            const Mat3 E = localOptimize(models[batchBest], points1, points2, th2, Model::Fundamental, optimized);
            best = optimized;
            bestE = E;
        }
        iteration += batch;

        // Adaptive stopping: enough iterations to draw one all-inlier sample.
        if (best.inliers > 0) {
            const double w = static_cast<double>(best.inliers) / n;
            const double pAllInliers = std::pow(w, 8);
            if (pAllInliers > 1e-12) {
                const double needed = std::log(1 - params.confidence) / std::log(1 - std::min(pAllInliers, 1 - 1e-12));
                maxIterations = std::min(params.maxIterations, static_cast<int>(std::ceil(needed)));
            }
        }
    }
    result.iterations = iteration;
    if (best.inliers < params.minInliers) return result;

    // bestE is a locally optimized rank-2 model. Fit the essential matrix on
    // its inliers (accurate with many points) and optimize once more.
    std::vector<int> inlierIdx;
    for (int i = 0; i < n; ++i) {
        if (sampsonError(bestE, points1[i], points2[i]) < th2) inlierIdx.push_back(i);
    }
    Mat3 E;
    if (!eightPoint(points1, points2, inlierIdx.data(), static_cast<int>(inlierIdx.size()), E, Model::Essential)) {
        return result;
    }
    Score finalScore;
    E = localOptimize(E, points1, points2, th2, Model::Essential, finalScore);
    inlierIdx.clear();
    for (int i = 0; i < n; ++i) {
        if (sampsonError(E, points1[i], points2[i]) < th2) inlierIdx.push_back(i);
    }

    for (int i : inlierIdx) result.inlier[i] = 1;
    result.inliers = static_cast<int>(inlierIdx.size());
    result.E = E;
    if (result.inliers < params.minInliers) return result;

    recoverPose(E, points1, points2, result);
    result.ok = result.inFront >= params.minInliers;
    return result;
}

}  // namespace vg
