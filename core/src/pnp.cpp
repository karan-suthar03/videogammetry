#include "videogammetry/pnp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <random>

namespace vg {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Evaluates a polynomial with coefficients c[0] + c[1] x + ... + c[n] x^n.
double polyEval(const double* c, int n, double x) {
    double v = c[n];
    for (int i = n - 1; i >= 0; --i) v = v * x + c[i];
    return v;
}

// All real roots of a polynomial of degree n <= 4 (c[n] != 0), found by
// splitting the real line at the roots of the derivative (recursively) and
// bisecting every monotone interval with a sign change.
int realRoots(const double* c, int n, double* roots) {
    if (n == 1) {
        roots[0] = -c[0] / c[1];
        return 1;
    }
    double bound = 0;  // Cauchy bound: all roots lie in [-bound, bound]
    for (int i = 0; i < n; ++i) bound = std::max(bound, std::abs(c[i] / c[n]));
    bound += 1;

    double d[4];
    for (int i = 1; i <= n; ++i) d[i - 1] = i * c[i];
    double critical[4];
    const int nc = realRoots(d, n - 1, critical);
    std::sort(critical, critical + nc);

    double edges[6];
    int ne = 0;
    edges[ne++] = -bound;
    for (int i = 0; i < nc; ++i) {
        if (critical[i] > -bound && critical[i] < bound) edges[ne++] = critical[i];
    }
    edges[ne++] = bound;

    int count = 0;
    for (int i = 0; i + 1 < ne; ++i) {
        double lo = edges[i], hi = edges[i + 1];
        double flo = polyEval(c, n, lo), fhi = polyEval(c, n, hi);
        if (flo == 0) {
            if (count == 0 || std::abs(roots[count - 1] - lo) > 1e-12) roots[count++] = lo;
            continue;
        }
        if ((flo < 0) == (fhi < 0)) continue;
        for (int it = 0; it < 100; ++it) {
            const double mid = 0.5 * (lo + hi);
            const double fm = polyEval(c, n, mid);
            if ((fm < 0) == (flo < 0)) {
                lo = mid;
                flo = fm;
            } else {
                hi = mid;
            }
        }
        roots[count++] = 0.5 * (lo + hi);
    }
    return count;
}

// Rigid transform (no scale) mapping world points onto camera points: Kabsch.
Pose alignPoints(const Vec3 world[3], const Vec3 camera[3]) {
    Vec3 cw, cc;
    for (int i = 0; i < 3; ++i) {
        cw = cw + world[i];
        cc = cc + camera[i];
    }
    cw = cw * (1.0 / 3);
    cc = cc * (1.0 / 3);
    Mat3 h;  // sum (camera - cc)(world - cw)^T
    for (int i = 0; i < 3; ++i) {
        const Vec3 a = camera[i] - cc, b = world[i] - cw;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) h(r, c) += a[r] * b[c];
    }
    Pose p;
    p.R = nearestRotation(h);
    p.t = cc - p.R * cw;
    return p;
}

double projectError2(const Pose& pose, const Vec3& world, const Vec2& obs) {
    const Vec3 c = pose.transform(world);
    if (c.z <= 1e-9) return kInf;
    const double dx = c.x / c.z - obs.x, dy = c.y / c.z - obs.y;
    return dx * dx + dy * dy;
}

struct Score {
    double cost = kInf;
    int inliers = 0;
};

Score scorePose(const Pose& pose, const std::vector<Vec3>& w, const std::vector<Vec2>& o, double th2) {
    Score s{0, 0};
    for (std::size_t i = 0; i < w.size(); ++i) {
        const double e = projectError2(pose, w[i], o[i]);
        if (e < th2) {
            s.cost += e;
            ++s.inliers;
        } else {
            s.cost += th2;
        }
    }
    return s;
}

// Local optimization: refine on inliers selected with a shrinking threshold.
Pose localOptimize(const Pose& start, const std::vector<Vec3>& w, const std::vector<Vec2>& o, double focal,
                   double th2, Score& score) {
    Pose best = start;
    score = scorePose(start, w, o, th2);
    std::vector<std::uint8_t> mask(w.size());
    for (double factor : {3.0, 2.0, 1.0}) {
        const double t2 = th2 * factor * factor;
        int n = 0;
        for (std::size_t i = 0; i < w.size(); ++i) {
            mask[i] = projectError2(best, w[i], o[i]) < t2 ? 1 : 0;
            n += mask[i];
        }
        if (n < 4) continue;
        Pose refined = best;
        if (!refinePose(w, o, &mask, focal, std::sqrt(th2) * focal, 10, refined)) continue;
        const Score s = scorePose(refined, w, o, th2);
        if (s.cost < score.cost) {
            score = s;
            best = refined;
        }
    }
    return best;
}

}  // namespace

int solveP3P(const Vec3 bearings[3], const Vec3 worldPoints[3], Pose solutions[4]) {
    const Vec3 f1 = normalized(bearings[0]), f2 = normalized(bearings[1]), f3 = normalized(bearings[2]);
    const Vec3 &P1 = worldPoints[0], &P2 = worldPoints[1], &P3 = worldPoints[2];

    // Grunert: s2 = u s1, s3 = v s1; law of cosines on the three point pairs.
    const double a2 = dot(P2 - P3, P2 - P3), b2 = dot(P1 - P3, P1 - P3), c2 = dot(P1 - P2, P1 - P2);
    if (a2 < 1e-18 || b2 < 1e-18 || c2 < 1e-18) return 0;
    const double ca = dot(f2, f3), cb = dot(f1, f3), cg = dot(f1, f2);  // cos alpha, beta, gamma

    const double amc = (a2 - c2) / b2, apc = (a2 + c2) / b2, bmc = (b2 - c2) / b2, bma = (b2 - a2) / b2;
    double A[5];
    A[4] = (amc - 1) * (amc - 1) - 4 * c2 / b2 * ca * ca;
    A[3] = 4 * (amc * (1 - amc) * cb - (1 - apc) * ca * cg + 2 * c2 / b2 * ca * ca * cb);
    A[2] = 2 * (amc * amc - 1 + 2 * amc * amc * cb * cb + 2 * bmc * ca * ca - 4 * apc * ca * cb * cg +
                2 * bma * cg * cg);
    A[1] = 4 * (-amc * (1 + amc) * cb + 2 * a2 / b2 * cg * cg * cb - (1 - apc) * ca * cg);
    A[0] = (1 + amc) * (1 + amc) - 4 * a2 / b2 * cg * cg;
    if (std::abs(A[4]) < 1e-14) return 0;

    double vs[4];
    const int nv = realRoots(A, 4, vs);
    int count = 0;
    for (int k = 0; k < nv; ++k) {
        const double v = vs[k];
        if (v <= 0) continue;
        const double den = 2 * (cg - v * ca);
        if (std::abs(den) < 1e-12) continue;
        const double u = ((-1 + amc) * v * v - 2 * amc * cb * v + 1 + amc) / den;
        if (u <= 0) continue;
        const double q = 1 + v * v - 2 * v * cb;
        if (q <= 0) continue;
        const double s1 = std::sqrt(b2 / q), s2 = u * s1, s3 = v * s1;

        // Reject spurious roots: all three distances must be reproduced.
        const Vec3 X1 = f1 * s1, X2 = f2 * s2, X3 = f3 * s3;
        const double e = std::abs(dot(X2 - X3, X2 - X3) - a2) / a2 + std::abs(dot(X1 - X3, X1 - X3) - b2) / b2 +
                         std::abs(dot(X1 - X2, X1 - X2) - c2) / c2;
        if (e > 1e-4) continue;

        const Vec3 cam[3] = {X1, X2, X3};
        solutions[count++] = alignPoints(worldPoints, cam);
        if (count == 4) break;
    }
    return count;
}

double reprojectionErrorPx(const Pose& pose, const Vec3& world, const Vec2& observation, double focalPx) {
    const double e2 = projectError2(pose, world, observation);
    return std::isinf(e2) ? kInf : std::sqrt(e2) * focalPx;
}

bool refinePose(const std::vector<Vec3>& w, const std::vector<Vec2>& o, const std::vector<std::uint8_t>* mask,
                double focalPx, double huberPx, int iterations, Pose& pose) {
    const double huber = huberPx / focalPx;  // normalized units
    double lambda = 1e-4;
    const auto cost = [&](const Pose& p) {
        double c = 0;
        for (std::size_t i = 0; i < w.size(); ++i) {
            if (mask && !(*mask)[i]) continue;
            const Vec3 x = p.transform(w[i]);
            if (x.z <= 1e-9) {
                c += 2 * huber * 10;  // heavily penalize, keeps the cost finite
                continue;
            }
            const double r = std::hypot(x.x / x.z - o[i].x, x.y / x.z - o[i].y);
            c += r <= huber ? r * r : 2 * huber * r - huber * huber;
        }
        return c;
    };

    double current = cost(pose);
    for (int it = 0; it < iterations; ++it) {
        double H[36] = {}, g[6] = {};
        int used = 0;
        for (std::size_t i = 0; i < w.size(); ++i) {
            if (mask && !(*mask)[i]) continue;
            const Vec3 x = pose.transform(w[i]);
            if (x.z <= 1e-9) continue;
            const double iz = 1 / x.z;
            const double rx = x.x * iz - o[i].x, ry = x.y * iz - o[i].y;
            const double r = std::hypot(rx, ry);
            const double weight = r <= huber ? 1.0 : huber / r;

            // d(projection)/dX and dX/d(omega, v) for X' = exp(omega) X + v.
            const double du[3] = {iz, 0, -x.x * iz * iz};
            const double dv[3] = {0, iz, -x.y * iz * iz};
            double Ju[6], Jv[6];
            // dX/domega = -[X]x
            const Mat3 dXdw = skew(x) * -1.0;
            for (int k = 0; k < 3; ++k) {
                Ju[k] = du[0] * dXdw(0, k) + du[1] * dXdw(1, k) + du[2] * dXdw(2, k);
                Jv[k] = dv[0] * dXdw(0, k) + dv[1] * dXdw(1, k) + dv[2] * dXdw(2, k);
                Ju[3 + k] = du[k];
                Jv[3 + k] = dv[k];
            }
            for (int a = 0; a < 6; ++a) {
                g[a] += weight * (Ju[a] * rx + Jv[a] * ry);
                for (int b = 0; b < 6; ++b) H[a * 6 + b] += weight * (Ju[a] * Ju[b] + Jv[a] * Jv[b]);
            }
            ++used;
        }
        if (used < 3) return false;

        bool improved = false;
        for (int attempt = 0; attempt < 10 && !improved; ++attempt) {
            double Hd[36];
            for (int k = 0; k < 36; ++k) Hd[k] = H[k];
            for (int k = 0; k < 6; ++k) Hd[k * 6 + k] *= (1 + lambda);
            double neg[6], delta[6];
            for (int k = 0; k < 6; ++k) neg[k] = -g[k];
            if (!solveSpd(Hd, neg, 6, delta)) {
                lambda *= 10;
                continue;
            }
            const Mat3 dR = rotationFromAxisAngle({delta[0], delta[1], delta[2]});
            Pose candidate{dR * pose.R, dR * pose.t + Vec3{delta[3], delta[4], delta[5]}};
            candidate.R = nearestRotation(candidate.R);
            const double c = cost(candidate);
            if (c < current) {
                const double gain = (current - c) / std::max(current, 1e-300);
                pose = candidate;
                current = c;
                lambda = std::max(lambda * 0.3, 1e-9);
                improved = true;
                if (gain < 1e-10) return true;  // converged
            } else {
                lambda *= 10;
            }
        }
        if (!improved) break;  // no further progress possible
    }
    return true;
}

PnPResult solvePnPRansac(const std::vector<Vec3>& w, const std::vector<Vec2>& o, double focalPx,
                         const PnPParams& params, const Pose* prior, ThreadPool* pool) {
    PnPResult result;
    const int n = static_cast<int>(w.size());
    result.inlier.assign(static_cast<std::size_t>(n), 0);
    if (n < 4 || o.size() != w.size()) return result;

    const double th = params.inlierThresholdPx / focalPx;
    const double th2 = th * th;

    Pose bestPose;
    Score best;
    if (prior) {
        const Pose p = localOptimize(*prior, w, o, focalPx, th2, best);
        bestPose = p;
    }

    // Batched RANSAC: samples drawn in order from one seeded RNG, hypotheses
    // scored in parallel, best picked by (cost, index): deterministic.
    constexpr int kBatch = 32;
    std::mt19937 rng(params.seed);
    std::vector<std::array<int, 3>> samples(kBatch);
    std::vector<std::array<Pose, 4>> hyps(kBatch);
    std::vector<int> nh(kBatch);
    std::vector<std::array<Score, 4>> scores(kBatch);
    int maxIterations = params.maxIterations;
    int iteration = 0;

    const auto updateBound = [&] {
        if (best.inliers <= 0) return;
        const double ratio = static_cast<double>(best.inliers) / n;
        const double p = ratio * ratio * ratio;
        if (p > 1e-12) {
            const double needed = std::log(1 - params.confidence) / std::log(1 - std::min(p, 1 - 1e-12));
            maxIterations = std::min(params.maxIterations, static_cast<int>(std::ceil(needed)));
        }
    };
    updateBound();

    while (iteration < maxIterations) {
        const int batch = std::min(kBatch, maxIterations - iteration);
        for (int b = 0; b < batch; ++b) {
            auto& s = samples[b];
            for (int k = 0; k < 3; ++k) {
                int c;
                do {
                    c = static_cast<int>(rng() % static_cast<std::uint32_t>(n));
                } while (std::find(s.begin(), s.begin() + k, c) != s.begin() + k);
                s[k] = c;
            }
        }
        parallelFor(pool, 0, batch, 1, [&](int b0, int b1) {
            for (int b = b0; b < b1; ++b) {
                Vec3 bearings[3], points[3];
                for (int k = 0; k < 3; ++k) {
                    bearings[k] = {o[samples[b][k]].x, o[samples[b][k]].y, 1};
                    points[k] = w[samples[b][k]];
                }
                nh[b] = solveP3P(bearings, points, hyps[b].data());
                for (int h = 0; h < nh[b]; ++h) scores[b][h] = scorePose(hyps[b][h], w, o, th2);
            }
        });
        int bb = -1, bh = -1;
        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < nh[b]; ++h) {
                if (bb < 0 || scores[b][h].cost < scores[bb][bh].cost) {
                    bb = b;
                    bh = h;
                }
            }
        }
        if (bb >= 0 && scores[bb][bh].cost < best.cost) {
            Score optimized;
            const Pose p = localOptimize(hyps[bb][bh], w, o, focalPx, th2, optimized);
            if (optimized.cost < best.cost) {
                best = optimized;
                bestPose = p;
            }
        }
        iteration += batch;
        updateBound();
    }
    result.iterations = iteration;
    if (best.inliers < params.minInliers) return result;

    int count = 0;
    for (int i = 0; i < n; ++i) {
        result.inlier[i] = projectError2(bestPose, w[i], o[i]) < th2 ? 1 : 0;
        count += result.inlier[i];
    }
    result.pose = bestPose;
    result.inliers = count;
    result.ok = count >= params.minInliers;
    return result;
}

}  // namespace vg
