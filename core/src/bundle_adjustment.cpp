#include "videogammetry/bundle_adjustment.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace vg {

namespace {

using Vec6 = std::array<double, 6>;
using Mat6 = std::array<double, 36>;     // row-major 6x6
using Mat63 = std::array<double, 18>;    // row-major 6x3
using Mat33 = std::array<double, 9>;     // row-major 3x3

// Per-observation linearization: residual, robust weight, Jacobians.
struct Linearized {
    double r[2] = {0, 0};
    double w = 0;          // Huber weight (0 = observation ignored, e.g. behind camera)
    double Jc[2][6] = {};  // w.r.t. camera (omega, v)
    double Jx[2][3] = {};  // w.r.t. point
};

double huberCost(double s, double h) { return s <= h ? s * s : 2 * h * s - h * h; }

bool invert3(const Mat33& m, Mat33& inv) {
    const double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
    const double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const double det = a * A + b * B + c * C;
    if (!(std::abs(det) > 1e-300)) return false;
    const double s = 1.0 / det;
    inv = {A * s, -(b * i - c * h) * s, (b * f - c * e) * s,
           B * s, (a * i - c * g) * s, -(a * f - c * d) * s,
           C * s, -(a * h - b * g) * s, (a * e - b * d) * s};
    return true;
}

Pose applyPoseDelta(const Pose& p, const double* d) {
    const Mat3 dR = rotationFromAxisAngle({d[0], d[1], d[2]});
    Pose out{dR * p.R, dR * p.t + Vec3{d[3], d[4], d[5]}};
    out.R = nearestRotation(out.R);
    return out;
}

class Solver {
public:
    Solver(BAProblem& problem, double focal, const BAParams& params, ThreadPool* pool)
        : P_(problem), focal_(focal), h_(params.huberPx / focal), params_(params), pool_(pool) {
        const int nc = static_cast<int>(P_.poses.size());
        varIndex_.assign(static_cast<std::size_t>(nc), -1);
        for (int c = 0; c < nc; ++c) {
            if (c >= static_cast<int>(P_.fixedPose.size()) || !P_.fixedPose[c]) varIndex_[c] = numVar_++;
        }
        cameraObs_.resize(static_cast<std::size_t>(nc));
        pointObs_.resize(P_.points.size());
        for (int o = 0; o < static_cast<int>(P_.observations.size()); ++o) {
            cameraObs_[P_.observations[o].camera].push_back(o);
            pointObs_[P_.observations[o].point].push_back(o);
        }
    }

    BAReport run() {
        BAReport report;
        active_.assign(P_.observations.size(), 1);
        report.initialRmsPx = rmsPx(P_.poses, P_.points);

        report.iterations = optimize();
        if (params_.outlierThresholdPx > 0) {
            const double th = params_.outlierThresholdPx / focal_;
            int rejected = 0;
            for (std::size_t o = 0; o < P_.observations.size(); ++o) {
                const BAObservation& ob = P_.observations[o];
                double r[2];
                if (!residual(P_.poses[ob.camera], P_.points[ob.point], ob.xy, r) || std::hypot(r[0], r[1]) > th) {
                    active_[o] = 0;
                    ++rejected;
                }
            }
            if (rejected > 0) report.iterations += optimize();
            report.outliers = rejected;
        }
        report.inlier = active_;
        report.finalRmsPx = rmsPx(P_.poses, P_.points);
        report.ok = true;
        return report;
    }

private:
    // Levenberg-Marquardt over the active observations; returns accepted steps.
    int optimize() {
        double cost = totalCost(P_.poses, P_.points);
        double lambda = params_.initialLambda;
        int accepted = 0;
        std::vector<Pose> newPoses;
        std::vector<Vec3> newPoints;
        for (int it = 0; it < params_.maxIterations; ++it) {
            linearize();
            bool ok = false;
            for (int attempt = 0; attempt < 8 && !ok; ++attempt) {
                if (!solve(lambda, newPoses, newPoints)) {
                    lambda *= 10;
                    continue;
                }
                const double newCost = totalCost(newPoses, newPoints);
                if (newCost < cost) {
                    const double gain = (cost - newCost) / std::max(cost, 1e-300);
                    P_.poses = newPoses;
                    P_.points = newPoints;
                    cost = newCost;
                    lambda = std::max(lambda / 3, 1e-9);
                    ok = true;
                    ++accepted;
                    if (gain < 1e-6) it = params_.maxIterations;  // converged
                } else {
                    lambda *= 4;
                }
            }
            if (!ok) break;
        }
        return accepted;
    }

    // Residual of observation o for given pose/point; false if behind the camera.
    bool residual(const Pose& pose, const Vec3& X, const Vec2& obs, double r[2], Vec3* Xc = nullptr) const {
        const Vec3 c = pose.transform(X);
        if (c.z <= 1e-9) return false;
        r[0] = c.x / c.z - obs.x;
        r[1] = c.y / c.z - obs.y;
        if (Xc) *Xc = c;
        return true;
    }

    double totalCost(const std::vector<Pose>& poses, const std::vector<Vec3>& points) const {
        double c = 0;
        for (std::size_t i = 0; i < P_.observations.size(); ++i) {
            if (!active_[i]) continue;
            const BAObservation& o = P_.observations[i];
            double r[2];
            if (!residual(poses[o.camera], points[o.point], o.xy, r)) {
                c += huberCost(10 * h_, h_);  // constant penalty keeps the cost finite
                continue;
            }
            c += huberCost(std::hypot(r[0], r[1]), h_);
        }
        return c;
    }

    double rmsPx(const std::vector<Pose>& poses, const std::vector<Vec3>& points) const {
        double sum = 0;
        int n = 0;
        for (std::size_t i = 0; i < P_.observations.size(); ++i) {
            if (!active_[i]) continue;
            const BAObservation& o = P_.observations[i];
            double r[2];
            if (!residual(poses[o.camera], points[o.point], o.xy, r)) continue;
            sum += r[0] * r[0] + r[1] * r[1];
            ++n;
        }
        return n ? std::sqrt(sum / n) * focal_ : 0.0;
    }

    void linearize() {
        const int no = static_cast<int>(P_.observations.size());
        lin_.assign(static_cast<std::size_t>(no), Linearized{});
        parallelFor(pool_, 0, no, 256, [&](int o0, int o1) {
            for (int o = o0; o < o1; ++o) {
                const BAObservation& ob = P_.observations[o];
                const Pose& pose = P_.poses[ob.camera];
                Linearized& L = lin_[o];
                Vec3 c;
                if (!active_[o] || !residual(pose, P_.points[ob.point], ob.xy, L.r, &c)) continue;
                const double s = std::hypot(L.r[0], L.r[1]);
                L.w = s <= h_ ? 1.0 : h_ / s;
                const double iz = 1 / c.z;
                const double du[3] = {iz, 0, -c.x * iz * iz};
                const double dv[3] = {0, iz, -c.y * iz * iz};
                // Camera: X' = exp(omega) Xc + v  ->  dXc/domega = -[Xc]x, dXc/dv = I.
                const Mat3 dXdw = skew(c) * -1.0;
                for (int k = 0; k < 3; ++k) {
                    L.Jc[0][k] = du[0] * dXdw(0, k) + du[1] * dXdw(1, k) + du[2] * dXdw(2, k);
                    L.Jc[1][k] = dv[0] * dXdw(0, k) + dv[1] * dXdw(1, k) + dv[2] * dXdw(2, k);
                    L.Jc[0][3 + k] = du[k];
                    L.Jc[1][3 + k] = dv[k];
                    // Point: dXc/dX = R.
                    L.Jx[0][k] = du[0] * pose.R(0, k) + du[1] * pose.R(1, k) + du[2] * pose.R(2, k);
                    L.Jx[1][k] = dv[0] * pose.R(0, k) + dv[1] * pose.R(1, k) + dv[2] * pose.R(2, k);
                }
            }
        });

        // Camera blocks U, gradient gc; point blocks V, gradient gp; cross terms W.
        U_.assign(static_cast<std::size_t>(numVar_), Mat6{});
        gc_.assign(static_cast<std::size_t>(numVar_), Vec6{});
        parallelFor(pool_, 0, static_cast<int>(cameraObs_.size()), 1, [&](int c0, int c1) {
            for (int c = c0; c < c1; ++c) {
                const int k = varIndex_[c];
                if (k < 0) continue;
                for (int o : cameraObs_[c]) {
                    const Linearized& L = lin_[o];
                    if (L.w == 0) continue;
                    for (int a = 0; a < 6; ++a) {
                        gc_[k][a] += L.w * (L.Jc[0][a] * L.r[0] + L.Jc[1][a] * L.r[1]);
                        for (int b = 0; b < 6; ++b) {
                            U_[k][a * 6 + b] += L.w * (L.Jc[0][a] * L.Jc[0][b] + L.Jc[1][a] * L.Jc[1][b]);
                        }
                    }
                }
            }
        });
        V_.assign(P_.points.size(), Mat33{});
        gp_.assign(P_.points.size(), Vec3{});
        W_.assign(P_.observations.size(), Mat63{});
        parallelFor(pool_, 0, static_cast<int>(P_.points.size()), 64, [&](int p0, int p1) {
            for (int p = p0; p < p1; ++p) {
                double g[3] = {0, 0, 0};
                for (int o : pointObs_[p]) {
                    const Linearized& L = lin_[o];
                    if (L.w == 0) continue;
                    for (int a = 0; a < 3; ++a) {
                        g[a] += L.w * (L.Jx[0][a] * L.r[0] + L.Jx[1][a] * L.r[1]);
                        for (int b = 0; b < 3; ++b) {
                            V_[p][a * 3 + b] += L.w * (L.Jx[0][a] * L.Jx[0][b] + L.Jx[1][a] * L.Jx[1][b]);
                        }
                    }
                    if (varIndex_[P_.observations[o].camera] >= 0) {
                        for (int a = 0; a < 6; ++a)
                            for (int b = 0; b < 3; ++b)
                                W_[o][a * 3 + b] = L.w * (L.Jc[0][a] * L.Jx[0][b] + L.Jc[1][a] * L.Jx[1][b]);
                    }
                }
                gp_[p] = {g[0], g[1], g[2]};
            }
        });
    }

    // Solves the damped normal equations via the Schur complement and returns
    // the updated poses/points (not yet accepted).
    bool solve(double lambda, std::vector<Pose>& poses, std::vector<Vec3>& points) {
        const int np = static_cast<int>(P_.points.size());
        const int n = 6 * numVar_;

        // Damped, inverted point blocks.
        std::vector<Mat33> Vinv(static_cast<std::size_t>(np));
        std::vector<std::uint8_t> pointOk(static_cast<std::size_t>(np), 1);
        for (int p = 0; p < np; ++p) {
            Mat33 v = V_[p];
            for (int d = 0; d < 3; ++d) v[d * 4] = v[d * 4] * (1 + lambda) + 1e-12;
            if (!invert3(v, Vinv[p])) {
                pointOk[p] = 0;
                Vinv[p] = {};
            }
        }

        // Y_o = W_o * Vinv(point of o)  (6x3)
        std::vector<Mat63> Y(P_.observations.size());
        parallelFor(pool_, 0, static_cast<int>(P_.observations.size()), 256, [&](int o0, int o1) {
            for (int o = o0; o < o1; ++o) {
                const int p = P_.observations[o].point;
                if (varIndex_[P_.observations[o].camera] < 0 || !pointOk[p]) {
                    Y[o] = {};
                    continue;
                }
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 3; ++b) {
                        double s = 0;
                        for (int k = 0; k < 3; ++k) s += W_[o][a * 3 + k] * Vinv[p][k * 3 + b];
                        Y[o][a * 3 + b] = s;
                    }
            }
        });

        // Reduced camera system S dc = b. Each camera's block row is written by
        // one task only, so the parallel build is race-free and deterministic.
        std::vector<double> S(static_cast<std::size_t>(n) * n, 0.0), b(static_cast<std::size_t>(n), 0.0);
        parallelFor(pool_, 0, static_cast<int>(cameraObs_.size()), 1, [&](int c0, int c1) {
            for (int c = c0; c < c1; ++c) {
                const int k = varIndex_[c];
                if (k < 0) continue;
                for (int a = 0; a < 6; ++a) {
                    for (int bb = 0; bb < 6; ++bb) {
                        double u = U_[k][a * 6 + bb];
                        if (a == bb) u = u * (1 + lambda) + 1e-12;
                        S[static_cast<std::size_t>(k * 6 + a) * n + k * 6 + bb] += u;
                    }
                    b[k * 6 + a] = -gc_[k][a];
                }
                for (int o : cameraObs_[c]) {
                    const int p = P_.observations[o].point;
                    if (!pointOk[p]) continue;
                    for (int a = 0; a < 6; ++a) {  // b_k += Y_o gp
                        b[k * 6 + a] += Y[o][a * 3] * gp_[p].x + Y[o][a * 3 + 1] * gp_[p].y + Y[o][a * 3 + 2] * gp_[p].z;
                    }
                    for (int o2 : pointObs_[p]) {  // S_kk' -= Y_o W_o2^T
                        const int k2 = varIndex_[P_.observations[o2].camera];
                        if (k2 < 0) continue;
                        for (int a = 0; a < 6; ++a)
                            for (int bb = 0; bb < 6; ++bb) {
                                const double s = Y[o][a * 3] * W_[o2][bb * 3] + Y[o][a * 3 + 1] * W_[o2][bb * 3 + 1] +
                                                 Y[o][a * 3 + 2] * W_[o2][bb * 3 + 2];
                                S[static_cast<std::size_t>(k * 6 + a) * n + k2 * 6 + bb] -= s;
                            }
                    }
                }
            }
        });

        std::vector<double> dc(static_cast<std::size_t>(n), 0.0);
        if (n > 0 && !solveSpd(S.data(), b.data(), n, dc.data())) return false;

        poses = P_.poses;
        for (int c = 0; c < static_cast<int>(poses.size()); ++c) {
            if (varIndex_[c] >= 0) poses[c] = applyPoseDelta(P_.poses[c], &dc[varIndex_[c] * 6]);
        }
        // Back-substitute the points: dp = Vinv (-gp - sum W^T dc).
        points = P_.points;
        parallelFor(pool_, 0, np, 64, [&](int p0, int p1) {
            for (int p = p0; p < p1; ++p) {
                if (!pointOk[p]) continue;
                double rhs[3] = {-gp_[p].x, -gp_[p].y, -gp_[p].z};
                for (int o : pointObs_[p]) {
                    const int k = varIndex_[P_.observations[o].camera];
                    if (k < 0) continue;
                    for (int bb = 0; bb < 3; ++bb)
                        for (int a = 0; a < 6; ++a) rhs[bb] -= W_[o][a * 3 + bb] * dc[k * 6 + a];
                }
                Vec3 d;
                d.x = Vinv[p][0] * rhs[0] + Vinv[p][1] * rhs[1] + Vinv[p][2] * rhs[2];
                d.y = Vinv[p][3] * rhs[0] + Vinv[p][4] * rhs[1] + Vinv[p][5] * rhs[2];
                d.z = Vinv[p][6] * rhs[0] + Vinv[p][7] * rhs[1] + Vinv[p][8] * rhs[2];
                points[p] = P_.points[p] + d;
            }
        });
        return true;
    }

    BAProblem& P_;
    double focal_, h_;
    BAParams params_;
    ThreadPool* pool_;

    std::vector<std::uint8_t> active_;  // observations taking part (outliers removed after pass 1)
    std::vector<int> varIndex_;  // camera -> variable index, -1 if fixed
    int numVar_ = 0;
    std::vector<std::vector<int>> cameraObs_, pointObs_;

    std::vector<Linearized> lin_;
    std::vector<Mat6> U_;
    std::vector<Vec6> gc_;
    std::vector<Mat33> V_;
    std::vector<Vec3> gp_;
    std::vector<Mat63> W_;
};

}  // namespace

BAReport bundleAdjust(BAProblem& problem, double focalPx, const BAParams& params, ThreadPool* pool) {
    if (problem.points.empty() || problem.observations.empty()) return {};
    Solver solver(problem, focalPx, params, pool);
    return solver.run();
}

}  // namespace vg
