#include "videogammetry/bundle_adjustment.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace vg {

namespace {

using Vec6 = std::array<double, 6>;
using Mat6 = std::array<double, 36>;     // row-major 6x6
using Mat63 = std::array<double, 18>;    // row-major 6x3
using Mat33 = std::array<double, 9>;     // row-major 3x3
using Mat26 = std::array<double, 12>;    // up to 2 intrinsics x 6 camera parameters
using Mat23 = std::array<double, 6>;     // up to 2 intrinsics x 3 point coordinates

// Per-observation linearization: residual, robust weight, Jacobians.
struct Linearized {
    double r[2] = {0, 0};
    double w = 0;          // Huber weight (0 = observation ignored, e.g. behind camera)
    double Jc[2][6] = {};  // w.r.t. camera (omega, v)
    double Jx[2][3] = {};  // w.r.t. point
    double Jg[2][2] = {};  // w.r.t. the refined intrinsics (focal scale, k1)
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

// Symmetric positive-definite matrix stored by its lower skyline: row i keeps
// columns [start(i), i]. Cholesky factorization stays inside that envelope (no
// fill-in outside it), so the reduced camera system of a video -- keyframes
// share points only with keyframes near them in time, a band -- costs
// O(n * band^2) instead of O(n^3), with the same result as a dense solve.
class SkylineMatrix {
public:
    // rowStart[i] <= i: first stored column of row i.
    void reset(const std::vector<int>& rowStart) {
        start_ = rowStart;
        const std::size_t n = start_.size();
        base_.resize(n);
        std::size_t total = 0;
        for (std::size_t i = 0; i < n; ++i) {
            base_[i] = static_cast<std::ptrdiff_t>(total) - start_[i];  // element (i, j) at base_[i] + j
            total += i - static_cast<std::size_t>(start_[i]) + 1;
        }
        data_.assign(total, 0.0);
    }

    int size() const { return static_cast<int>(start_.size()); }
    int start(int i) const { return start_[i]; }
    // Element (i, j), start(i) <= j <= i.
    double& at(int i, int j) { return data_[static_cast<std::size_t>(base_[i] + j)]; }

    // In place: A = L L^T. False if A is not positive definite.
    bool factor() {
        double* d = data_.data();
        for (int i = 0; i < size(); ++i) {
            double* li = d + base_[i];
            for (int j = start_[i]; j <= i; ++j) {
                const double* lj = d + base_[j];
                double s = li[j];
                for (int k = std::max(start_[i], start_[j]); k < j; ++k) s -= li[k] * lj[k];
                if (j < i) {
                    li[j] = s / lj[j];
                } else {
                    if (!(s > 0)) return false;
                    li[i] = std::sqrt(s);
                }
            }
        }
        return true;
    }

    // Solves L L^T x = b in place (after factor()).
    void solve(std::vector<double>& b) const {
        const double* d = data_.data();
        for (int i = 0; i < size(); ++i) {  // L y = b
            const double* li = d + base_[i];
            double s = b[i];
            for (int k = start_[i]; k < i; ++k) s -= li[k] * b[k];
            b[i] = s / li[i];
        }
        for (int i = size() - 1; i >= 0; --i) {  // L^T x = y, column by column
            const double* li = d + base_[i];
            b[i] /= li[i];
            for (int k = start_[i]; k < i; ++k) b[k] -= li[k] * b[i];
        }
    }

private:
    std::vector<int> start_;
    std::vector<std::ptrdiff_t> base_;
    std::vector<double> data_;
};

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

        // Shared camera unknowns (intrinsics), in this order: focal scale, k1.
        k_ = params_.k1;
        if (params_.refineFocal) {
            focalSlot_ = G_++;
            priorWeight_[focalSlot_] = 1.0 / (params_.focalPriorSigma * params_.focalPriorSigma * focal_ * focal_);
        }
        if (params_.refineDistortion) {
            k1Slot_ = G_++;
            priorWeight_[k1Slot_] =
                1.0 / (params_.distortionPriorSigma * params_.distortionPriorSigma * focal_ * focal_);
        }

        // Envelope of the reduced camera system: camera k couples with every
        // camera that shares a point with it; row block k starts at the
        // lowest such camera.
        std::vector<int> firstCoupled(static_cast<std::size_t>(numVar_));
        for (int k = 0; k < numVar_; ++k) firstCoupled[k] = k;
        for (const std::vector<int>& obs : pointObs_) {
            int lowest = numVar_;
            for (int o : obs) {
                const int k = varIndex_[P_.observations[o].camera];
                if (k >= 0) lowest = std::min(lowest, k);
            }
            for (int o : obs) {
                const int k = varIndex_[P_.observations[o].camera];
                if (k >= 0) firstCoupled[k] = std::min(firstCoupled[k], lowest);
            }
        }
        rowStart_.resize(static_cast<std::size_t>(6 * numVar_));
        for (int k = 0; k < numVar_; ++k) {
            for (int a = 0; a < 6; ++a) rowStart_[k * 6 + a] = 6 * firstCoupled[k];
        }
        // Intrinsics couple with every camera: full rows at the end. Being
        // last, their marginal information is L_gg L_gg^T, from the trailing
        // block of the Cholesky factor.
        for (int g = 0; g < G_; ++g) rowStart_.push_back(0);
    }

    BAReport run() {
        BAReport report;
        active_.assign(P_.observations.size(), 1);
        report.initialRmsPx = rmsPx(P_.poses, P_.points);

        report.iterations = optimize();
        if (cancelled()) return report;  // not ok
        if (params_.outlierThresholdPx > 0) {
            const double th = params_.outlierThresholdPx / focal_;
            int rejected = 0;
            for (std::size_t o = 0; o < P_.observations.size(); ++o) {
                const BAObservation& ob = P_.observations[o];
                double r[2];
                if (!residual(P_.poses[ob.camera], P_.points[ob.point], ob.xy, r, s_, k_) ||
                    std::hypot(r[0], r[1]) > th * ob.sigma) {
                    active_[o] = 0;
                    ++rejected;
                }
            }
            if (rejected > 0) report.iterations += optimize();
            report.outliers = rejected;
        }
        if (cancelled()) return report;
        report.inlier = active_;
        report.finalRmsPx = rmsPx(P_.poses, P_.points);
        report.focalScale = s_;
        report.k1 = k_;
        if (G_ > 0) {
            intrinsicsUncertainty(report);  // linearizes at the solution
        } else {
            linearize();
        }
        pointUncertainty(report);
        report.ok = true;
        return report;
    }

private:
    bool cancelled() const { return params_.cancel && params_.cancel->load(std::memory_order_relaxed); }

    // Data-only standard deviations of the intrinsics at the solution (for
    // 1 px image noise): marginal information L_gg L_gg^T minus the priors'.
    void intrinsicsUncertainty(BAReport& report) {
        linearize();
        std::vector<Pose> poses;
        std::vector<Vec3> points;
        double s, k;
        if (!solve(0.0, poses, points, s, k)) return;
        const int base = S_.size() - G_;
        double L[2][2] = {};
        for (int a = 0; a < G_; ++a)
            for (int b = 0; b <= a; ++b) L[a][b] = S_.at(base + a, base + b);
        double info[2][2] = {};
        for (int a = 0; a < G_; ++a)
            for (int b = 0; b < G_; ++b) {
                double sum = 0;
                for (int c = 0; c < G_; ++c) sum += L[a][c] * L[b][c];
                info[a][b] = sum * focal_ * focal_;
            }
        for (int g = 0; g < G_; ++g) info[g][g] -= priorWeight_[g] * focal_ * focal_;
        double var[2] = {1e60, 1e60};
        if (G_ == 1) {
            if (info[0][0] > 0) var[0] = 1.0 / info[0][0];
        } else {
            const double det = info[0][0] * info[1][1] - info[0][1] * info[1][0];
            if (det > 0 && info[0][0] > 0) {
                var[0] = info[1][1] / det;
                var[1] = info[0][0] / det;
            }
        }
        if (focalSlot_ >= 0) report.focalStd = std::sqrt(var[focalSlot_]);
        if (k1Slot_ >= 0) report.k1Std = std::sqrt(var[k1Slot_]);
    }

    // Point covariance with the cameras fixed: the inverse of its information
    // block V (normalized units); its largest eigenvalue is the inverse of V's
    // smallest.
    void pointUncertainty(BAReport& report) {
        const double focal = focal_ * s_;
        std::vector<Vec3> centers(P_.poses.size());
        for (std::size_t c = 0; c < centers.size(); ++c) centers[c] = P_.poses[c].center();
        report.pointNoise.assign(P_.points.size(), std::numeric_limits<double>::infinity());
        parallelFor(pool_, 0, static_cast<int>(P_.points.size()), 256, [&](int p0, int p1) {
            for (int p = p0; p < p1; ++p) {
                double nearest = std::numeric_limits<double>::infinity();
                for (int o : pointObs_[p]) {
                    if (!active_[o]) continue;
                    nearest = std::min(nearest, norm(P_.points[p] - centers[P_.observations[o].camera]));
                }
                Mat33 v = V_[p];
                double values[3], vectors[9];
                symmetricEigen(v.data(), 3, values, vectors);
                const double smallest = std::min({values[0], values[1], values[2]});
                if (smallest > 0 && nearest > 0 && nearest < std::numeric_limits<double>::infinity()) {
                    report.pointNoise[p] = 1.0 / (focal * std::sqrt(smallest) * nearest);
                }
            }
        });
    }

    // Levenberg-Marquardt over the active observations; returns accepted steps.
    int optimize() {
        double cost = totalCost(P_.poses, P_.points, s_, k_);
        double lambda = params_.initialLambda;
        int accepted = 0;
        std::vector<Pose> newPoses;
        std::vector<Vec3> newPoints;
        double newS = s_, newK = k_;
        for (int it = 0; it < params_.maxIterations; ++it) {
            if (cancelled()) break;
            linearize();
            bool ok = false;
            for (int attempt = 0; attempt < 8 && !ok; ++attempt) {
                if (!solve(lambda, newPoses, newPoints, newS, newK)) {
                    lambda *= 10;
                    continue;
                }
                const double newCost = totalCost(newPoses, newPoints, newS, newK);
                if (newCost < cost) {
                    const double gain = (cost - newCost) / std::max(cost, 1e-300);
                    P_.poses = newPoses;
                    P_.points = newPoints;
                    s_ = newS;
                    k_ = newK;
                    cost = newCost;
                    lambda = std::max(lambda / 3, 1e-9);
                    ok = true;
                    ++accepted;
                    if (gain < params_.minRelativeDecrease) it = params_.maxIterations;  // converged
                } else {
                    lambda *= 4;
                }
            }
            if (!ok) break;
        }
        return accepted;
    }

    // Residual of an observation: the point projected with focal scale s and
    // radial distortion k (SIMPLE_RADIAL: m = s (1 + k |p|^2) p for the ideal
    // normalized point p), minus the observed (distorted) normalized position.
    // False if behind the camera.
    bool residual(const Pose& pose, const Vec3& X, const Vec2& obs, double r[2], double s, double k,
                  Vec3* Xc = nullptr) const {
        const Vec3 c = pose.transform(X);
        if (c.z <= 1e-9) return false;
        const double px = c.x / c.z, py = c.y / c.z;
        const double d = 1 + k * (px * px + py * py);
        r[0] = s * d * px - obs.x;
        r[1] = s * d * py - obs.y;
        if (Xc) *Xc = c;
        return true;
    }

    double priorCost(double s, double k) const {
        double c = 0;
        if (focalSlot_ >= 0) c += priorWeight_[focalSlot_] * (s - 1) * (s - 1);
        if (k1Slot_ >= 0) c += priorWeight_[k1Slot_] * (k - params_.k1) * (k - params_.k1);
        return c;
    }

    double totalCost(const std::vector<Pose>& poses, const std::vector<Vec3>& points, double s, double k) const {
        double c = priorCost(s, k);
        for (std::size_t i = 0; i < P_.observations.size(); ++i) {
            if (!active_[i]) continue;
            const BAObservation& o = P_.observations[i];
            double r[2];
            if (!residual(poses[o.camera], points[o.point], o.xy, r, s, k)) {
                c += huberCost(10 * h_, h_);  // constant penalty keeps the cost finite
                continue;
            }
            c += huberCost(std::hypot(r[0], r[1]) / o.sigma, h_);
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
            if (!residual(poses[o.camera], points[o.point], o.xy, r, s_, k_)) continue;
            sum += r[0] * r[0] + r[1] * r[1];
            ++n;
        }
        return n ? std::sqrt(sum / n) * focal_ : 0.0;
    }

    void linearize() {
        const int no = static_cast<int>(P_.observations.size());
        lin_.resize(static_cast<std::size_t>(no));  // every entry is (re)written below: no zeroing
        parallelFor(pool_, 0, no, 256, [&](int o0, int o1) {
            for (int o = o0; o < o1; ++o) {
                const BAObservation& ob = P_.observations[o];
                const Pose& pose = P_.poses[ob.camera];
                Linearized& L = lin_[o];
                Vec3 c;
                if (!active_[o] || !residual(pose, P_.points[ob.point], ob.xy, L.r, s_, k_, &c)) {
                    L.w = 0;  // ignored; other fields are not read
                    continue;
                }
                // Whitened: residual and Jacobians divided by the observation's sigma.
                const double ws = 1.0 / ob.sigma;
                L.r[0] *= ws;
                L.r[1] *= ws;
                const double rn = std::hypot(L.r[0], L.r[1]);
                L.w = rn <= h_ ? 1.0 : h_ / rn;

                const double iz = 1 / c.z;
                const double px = c.x * iz, py = c.y * iz;
                const double q = px * px + py * py;
                const double d = 1 + k_ * q;
                // dm/dp = s ((1 + k q) I + 2 k p p^T), then dp/dXc.
                const double a00 = ws * s_ * (d + 2 * k_ * px * px), a01 = ws * s_ * 2 * k_ * px * py;
                const double a11 = ws * s_ * (d + 2 * k_ * py * py);
                const double dpx[3] = {iz, 0, -px * iz};
                const double dpy[3] = {0, iz, -py * iz};
                double du[3], dv[3];
                for (int j = 0; j < 3; ++j) {
                    du[j] = a00 * dpx[j] + a01 * dpy[j];
                    dv[j] = a01 * dpx[j] + a11 * dpy[j];
                }
                if (focalSlot_ >= 0) {
                    L.Jg[0][focalSlot_] = ws * d * px;
                    L.Jg[1][focalSlot_] = ws * d * py;
                }
                if (k1Slot_ >= 0) {
                    L.Jg[0][k1Slot_] = ws * s_ * q * px;
                    L.Jg[1][k1Slot_] = ws * s_ * q * py;
                }
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

        // Camera blocks U, gradient gc, camera-intrinsics Ucg; point blocks V,
        // gradient gp, point-intrinsics Wgp; cross terms W.
        U_.assign(static_cast<std::size_t>(numVar_), Mat6{});
        gc_.assign(static_cast<std::size_t>(numVar_), Vec6{});
        Ucg_.assign(static_cast<std::size_t>(numVar_), Mat26{});
        parallelFor(pool_, 0, static_cast<int>(cameraObs_.size()), 1, [&](int c0, int c1) {
            for (int c = c0; c < c1; ++c) {
                const int k = varIndex_[c];
                if (k < 0) continue;
                for (int o : cameraObs_[c]) {
                    const Linearized& L = lin_[o];
                    if (L.w == 0) continue;
                    for (int a = 0; a < 6; ++a) {
                        for (int g = 0; g < G_; ++g) {
                            Ucg_[k][g * 6 + a] += L.w * (L.Jc[0][a] * L.Jg[0][g] + L.Jc[1][a] * L.Jg[1][g]);
                        }
                        gc_[k][a] += L.w * (L.Jc[0][a] * L.r[0] + L.Jc[1][a] * L.r[1]);
                        for (int b = 0; b < 6; ++b) {
                            U_[k][a * 6 + b] += L.w * (L.Jc[0][a] * L.Jc[0][b] + L.Jc[1][a] * L.Jc[1][b]);
                        }
                    }
                }
            }
        });
        // Every entry is written below (no zeroing pass over large arrays).
        V_.resize(P_.points.size());
        gp_.resize(P_.points.size());
        W_.resize(P_.observations.size());
        Wgp_.resize(P_.points.size());
        parallelFor(pool_, 0, static_cast<int>(P_.points.size()), 64, [&](int p0, int p1) {
            for (int p = p0; p < p1; ++p) {
                double g[3] = {0, 0, 0};
                Mat23 wg{};
                Mat33 v{};
                for (int o : pointObs_[p]) {
                    const Linearized& L = lin_[o];
                    if (L.w == 0) {
                        W_[o] = {};
                        continue;
                    }
                    for (int a = 0; a < 3; ++a) {
                        for (int gi = 0; gi < G_; ++gi) {
                            wg[gi * 3 + a] += L.w * (L.Jg[0][gi] * L.Jx[0][a] + L.Jg[1][gi] * L.Jx[1][a]);
                        }
                        g[a] += L.w * (L.Jx[0][a] * L.r[0] + L.Jx[1][a] * L.r[1]);
                        for (int b = 0; b < 3; ++b) {
                            v[a * 3 + b] += L.w * (L.Jx[0][a] * L.Jx[0][b] + L.Jx[1][a] * L.Jx[1][b]);
                        }
                    }
                    if (varIndex_[P_.observations[o].camera] >= 0) {
                        for (int a = 0; a < 6; ++a)
                            for (int b = 0; b < 3; ++b)
                                W_[o][a * 3 + b] = L.w * (L.Jc[0][a] * L.Jx[0][b] + L.Jc[1][a] * L.Jx[1][b]);
                    }
                }
                V_[p] = v;
                gp_[p] = {g[0], g[1], g[2]};
                Wgp_[p] = wg;
            }
        });

        // Intrinsics block and gradient, with the priors pulling toward the start.
        Ugg_ = {};
        gg_ = {};
        if (G_ > 0) for (const Linearized& L : lin_) {
            if (L.w == 0) continue;
            for (int a = 0; a < G_; ++a) {
                gg_[a] += L.w * (L.Jg[0][a] * L.r[0] + L.Jg[1][a] * L.r[1]);
                for (int b = 0; b < G_; ++b) Ugg_[a * 2 + b] += L.w * (L.Jg[0][a] * L.Jg[0][b] + L.Jg[1][a] * L.Jg[1][b]);
            }
        }
        if (focalSlot_ >= 0) {
            Ugg_[focalSlot_ * 3] += priorWeight_[focalSlot_];
            gg_[focalSlot_] += priorWeight_[focalSlot_] * (s_ - 1);
        }
        if (k1Slot_ >= 0) {
            Ugg_[k1Slot_ * 3] += priorWeight_[k1Slot_];
            gg_[k1Slot_] += priorWeight_[k1Slot_] * (k_ - params_.k1);
        }
    }

    // Solves the damped normal equations via the Schur complement and returns
    // the updated poses/points/intrinsics (not yet accepted).
    bool solve(double lambda, std::vector<Pose>& poses, std::vector<Vec3>& points, double& s, double& k1) {
        const int np = static_cast<int>(P_.points.size());
        const int nc = 6 * numVar_;
        const int n = nc + G_;

        // Damped, inverted point blocks.
        std::vector<Mat33>& Vinv = Vinv_;
        std::vector<std::uint8_t>& pointOk = pointOk_;
        Vinv.resize(static_cast<std::size_t>(np));
        pointOk.resize(static_cast<std::size_t>(np));
        parallelFor(pool_, 0, np, 256, [&](int p0, int p1) {
            for (int p = p0; p < p1; ++p) {
                Mat33 v = V_[p];
                for (int d = 0; d < 3; ++d) v[d * 4] = v[d * 4] * (1 + lambda) + 1e-12;
                pointOk[p] = invert3(v, Vinv[p]) ? 1 : 0;
                if (!pointOk[p]) Vinv[p] = {};
            }
        });

        // Y_o = W_o * Vinv(point of o)  (6x3)
        std::vector<Mat63>& Y = Y_;
        Y.resize(P_.observations.size());  // every entry is written below
        parallelFor(pool_, 0, static_cast<int>(P_.observations.size()), 256, [&](int o0, int o1) {
            for (int o = o0; o < o1; ++o) {
                const int p = P_.observations[o].point;
                if (varIndex_[P_.observations[o].camera] < 0 || !pointOk[p]) {
                    Y[o] = {};
                    continue;
                }
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 3; ++b) {
                        double sum = 0;
                        for (int j = 0; j < 3; ++j) sum += W_[o][a * 3 + j] * Vinv[p][j * 3 + b];
                        Y[o][a * 3 + b] = sum;
                    }
            }
        });

        // Reduced camera system S dc = b, lower triangle only (S is symmetric).
        // Each camera's block row is written by one task only, so the parallel
        // build is race-free and deterministic.
        S_.reset(rowStart_);
        std::vector<double> b(static_cast<std::size_t>(n), 0.0);
        parallelFor(pool_, 0, static_cast<int>(cameraObs_.size()), 1, [&](int c0, int c1) {
            for (int c = c0; c < c1; ++c) {
                const int k = varIndex_[c];
                if (k < 0) continue;
                for (int a = 0; a < 6; ++a) {
                    for (int bb = 0; bb <= a; ++bb) {
                        double u = U_[k][a * 6 + bb];
                        if (a == bb) u = u * (1 + lambda) + 1e-12;
                        S_.at(k * 6 + a, k * 6 + bb) += u;
                    }
                    b[k * 6 + a] = -gc_[k][a];
                }
                for (int o : cameraObs_[c]) {
                    const int p = P_.observations[o].point;
                    if (!pointOk[p]) continue;
                    for (int a = 0; a < 6; ++a) {  // b_k += Y_o gp
                        b[k * 6 + a] += Y[o][a * 3] * gp_[p].x + Y[o][a * 3 + 1] * gp_[p].y + Y[o][a * 3 + 2] * gp_[p].z;
                    }
                    for (int o2 : pointObs_[p]) {  // S_kk' -= Y_o W_o2^T  (k' <= k)
                        const int k2 = varIndex_[P_.observations[o2].camera];
                        if (k2 < 0 || k2 > k) continue;
                        for (int a = 0; a < 6; ++a)
                            for (int bb = 0; bb < (k2 == k ? a + 1 : 6); ++bb) {
                                const double v = Y[o][a * 3] * W_[o2][bb * 3] + Y[o][a * 3 + 1] * W_[o2][bb * 3 + 1] +
                                                 Y[o][a * 3 + 2] * W_[o2][bb * 3 + 2];
                                S_.at(k * 6 + a, k2 * 6 + bb) -= v;
                            }
                    }
                }
            }
        });

        // Intrinsics rows (after the cameras):
        //   S_gk = Ucg_k - sum_p Wgp^T Vinv W_o^T,  S_gh = Ugg - sum_p Wgp_g^T Vinv Wgp_h,
        //   b_g = -gg + sum_p Wgp_g^T Vinv gp.
        std::vector<Mat23> zg;
        if (G_ > 0) {
            zg.assign(static_cast<std::size_t>(np), Mat23{});
            double sgg[4] = {};
            double bg[2] = {};
            for (int a = 0; a < G_; ++a) {
                bg[a] = -gg_[a];
                for (int c = 0; c <= a; ++c) sgg[a * 2 + c] = Ugg_[a * 2 + c] * (a == c ? 1 + lambda : 1.0) + (a == c ? 1e-12 : 0.0);
                for (int k = 0; k < numVar_; ++k) {
                    for (int j = 0; j < 6; ++j) S_.at(nc + a, k * 6 + j) += Ucg_[k][a * 6 + j];
                }
            }
            for (int p = 0; p < np; ++p) {
                if (!pointOk[p]) continue;
                const Mat33& vi = Vinv[p];
                Mat23 z{};
                for (int a = 0; a < G_; ++a) {
                    const double* w = &Wgp_[p][a * 3];
                    for (int r = 0; r < 3; ++r) z[a * 3 + r] = vi[r * 3] * w[0] + vi[r * 3 + 1] * w[1] + vi[r * 3 + 2] * w[2];
                    bg[a] += z[a * 3] * gp_[p].x + z[a * 3 + 1] * gp_[p].y + z[a * 3 + 2] * gp_[p].z;
                    for (int c = 0; c <= a; ++c) {
                        sgg[a * 2 + c] -= Wgp_[p][c * 3] * z[a * 3] + Wgp_[p][c * 3 + 1] * z[a * 3 + 1] +
                                          Wgp_[p][c * 3 + 2] * z[a * 3 + 2];
                    }
                }
                zg[p] = z;
                for (int o : pointObs_[p]) {
                    const int k = varIndex_[P_.observations[o].camera];
                    if (k < 0) continue;
                    for (int a = 0; a < G_; ++a) {
                        for (int j = 0; j < 6; ++j) {
                            S_.at(nc + a, k * 6 + j) -= W_[o][j * 3] * z[a * 3] + W_[o][j * 3 + 1] * z[a * 3 + 1] +
                                                        W_[o][j * 3 + 2] * z[a * 3 + 2];
                        }
                    }
                }
            }
            for (int a = 0; a < G_; ++a) {
                b[nc + a] = bg[a];
                for (int c = 0; c <= a; ++c) S_.at(nc + a, nc + c) = sgg[a * 2 + c];
            }
        }

        if (n > 0) {
            if (!S_.factor()) return false;
            S_.solve(b);
        }
        const std::vector<double>& dc = b;
        double dg[2] = {0, 0};
        for (int a = 0; a < G_; ++a) dg[a] = dc[nc + a];
        s = s_ + (focalSlot_ >= 0 ? dg[focalSlot_] : 0.0);
        k1 = k_ + (k1Slot_ >= 0 ? dg[k1Slot_] : 0.0);

        poses = P_.poses;
        for (int c = 0; c < static_cast<int>(poses.size()); ++c) {
            if (varIndex_[c] >= 0) poses[c] = applyPoseDelta(P_.poses[c], &dc[varIndex_[c] * 6]);
        }
        // Back-substitute the points: dp = Vinv (-gp - sum W^T dc - Wgp^T dg).
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
                for (int a = 0; a < G_; ++a) {
                    for (int bb = 0; bb < 3; ++bb) rhs[bb] -= Wgp_[p][a * 3 + bb] * dg[a];
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
    std::vector<int> rowStart_;  // skyline envelope of the reduced camera system
    SkylineMatrix S_;

    std::vector<Linearized> lin_;
    std::vector<Mat6> U_;
    std::vector<Vec6> gc_;
    std::vector<Mat33> V_;
    std::vector<Vec3> gp_;
    std::vector<Mat63> W_;
    std::vector<Mat63> Y_;                 // per solve: W_o V^-1
    std::vector<Mat33> Vinv_;
    std::vector<std::uint8_t> pointOk_;

    // Intrinsics being refined (G_ of them): focal scale (refined focal =
    // s_ * focal_) and radial distortion k_.
    int G_ = 0, focalSlot_ = -1, k1Slot_ = -1;
    double s_ = 1.0, k_ = 0.0;
    double priorWeight_[2] = {0, 0};
    std::vector<Mat26> Ucg_;  // camera-intrinsics, per camera: [g * 6 + a]
    std::vector<Mat23> Wgp_;  // point-intrinsics, per point: [g * 3 + a]
    std::array<double, 4> Ugg_{};
    std::array<double, 2> gg_{};
};

}  // namespace

BAReport bundleAdjust(BAProblem& problem, double focalPx, const BAParams& params, ThreadPool* pool) {
    if (problem.points.empty() || problem.observations.empty()) return {};
    Solver solver(problem, focalPx, params, pool);
    return solver.run();
}

}  // namespace vg
