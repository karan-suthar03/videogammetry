#include "videogammetry/linalg.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace vg {

void symmetricEigen(double* a, int n, double* values, double* vectors) {
    double v[81];
    for (int i = 0; i < n * n; ++i) v[i] = 0;
    for (int i = 0; i < n; ++i) v[i * n + i] = 1;

    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) off += a[p * n + q] * a[p * n + q];
        if (off < 1e-30) break;

        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = a[p * n + q];
                if (std::abs(apq) < 1e-300) continue;
                const double app = a[p * n + p], aqq = a[q * n + q];
                const double theta = (aqq - app) / (2 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1));
                const double c = 1 / std::sqrt(t * t + 1), s = t * c;

                for (int k = 0; k < n; ++k) {  // A = J^T A J, columns p and q
                    const double akp = a[k * n + p], akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {  // rows p and q
                    const double apk = a[p * n + k], aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    const double vkp = v[k * n + p], vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    int order[9];
    std::iota(order, order + n, 0);
    std::sort(order, order + n, [&](int i, int j) { return a[i * n + i] < a[j * n + j]; });
    for (int k = 0; k < n; ++k) {
        values[k] = a[order[k] * n + order[k]];
        for (int r = 0; r < n; ++r) vectors[r * n + k] = v[r * n + order[k]];
    }
}

void svd3(const Mat3& a, Mat3& u, Vec3& s, Mat3& v) {
    // V and s^2 from the eigen decomposition of A^T A (ascending -> reverse).
    const Mat3 ata = a.transposed() * a;
    double m[9], values[3], vectors[9];
    for (int i = 0; i < 9; ++i) m[i] = ata.m[i];
    symmetricEigen(m, 3, values, vectors);

    for (int k = 0; k < 3; ++k) {
        const int src = 2 - k;
        v.setColumn(k, {vectors[0 * 3 + src], vectors[1 * 3 + src], vectors[2 * 3 + src]});
    }
    const double s0 = std::sqrt(std::max(values[2], 0.0));
    const double s1 = std::sqrt(std::max(values[1], 0.0));
    const double s2 = std::sqrt(std::max(values[0], 0.0));
    s = {s0, s1, s2};

    // U columns: A v_i / s_i, completed to an orthonormal basis when s_i ~ 0.
    const double eps = 1e-12 * std::max(s0, 1e-300);
    Vec3 u0 = s0 > eps ? normalized(a * v.column(0)) : Vec3{1, 0, 0};
    Vec3 u1;
    if (s1 > eps) {
        u1 = a * v.column(1);
        u1 = normalized(u1 - u0 * dot(u0, u1));  // re-orthogonalize
    } else {
        u1 = normalized(cross(u0, std::abs(u0.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0}));
    }
    Vec3 u2;
    if (s2 > eps) {
        u2 = a * v.column(2);
        u2 = normalized(u2 - u0 * dot(u0, u2) - u1 * dot(u1, u2));
    } else {
        u2 = cross(u0, u1);
    }
    u.setColumn(0, u0);
    u.setColumn(1, u1);
    u.setColumn(2, u2);
}

}  // namespace vg

namespace vg {

Mat3 nearestRotation(const Mat3& m) {
    Mat3 u, v;
    Vec3 s;
    svd3(m, u, s, v);
    Mat3 r = u * v.transposed();
    if (r.determinant() < 0) {
        u.setColumn(2, -u.column(2));
        r = u * v.transposed();
    }
    return r;
}

bool solveSpd(const double* a, const double* b, int n, double* x) {
    std::vector<double> l(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = a[i * n + j];
            for (int k = 0; k < j; ++k) sum -= l[i * n + k] * l[j * n + k];
            if (i == j) {
                if (sum <= 0) return false;
                l[i * n + i] = std::sqrt(sum);
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    std::vector<double> y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {  // L y = b
        double sum = b[i];
        for (int k = 0; k < i; ++k) sum -= l[i * n + k] * y[k];
        y[i] = sum / l[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {  // L^T x = y
        double sum = y[i];
        for (int k = i + 1; k < n; ++k) sum -= l[k * n + i] * x[k];
        x[i] = sum / l[i * n + i];
    }
    return true;
}

}  // namespace vg
