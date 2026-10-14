#pragma once

// Minimal dense linear algebra for geometry code: fixed-size vectors and
// matrices plus a symmetric eigen solver. Double precision throughout.

#include <array>
#include <cmath>

namespace vg {

struct Vec2 {
    double x = 0, y = 0;
};

struct Vec3 {
    double x = 0, y = 0, z = 0;

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalized(const Vec3& a) {
    const double n = norm(a);
    return n > 0 ? a * (1.0 / n) : a;
}

// Row-major 3x3 matrix.
struct Mat3 {
    std::array<double, 9> m{};

    static Mat3 identity() {
        Mat3 r;
        r.m = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        return r;
    }
    double& operator()(int r, int c) { return m[r * 3 + c]; }
    double operator()(int r, int c) const { return m[r * 3 + c]; }

    Mat3 operator*(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r(i, j) = (*this)(i, 0) * o(0, j) + (*this)(i, 1) * o(1, j) + (*this)(i, 2) * o(2, j);
        return r;
    }
    Vec3 operator*(const Vec3& v) const {
        return {m[0] * v.x + m[1] * v.y + m[2] * v.z, m[3] * v.x + m[4] * v.y + m[5] * v.z,
                m[6] * v.x + m[7] * v.y + m[8] * v.z};
    }
    Mat3 operator*(double s) const {
        Mat3 r = *this;
        for (double& v : r.m) v *= s;
        return r;
    }
    Mat3 transposed() const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r(i, j) = (*this)(j, i);
        return r;
    }
    double determinant() const {
        return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
               m[2] * (m[3] * m[7] - m[4] * m[6]);
    }
    Vec3 column(int c) const { return {m[c], m[3 + c], m[6 + c]}; }
    void setColumn(int c, const Vec3& v) {
        m[c] = v.x;
        m[3 + c] = v.y;
        m[6 + c] = v.z;
    }
};

inline Mat3 skew(const Vec3& v) {
    Mat3 r;
    r.m = {0, -v.z, v.y, v.z, 0, -v.x, -v.y, v.x, 0};
    return r;
}

// Rotation angle of a rotation matrix, in radians [0, pi].
inline double rotationAngle(const Mat3& R) {
    const double c = (R(0, 0) + R(1, 1) + R(2, 2) - 1.0) * 0.5;
    return std::acos(c > 1 ? 1 : (c < -1 ? -1 : c));
}

// Eigen decomposition of a symmetric n x n matrix (row-major, n <= 9) by
// cyclic Jacobi rotations. Eigenvalues are returned in ascending order;
// eigenvector k is column k of `vectors`. `a` is destroyed.
void symmetricEigen(double* a, int n, double* values, double* vectors);

// Singular value decomposition of a 3x3 matrix: A = U * diag(s) * V^T with
// s sorted descending and U, V orthonormal.
void svd3(const Mat3& a, Mat3& u, Vec3& s, Mat3& v);

}  // namespace vg
