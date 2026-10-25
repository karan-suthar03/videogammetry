#include "renderer.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr double kPi = 3.14159265358979323846;

constexpr std::uint32_t rgb(int r, int g, int b) {
    return (static_cast<std::uint32_t>(r) << 16) | (static_cast<std::uint32_t>(g) << 8) | static_cast<std::uint32_t>(b);
}

constexpr std::uint32_t kBackground = rgb(22, 24, 29);
constexpr std::uint32_t kTracked = rgb(70, 255, 110);
constexpr std::uint32_t kKeyframe = rgb(90, 150, 255);
constexpr std::uint32_t kPath = rgb(255, 175, 40);
constexpr std::uint32_t kCurrent = rgb(255, 60, 60);
constexpr std::uint32_t kLostCamera = rgb(150, 150, 150);

// Viewer camera model: pinhole at the viewport center.
struct Projector {
    vg::Pose pose;
    double focal = 1, cx = 0, cy = 0, nearZ = 1e-3;
};

// Clips the segment (x0,y0)-(x1,y1) to [0,w-1] x [0,h-1] (Liang-Barsky).
bool clipSegment(double& x0, double& y0, double& x1, double& y1, int w, int h) {
    double t0 = 0, t1 = 1;
    const double dx = x1 - x0, dy = y1 - y0;
    const double p[4] = {-dx, dx, -dy, dy};
    const double q[4] = {x0, (w - 1) - x0, y0, (h - 1) - y0};
    for (int i = 0; i < 4; ++i) {
        if (p[i] == 0) {
            if (q[i] < 0) return false;
            continue;
        }
        const double t = q[i] / p[i];
        if (p[i] < 0) {
            t0 = std::max(t0, t);
        } else {
            t1 = std::min(t1, t);
        }
        if (t0 > t1) return false;
    }
    const double ax = x0 + t0 * dx, ay = y0 + t0 * dy;
    x1 = x0 + t1 * dx;
    y1 = y0 + t1 * dy;
    x0 = ax;
    y0 = ay;
    return true;
}

// Depth uncertainty (fraction of the distance) as a color: green at 0.5% or
// less, through yellow (1.6%) to red at 5% or more (log scale); gray when unknown.
std::uint32_t noiseColor(float noise) {
    if (noise < 0) return rgb(128, 128, 128);
    const double t = std::clamp(std::log10(std::max(noise, 1e-6f) / 0.005), 0.0, 1.0);
    const double red = std::min(1.0, 2 * t), green = std::min(1.0, 2 * (1 - t));
    return rgb(static_cast<int>(40 + 215 * red), static_cast<int>(40 + 215 * green), 40);
}

void drawLine2D(double x0, double y0, double x1, double y1, std::uint32_t color, int thickness, Framebuffer& fb) {
    if (!clipSegment(x0, y0, x1, y1, fb.width, fb.height)) return;
    const int steps = std::max(1, static_cast<int>(std::ceil(std::max(std::abs(x1 - x0), std::abs(y1 - y0)))));
    const int lo = -(thickness - 1) / 2, hi = thickness / 2;
    for (int i = 0; i <= steps; ++i) {
        const double t = double(i) / steps;
        const int x = static_cast<int>(std::lround(x0 + t * (x1 - x0)));
        const int y = static_cast<int>(std::lround(y0 + t * (y1 - y0)));
        for (int oy = lo; oy <= hi; ++oy) {
            for (int ox = lo; ox <= hi; ++ox) {
                const int px = x + ox, py = y + oy;
                if (px >= 0 && py >= 0 && px < fb.width && py < fb.height) fb.pixels[py * fb.stride + px] = color;
            }
        }
    }
}

// World-space segment: clipped against the near plane, then projected.
void drawLine3D(const vg::Vec3& a, const vg::Vec3& b, std::uint32_t color, int thickness, const Projector& pr,
                Framebuffer& fb) {
    vg::Vec3 ca = pr.pose.transform(a), cb = pr.pose.transform(b);
    if (ca.z < pr.nearZ && cb.z < pr.nearZ) return;
    if (ca.z < pr.nearZ || cb.z < pr.nearZ) {
        const double t = (pr.nearZ - ca.z) / (cb.z - ca.z);
        const vg::Vec3 cut = ca + (cb - ca) * t;
        (ca.z < pr.nearZ ? ca : cb) = cut;
    }
    drawLine2D(pr.focal * ca.x / ca.z + pr.cx, pr.focal * ca.y / ca.z + pr.cy, pr.focal * cb.x / cb.z + pr.cx,
               pr.focal * cb.y / cb.z + pr.cy, color, thickness, fb);
}

// Camera as a pyramid from its center to the image rectangle at distance
// `size`, with a small triangle on the top edge so roll is visible.
void drawFrustum(const vg::Pose& camera, const LiveSnapshot& scene, double size, std::uint32_t color, int thickness,
                 const Projector& pr, Framebuffer& fb) {
    const vg::Vec3 c = camera.center();
    const vg::Mat3 rt = camera.R.transposed();
    const double w = scene.imageWidth, h = scene.imageHeight, f = scene.focal;
    const auto at = [&](double u, double v) {
        const vg::Vec3 ray{(u - w / 2) / f, (v - h / 2) / f, 1.0};
        return c + rt * (ray * size);
    };
    const vg::Vec3 corners[4] = {at(0, 0), at(w, 0), at(w, h), at(0, h)};
    for (int i = 0; i < 4; ++i) {
        drawLine3D(c, corners[i], color, thickness, pr, fb);
        drawLine3D(corners[i], corners[(i + 1) % 4], color, thickness, pr, fb);
    }
    const vg::Vec3 t0 = at(0.35 * w, 0), t1 = at(0.5 * w, -0.22 * h), t2 = at(0.65 * w, 0);
    drawLine3D(t0, t1, color, thickness, pr, fb);
    drawLine3D(t1, t2, color, thickness, pr, fb);
}

}  // namespace

void OrbitView::orbit(double dxPx, double dyPx) {
    yaw += dxPx * 0.006;
    pitch = std::clamp(pitch + dyPx * 0.006, -1.5, 1.5);
}

void OrbitView::pan(double dxPx, double dyPx, double sceneScale, int viewportWidth) {
    const vg::Pose p = pose(sceneScale);
    const double focal = 0.5 * viewportWidth / std::tan(fovDeg * kPi / 360.0);
    const double worldPerPx = zoom * sceneScale / focal;
    const vg::Vec3 right{p.R(0, 0), p.R(0, 1), p.R(0, 2)};
    const vg::Vec3 down{p.R(1, 0), p.R(1, 1), p.R(1, 2)};
    target = target - right * (dxPx * worldPerPx) - down * (dyPx * worldPerPx);
    follow = false;
}

void OrbitView::zoomBy(double wheelSteps) { zoom = std::clamp(zoom * std::pow(0.88, wheelSteps), 0.05, 200.0); }

void OrbitView::update(const LiveSnapshot& scene) {
    if (!follow || scene.keyframes.empty()) return;
    const vg::Vec3 desired = scene.current.center();
    target = target + (desired - target) * 0.25;
}

void OrbitView::reset() {
    const double keepFov = fovDeg;
    *this = OrbitView{};
    fovDeg = keepFov;
}

vg::Pose OrbitView::pose(double sceneScale) const {
    const vg::Vec3 f{std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw)};
    const vg::Vec3 eye = target - f * (zoom * sceneScale);
    const vg::Vec3 x = vg::normalized(vg::cross(vg::Vec3{0, 1, 0}, f));
    const vg::Vec3 y = vg::cross(f, x);
    vg::Pose p;
    p.R.m = {x.x, x.y, x.z, y.x, y.y, y.z, f.x, f.y, f.z};
    p.t = -(p.R * eye);
    return p;
}

void SceneRenderer::render(const LiveSnapshot& scene, const OrbitView& view, const RenderOptions& options,
                           Framebuffer& fb) {
    for (int y = 0; y < fb.height; ++y) std::fill_n(fb.pixels + y * fb.stride, fb.width, kBackground);
    if (fb.width <= 0 || fb.height <= 0) return;

    const double scale = scene.sceneScale;
    Projector pr;
    const bool throughCamera = view.attached && !scene.keyframes.empty();
    if (throughCamera) {
        // The video camera's own view: its pose, and its focal length scaled
        // so the whole video frame fits the window.
        pr.pose = scene.current;
        pr.focal = scene.focal * std::min(double(fb.width) / scene.imageWidth, double(fb.height) / scene.imageHeight);
    } else {
        pr.pose = view.pose(scale);
        pr.focal = 0.5 * fb.width / std::tan(view.fovDeg * kPi / 360.0);
    }
    pr.cx = 0.5 * fb.width;
    pr.cy = 0.5 * fb.height;
    pr.nearZ = 1e-3 * scale;

    // Points: z-buffered square splats.
    depth_.assign(static_cast<std::size_t>(fb.width) * fb.height, std::numeric_limits<float>::max());
    float r[9], t[3];
    for (int i = 0; i < 9; ++i) r[i] = static_cast<float>(pr.pose.R.m[i]);
    t[0] = static_cast<float>(pr.pose.t.x);
    t[1] = static_cast<float>(pr.pose.t.y);
    t[2] = static_cast<float>(pr.pose.t.z);
    const float focal = static_cast<float>(pr.focal), cx = static_cast<float>(pr.cx), cy = static_cast<float>(pr.cy);
    const float nearZ = static_cast<float>(pr.nearZ);
    const auto splat = [&](float z, float x, float y, int size, std::uint32_t color) {
        const int u0 = static_cast<int>(std::floor(focal * x / z + cx)) - (size - 1) / 2;
        const int v0 = static_cast<int>(std::floor(focal * y / z + cy)) - (size - 1) / 2;
        if (u0 + size <= 0 || v0 + size <= 0 || u0 >= fb.width || v0 >= fb.height) return;
        for (int v = std::max(v0, 0); v < std::min(v0 + size, fb.height); ++v) {
            for (int u = std::max(u0, 0); u < std::min(u0 + size, fb.width); ++u) {
                float& d = depth_[static_cast<std::size_t>(v) * fb.width + u];
                if (z < d) {
                    d = z;
                    fb.pixels[v * fb.stride + u] = color;
                }
            }
        }
    };
    if (options.showDense && scene.dense) {
        for (const auto& chunk : *scene.dense)
            for (const vg::Point& p : *chunk) {
                const float z = r[6] * p.x + r[7] * p.y + r[8] * p.z + t[2];
                if (z < nearZ) continue;
                splat(z, r[0] * p.x + r[1] * p.y + r[2] * p.z + t[0], r[3] * p.x + r[4] * p.y + r[5] * p.z + t[1],
                      options.pointSize, rgb(p.r, p.g, p.b));
            }
    }
    for (std::size_t i = 0; i < scene.points.size(); ++i) {
        const vg::Point& p = scene.points[i];
        const float z = r[6] * p.x + r[7] * p.y + r[8] * p.z + t[2];
        if (z < nearZ) continue;
        const float x = r[0] * p.x + r[1] * p.y + r[2] * p.z + t[0];
        const float y = r[3] * p.x + r[4] * p.y + r[5] * p.z + t[1];
        const bool tracked = options.highlightTracked && scene.tracked[i];
        const int size = options.pointSize + (tracked ? 1 : 0);
        const int u0 = static_cast<int>(std::floor(focal * x / z + cx)) - (size - 1) / 2;
        const int v0 = static_cast<int>(std::floor(focal * y / z + cy)) - (size - 1) / 2;
        if (u0 + size <= 0 || v0 + size <= 0 || u0 >= fb.width || v0 >= fb.height) continue;
        const std::uint32_t color = tracked                ? kTracked
                                    : options.colorByNoise ? noiseColor(scene.noise[i])
                                                           : rgb(p.r, p.g, p.b);
        for (int v = std::max(v0, 0); v < std::min(v0 + size, fb.height); ++v) {
            for (int u = std::max(u0, 0); u < std::min(u0 + size, fb.width); ++u) {
                float& d = depth_[static_cast<std::size_t>(v) * fb.width + u];
                if (z < d) {
                    d = z;
                    fb.pixels[v * fb.stride + u] = color;
                }
            }
        }
    }

    // World axes at the first camera: x red, y (down) green, z (forward) blue.
    const double axis = 0.15 * scale;
    drawLine3D({}, {axis, 0, 0}, rgb(220, 70, 70), 1, pr, fb);
    drawLine3D({}, {0, axis, 0}, rgb(70, 200, 70), 1, pr, fb);
    drawLine3D({}, {0, 0, axis}, rgb(80, 120, 240), 1, pr, fb);

    // Camera path through the keyframes, then to the current camera.
    for (std::size_t i = 1; i < scene.keyframes.size(); ++i) {
        drawLine3D(scene.keyframes[i - 1].center(), scene.keyframes[i].center(), kPath, 2, pr, fb);
    }
    if (scene.currentValid && !scene.keyframes.empty()) {
        drawLine3D(scene.keyframes.back().center(), scene.current.center(), kPath, 2, pr, fb);
    }

    // Through the camera, recent keyframes sit right around the eye and would
    // cover the view.
    if (options.showKeyframes && !throughCamera) {
        for (const vg::Pose& k : scene.keyframes) drawFrustum(k, scene, 0.06 * scale, kKeyframe, 1, pr, fb);
    }
    if (!scene.keyframes.empty() && !throughCamera) {  // looking through it: nothing to draw
        drawFrustum(scene.current, scene, 0.14 * scale, scene.currentValid ? kCurrent : kLostCamera, 2, pr, fb);
    }
}

void blitImage(const RgbImage& image, int x, int y, Framebuffer& fb) {
    if (image.width <= 0) return;
    for (int row = -1; row <= image.height; ++row) {
        const int py = y + row;
        if (py < 0 || py >= fb.height) continue;
        for (int col = -1; col <= image.width; ++col) {
            const int px = x + col;
            if (px < 0 || px >= fb.width) continue;
            if (row < 0 || col < 0 || row == image.height || col == image.width) {
                fb.pixels[py * fb.stride + px] = rgb(120, 120, 130);
                continue;
            }
            const std::uint8_t* s = image.pixels.data() + (static_cast<std::size_t>(row) * image.width + col) * 3;
            fb.pixels[py * fb.stride + px] = rgb(s[0], s[1], s[2]);
        }
    }
}
