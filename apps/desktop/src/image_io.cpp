#include "image_io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>

namespace {

bool endsWith(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin(),
                      [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) == b; });
}

void put16(std::ofstream& out, std::uint16_t v) {
    out.put(static_cast<char>(v & 0xff));
    out.put(static_cast<char>(v >> 8));
}

void put32(std::ofstream& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.put(static_cast<char>((v >> (8 * i)) & 0xff));
}

bool writeBmp(const vg::FrameView& frame, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    const std::uint32_t rowBytes = (static_cast<std::uint32_t>(frame.width) * 3 + 3) & ~3u;
    const std::uint32_t dataBytes = rowBytes * static_cast<std::uint32_t>(frame.height);

    out.put('B');
    out.put('M');
    put32(out, 54 + dataBytes);
    put32(out, 0);
    put32(out, 54);
    put32(out, 40);
    put32(out, static_cast<std::uint32_t>(frame.width));
    put32(out, static_cast<std::uint32_t>(frame.height));  // positive = bottom-up rows
    put16(out, 1);
    put16(out, 24);
    put32(out, 0);
    put32(out, dataBytes);
    put32(out, 2835);
    put32(out, 2835);
    put32(out, 0);
    put32(out, 0);

    std::vector<char> row(rowBytes, 0);
    for (int y = frame.height - 1; y >= 0; --y) {
        const std::uint8_t* src = frame.data + static_cast<std::size_t>(y) * frame.stride;
        for (int x = 0; x < frame.width; ++x) {
            row[x * 3 + 0] = static_cast<char>(src[x * 3 + 2]);  // BMP stores BGR
            row[x * 3 + 1] = static_cast<char>(src[x * 3 + 1]);
            row[x * 3 + 2] = static_cast<char>(src[x * 3 + 0]);
        }
        out.write(row.data(), rowBytes);
    }
    return static_cast<bool>(out);
}

bool writePpm(const vg::FrameView& frame, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "P6\n" << frame.width << " " << frame.height << "\n255\n";
    for (int y = 0; y < frame.height; ++y) {
        out.write(reinterpret_cast<const char*>(frame.data + static_cast<std::size_t>(y) * frame.stride),
                  static_cast<std::streamsize>(frame.width) * 3);
    }
    return static_cast<bool>(out);
}

}  // namespace

bool writeImage(const vg::FrameView& frame, const std::string& path) {
    return endsWith(path, ".bmp") ? writeBmp(frame, path) : writePpm(frame, path);
}

std::vector<std::uint8_t> drawKeypoints(const vg::FrameView& frame,
                                        const std::vector<vg::Keypoint>& keypoints,
                                        float processScale) {
    static constexpr std::uint8_t kColors[8][3] = {
        {255, 60, 60}, {255, 160, 0}, {240, 230, 0},  {60, 220, 60},
        {0, 200, 255}, {60, 100, 255}, {190, 80, 255}, {255, 80, 200}};

    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(frame.width) * frame.height * 3);
    for (int y = 0; y < frame.height; ++y) {
        const std::uint8_t* src = frame.data + static_cast<std::size_t>(y) * frame.stride;
        std::copy(src, src + frame.width * 3, rgb.data() + static_cast<std::size_t>(y) * frame.width * 3);
    }

    const auto plot = [&](float fx, float fy, const std::uint8_t* c) {
        const int x = static_cast<int>(std::lround(fx));
        const int y = static_cast<int>(std::lround(fy));
        if (x < 0 || y < 0 || x >= frame.width || y >= frame.height) return;
        std::uint8_t* p = rgb.data() + (static_cast<std::size_t>(y) * frame.width + x) * 3;
        p[0] = c[0];
        p[1] = c[1];
        p[2] = c[2];
    };

    for (const vg::Keypoint& kp : keypoints) {
        const std::uint8_t* c = kColors[kp.level % 8];
        const float cx = (kp.x + 0.5f) * processScale - 0.5f;
        const float cy = (kp.y + 0.5f) * processScale - 0.5f;
        const float r = 4.0f * kp.scale * processScale;
        const int steps = std::max(16, static_cast<int>(r * 6.3f));
        for (int i = 0; i < steps; ++i) {
            const float a = 6.2831853f * i / steps;
            plot(cx + r * std::cos(a), cy + r * std::sin(a), c);
        }
        for (float t = 0; t <= r; t += 0.5f) {
            plot(cx + t * std::cos(kp.angle), cy + t * std::sin(kp.angle), c);
        }
    }
    return rgb;
}

RgbImage resizeRgb(const vg::FrameView& frame, int width, int height) {
    RgbImage out;
    out.width = width;
    out.height = height;
    out.pixels.resize(static_cast<std::size_t>(width) * height * 3);
    const float sx = static_cast<float>(frame.width) / width;
    const float sy = static_cast<float>(frame.height) / height;

    for (int y = 0; y < height; ++y) {
        const float fy = std::clamp((y + 0.5f) * sy - 0.5f, 0.0f, static_cast<float>(frame.height - 1));
        const int y0 = static_cast<int>(fy);
        const int y1 = std::min(y0 + 1, frame.height - 1);
        const float wy = fy - y0;
        const std::uint8_t* r0 = frame.data + static_cast<std::size_t>(y0) * frame.stride;
        const std::uint8_t* r1 = frame.data + static_cast<std::size_t>(y1) * frame.stride;
        std::uint8_t* dst = out.pixels.data() + static_cast<std::size_t>(y) * width * 3;
        for (int x = 0; x < width; ++x) {
            const float fx = std::clamp((x + 0.5f) * sx - 0.5f, 0.0f, static_cast<float>(frame.width - 1));
            const int x0 = static_cast<int>(fx);
            const int x1 = std::min(x0 + 1, frame.width - 1);
            const float wx = fx - x0;
            for (int c = 0; c < 3; ++c) {
                const float top = r0[x0 * 3 + c] * (1 - wx) + r0[x1 * 3 + c] * wx;
                const float bot = r1[x0 * 3 + c] * (1 - wx) + r1[x1 * 3 + c] * wx;
                dst[x * 3 + c] = static_cast<std::uint8_t>(top * (1 - wy) + bot * wy + 0.5f);
            }
        }
    }
    return out;
}

RgbImage drawMatches(const RgbImage& top, const std::vector<vg::Keypoint>& topKeypoints,
                     const RgbImage& bottom, const std::vector<vg::Keypoint>& bottomKeypoints,
                     const std::vector<vg::Match>& matches, int maxDistance) {
    RgbImage out;
    out.width = std::max(top.width, bottom.width);
    out.height = top.height + bottom.height;
    out.pixels.assign(static_cast<std::size_t>(out.width) * out.height * 3, 0);

    // Copy both images, darkened so the lines stand out.
    const auto blit = [&](const RgbImage& img, int yOffset) {
        for (int y = 0; y < img.height; ++y) {
            const std::uint8_t* src = img.pixels.data() + static_cast<std::size_t>(y) * img.width * 3;
            std::uint8_t* dst = out.pixels.data() + static_cast<std::size_t>(y + yOffset) * out.width * 3;
            for (int i = 0; i < img.width * 3; ++i) dst[i] = static_cast<std::uint8_t>(src[i] * 6 / 10);
        }
    };
    blit(top, 0);
    blit(bottom, top.height);

    const auto plot = [&](int x, int y, const std::uint8_t* c) {
        if (x < 0 || y < 0 || x >= out.width || y >= out.height) return;
        std::uint8_t* p = out.pixels.data() + (static_cast<std::size_t>(y) * out.width + x) * 3;
        p[0] = c[0];
        p[1] = c[1];
        p[2] = c[2];
    };
    const auto line = [&](float x0, float y0, float x1, float y1, const std::uint8_t* c) {
        const int steps = static_cast<int>(std::max(std::abs(x1 - x0), std::abs(y1 - y0))) + 1;
        for (int i = 0; i <= steps; ++i) {
            const float t = static_cast<float>(i) / steps;
            plot(static_cast<int>(std::lround(x0 + (x1 - x0) * t)), static_cast<int>(std::lround(y0 + (y1 - y0) * t)), c);
        }
    };
    const auto dot = [&](float x, float y, const std::uint8_t* c) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) plot(static_cast<int>(std::lround(x)) + dx, static_cast<int>(std::lround(y)) + dy, c);
        }
    };

    for (const vg::Match& m : matches) {
        const vg::Keypoint& a = topKeypoints[m.train];
        const vg::Keypoint& b = bottomKeypoints[m.query];
        const float t = std::clamp(static_cast<float>(m.distance) / std::max(maxDistance, 1), 0.0f, 1.0f);
        const std::uint8_t color[3] = {static_cast<std::uint8_t>(255 * t), static_cast<std::uint8_t>(255 * (1 - t)), 40};
        line(a.x, a.y, b.x, b.y + top.height, color);
        dot(a.x, a.y, color);
        dot(b.x, b.y + top.height, color);
    }
    return out;
}

RgbImage drawFlow(const RgbImage& image, const std::vector<vg::Keypoint>& trainKeypoints,
                  const std::vector<vg::Keypoint>& queryKeypoints,
                  const std::vector<vg::Match>& matches, int maxDistance,
                  const std::vector<std::uint8_t>* inlier) {
    RgbImage out = image;
    for (auto& p : out.pixels) p = static_cast<std::uint8_t>(p * 6 / 10);

    const auto plot = [&](int x, int y, const std::uint8_t* c) {
        if (x < 0 || y < 0 || x >= out.width || y >= out.height) return;
        std::uint8_t* p = out.pixels.data() + (static_cast<std::size_t>(y) * out.width + x) * 3;
        p[0] = c[0];
        p[1] = c[1];
        p[2] = c[2];
    };

    // Outliers first so inliers are drawn on top.
    std::vector<std::size_t> order(matches.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    if (inlier) {
        std::stable_partition(order.begin(), order.end(), [&](std::size_t i) { return !(*inlier)[i]; });
    }

    for (std::size_t k : order) {
        const vg::Match& m = matches[k];
        const vg::Keypoint& a = trainKeypoints[m.train];
        const vg::Keypoint& b = queryKeypoints[m.query];
        std::uint8_t color[3];
        if (inlier) {
            const bool in = (*inlier)[k] != 0;
            color[0] = in ? 60 : 255;
            color[1] = in ? 230 : 40;
            color[2] = in ? 60 : 40;
        } else {
            const float t = std::clamp(static_cast<float>(m.distance) / std::max(maxDistance, 1), 0.0f, 1.0f);
            color[0] = static_cast<std::uint8_t>(255 * t);
            color[1] = static_cast<std::uint8_t>(255 * (1 - t));
            color[2] = 40;
        }
        const int steps = static_cast<int>(std::max(std::abs(b.x - a.x), std::abs(b.y - a.y))) + 1;
        for (int i = 0; i <= steps; ++i) {
            const float s = static_cast<float>(i) / steps;
            plot(static_cast<int>(std::lround(a.x + (b.x - a.x) * s)), static_cast<int>(std::lround(a.y + (b.y - a.y) * s)), color);
        }
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) plot(static_cast<int>(std::lround(b.x)) + dx, static_cast<int>(std::lround(b.y)) + dy, color);
        }
    }
    return out;
}

RgbImage drawTracking(const vg::FrameView& frame, const std::vector<vg::Keypoint>& keypoints,
                      const std::vector<int>& trackedIds, float processScale) {
    RgbImage out;
    out.width = frame.width;
    out.height = frame.height;
    out.pixels.resize(static_cast<std::size_t>(frame.width) * frame.height * 3);
    for (int y = 0; y < frame.height; ++y) {
        const std::uint8_t* src = frame.data + static_cast<std::size_t>(y) * frame.stride;
        std::uint8_t* dst = out.pixels.data() + static_cast<std::size_t>(y) * frame.width * 3;
        for (int i = 0; i < frame.width * 3; ++i) dst[i] = static_cast<std::uint8_t>(src[i] * 6 / 10);
    }
    const auto plot = [&](int x, int y, const std::uint8_t* c) {
        if (x < 0 || y < 0 || x >= out.width || y >= out.height) return;
        std::uint8_t* p = out.pixels.data() + (static_cast<std::size_t>(y) * out.width + x) * 3;
        p[0] = c[0];
        p[1] = c[1];
        p[2] = c[2];
    };
    static constexpr std::uint8_t kTracked[3] = {60, 240, 60};
    static constexpr std::uint8_t kUntracked[3] = {230, 60, 60};
    for (std::size_t i = 0; i < keypoints.size(); ++i) {
        const bool tracked = i < trackedIds.size() && trackedIds[i] >= 0;
        const std::uint8_t* c = tracked ? kTracked : kUntracked;
        const float cx = (keypoints[i].x + 0.5f) * processScale - 0.5f;
        const float cy = (keypoints[i].y + 0.5f) * processScale - 0.5f;
        const float r = tracked ? 5.0f : 3.0f;
        for (int k = 0; k < 24; ++k) {
            const float a = 6.2831853f * k / 24;
            plot(static_cast<int>(std::lround(cx + r * std::cos(a))), static_cast<int>(std::lround(cy + r * std::sin(a))), c);
        }
    }
    return out;
}

RgbImage renderMapViews(const std::vector<vg::Point>& points, const std::vector<vg::Vec3>& cameraPath,
                        int panelSize) {
    RgbImage out;
    out.width = panelSize * 2 + 10;
    out.height = panelSize;
    out.pixels.assign(static_cast<std::size_t>(out.width) * out.height * 3, 24);

    // Each view: (horizontal, vertical) world axes; vertical grows upward.
    struct View {
        int offsetX;
        double (*h)(double, double, double);
        double (*v)(double, double, double);
    };
    const View views[2] = {
        {0, [](double x, double, double) { return x; }, [](double, double, double z) { return z; }},
        {panelSize + 10, [](double, double, double z) { return z; }, [](double, double y, double) { return -y; }},
    };

    for (const View& view : views) {
        // Robust bounds: 2nd-98th percentile of points, always including the camera path.
        std::vector<double> hs, vs;
        for (const vg::Point& p : points) {
            hs.push_back(view.h(p.x, p.y, p.z));
            vs.push_back(view.v(p.x, p.y, p.z));
        }
        double h0 = 1e300, h1 = -1e300, v0 = 1e300, v1 = -1e300;
        if (!hs.empty()) {
            std::sort(hs.begin(), hs.end());
            std::sort(vs.begin(), vs.end());
            const std::size_t lo = hs.size() / 50, hi = hs.size() - 1 - hs.size() / 50;
            h0 = hs[lo];
            h1 = hs[hi];
            v0 = vs[lo];
            v1 = vs[hi];
        }
        for (const vg::Vec3& c : cameraPath) {
            h0 = std::min(h0, view.h(c.x, c.y, c.z));
            h1 = std::max(h1, view.h(c.x, c.y, c.z));
            v0 = std::min(v0, view.v(c.x, c.y, c.z));
            v1 = std::max(v1, view.v(c.x, c.y, c.z));
        }
        if (!(h1 > h0) || !(v1 > v0)) continue;
        const double span = std::max(h1 - h0, v1 - v0) * 1.05;
        const double hc = 0.5 * (h0 + h1), vc = 0.5 * (v0 + v1);
        const double scale = (panelSize - 1) / span;

        const auto toPixel = [&](double h, double v, int& px, int& py) {
            px = view.offsetX + static_cast<int>(std::lround((h - hc) * scale + panelSize / 2.0));
            py = static_cast<int>(std::lround(panelSize / 2.0 - (v - vc) * scale));
            return px >= view.offsetX && px < view.offsetX + panelSize && py >= 0 && py < panelSize;
        };
        const auto plot = [&](int x, int y, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
            std::uint8_t* p = out.pixels.data() + (static_cast<std::size_t>(y) * out.width + x) * 3;
            p[0] = r;
            p[1] = g;
            p[2] = b;
        };

        for (const vg::Point& p : points) {
            int x, y;
            if (toPixel(view.h(p.x, p.y, p.z), view.v(p.x, p.y, p.z), x, y)) plot(x, y, p.r, p.g, p.b);
        }
        for (std::size_t i = 0; i < cameraPath.size(); ++i) {
            const vg::Vec3& c = cameraPath[i];
            int x, y;
            if (!toPixel(view.h(c.x, c.y, c.z), view.v(c.x, c.y, c.z), x, y)) continue;
            const bool start = i == 0;
            for (int dy = -2; dy <= 2; ++dy) {
                for (int dx = -2; dx <= 2; ++dx) {
                    const int px = x + dx, py = y + dy;
                    if (px >= view.offsetX && px < view.offsetX + panelSize && py >= 0 && py < panelSize) {
                        plot(px, py, start ? 60 : 255, start ? 255 : 50, 50);
                    }
                }
            }
        }
    }
    return out;
}
