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
