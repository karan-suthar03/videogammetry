#include "videogammetry/image.h"

#include <algorithm>
#include <cmath>

#include "videogammetry/thread_pool.h"

namespace vg {

namespace {
constexpr int kRowGrain = 16;  // rows per parallel chunk
}

void toGray(const FrameView& frame, Image& out, ThreadPool* pool) {
    out.resize(frame.width, frame.height);

    parallelFor(pool, 0, frame.height, kRowGrain, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            const std::uint8_t* src = frame.data + static_cast<std::size_t>(y) * frame.stride;
            std::uint8_t* dst = out.row(y);

            switch (frame.format) {
                case FrameView::Format::Gray8:
                    std::copy(src, src + frame.width, dst);
                    break;
                case FrameView::Format::RGB8:
                    for (int x = 0; x < frame.width; ++x, src += 3) {
                        dst[x] = static_cast<std::uint8_t>((77 * src[0] + 150 * src[1] + 29 * src[2] + 128) >> 8);
                    }
                    break;
                case FrameView::Format::RGBA8:
                    for (int x = 0; x < frame.width; ++x, src += 4) {
                        dst[x] = static_cast<std::uint8_t>((77 * src[0] + 150 * src[1] + 29 * src[2] + 128) >> 8);
                    }
                    break;
            }
        }
    });
}

void bilinearTaps(int srcLength, int dstLength, std::vector<int>& first, std::vector<int>& weight) {
    first.resize(static_cast<std::size_t>(dstLength));
    weight.resize(static_cast<std::size_t>(dstLength));
    const float s = static_cast<float>(srcLength) / dstLength;
    for (int i = 0; i < dstLength; ++i) {
        float f = (i + 0.5f) * s - 0.5f;
        f = std::clamp(f, 0.0f, static_cast<float>(srcLength - 1));
        const int i0 = std::max(std::min(static_cast<int>(f), srcLength - 2), 0);
        first[i] = i0;
        weight[i] = static_cast<int>((f - i0) * 256.0f + 0.5f);
    }
}

void resizeBilinear(const Image& src, int width, int height, Image& out, ThreadPool* pool) {
    out.resize(width, height);
    std::vector<int> x0, wx, y0s, wys;
    bilinearTaps(src.width, width, x0, wx);
    bilinearTaps(src.height, height, y0s, wys);

    parallelFor(pool, 0, height, kRowGrain, [&](int ya, int yb) {
        for (int y = ya; y < yb; ++y) {
            const int y0 = y0s[y];
            const int y1 = std::min(y0 + 1, src.height - 1);
            const int wy = wys[y];

            const std::uint8_t* r0 = src.row(y0);
            const std::uint8_t* r1 = src.row(y1);
            std::uint8_t* dst = out.row(y);

            for (int x = 0; x < width; ++x) {
                const int a = x0[x];
                const int b = std::min(a + 1, src.width - 1);
                const int top = r0[a] * (256 - wx[x]) + r0[b] * wx[x];
                const int bot = r1[a] * (256 - wx[x]) + r1[b] * wx[x];
                dst[x] = static_cast<std::uint8_t>((top * (256 - wy) + bot * wy + (1 << 15)) >> 16);
            }
        }
    });
}

void gaussianBlur7(const Image& src, Image& out, ThreadPool* pool) {
    // Sigma 2 kernel, integer weights summing to 256: 18 34 49 54 49 34 18.
    static constexpr int k[7] = {18, 34, 49, 54, 49, 34, 18};
    const int w = src.width;
    const int h = src.height;
    out.resize(w, h);

    std::vector<std::uint16_t> tmp(static_cast<std::size_t>(w) * h);

    // Horizontal pass (clamped borders) into 16-bit.
    parallelFor(pool, 0, h, kRowGrain, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            const std::uint8_t* s = src.row(y);
            std::uint16_t* t = tmp.data() + static_cast<std::size_t>(y) * w;
            for (int x = 0; x < w; ++x) {
                int sum = 0;
                if (x >= 3 && x < w - 3) {
                    for (int i = 0; i < 7; ++i) sum += k[i] * s[x + i - 3];
                } else {
                    for (int i = 0; i < 7; ++i) sum += k[i] * s[std::clamp(x + i - 3, 0, w - 1)];
                }
                t[x] = static_cast<std::uint16_t>(sum);
            }
        }
    });

    // Vertical pass back to 8-bit.
    parallelFor(pool, 0, h, kRowGrain, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            const std::uint16_t* rows[7];
            for (int i = 0; i < 7; ++i) {
                rows[i] = tmp.data() + static_cast<std::size_t>(std::clamp(y + i - 3, 0, h - 1)) * w;
            }
            std::uint8_t* d = out.row(y);
            for (int x = 0; x < w; ++x) {
                std::uint32_t sum = 0;
                for (int i = 0; i < 7; ++i) sum += static_cast<std::uint32_t>(k[i]) * rows[i][x];
                d[x] = static_cast<std::uint8_t>((sum + (1u << 15)) >> 16);
            }
        }
    });
}

}  // namespace vg
