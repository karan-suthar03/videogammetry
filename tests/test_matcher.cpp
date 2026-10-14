// Matcher checks on a synthetic image shifted by a known offset:
//  1. Most features match, and matches recover the shift.
//  2. Windowed search agrees with exhaustive search.
//  3. Multi-threaded output is identical to single-threaded.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "videogammetry/image.h"
#include "videogammetry/matcher.h"
#include "videogammetry/orb.h"
#include "videogammetry/thread_pool.h"

namespace {

int failures = 0;

void check(bool ok, const char* what, double value) {
    std::printf("  [%s] %s (%.3f)\n", ok ? "PASS" : "FAIL", what, value);
    if (!ok) ++failures;
}

vg::Image makeTexture(int w, int h) {
    vg::Image noise, img;
    noise.resize(w, h);
    std::mt19937 rng(99);
    for (auto& p : noise.data) p = static_cast<std::uint8_t>(rng() & 0xff);
    vg::gaussianBlur7(noise, img);
    for (auto& p : img.data) p = static_cast<std::uint8_t>(std::min(255, std::max(0, 128 + 4 * (p - 128))));
    return img;
}

vg::Image shift(const vg::Image& src, int dx, int dy) {
    vg::Image out;
    out.resize(src.width, src.height);
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            const int sx = std::min(std::max(x - dx, 0), src.width - 1);
            const int sy = std::min(std::max(y - dy, 0), src.height - 1);
            out.row(y)[x] = src.at(sx, sy);
        }
    }
    return out;
}

}  // namespace

int main() {
    constexpr int kDx = 7, kDy = -4;
    const vg::Image a = makeTexture(800, 600);
    const vg::Image b = shift(a, kDx, kDy);

    vg::OrbParams orbParams;
    orbParams.maxFeatures = 1500;
    vg::OrbExtractor orb(orbParams);
    std::vector<vg::Keypoint> ka, kb;
    std::vector<vg::Descriptor> da, db;
    orb.extract(a, ka, da);
    orb.extract(b, kb, db);

    vg::MatchParams params;  // defaults: windowed, ratio, cross-check, orientation
    std::vector<vg::Match> matches;
    vg::MatchStats stats;
    vg::matchFeatures(kb, db, ka, da, params, matches, &stats);

    int correct = 0;
    for (const vg::Match& m : matches) {
        const float ex = ka[m.train].x + kDx, ey = ka[m.train].y + kDy;
        const float dx = kb[m.query].x - ex, dy = kb[m.query].y - ey;
        if (dx * dx + dy * dy <= 2.0f * 2.0f * kb[m.query].scale * kb[m.query].scale) ++correct;
    }
    const double matchRate = double(matches.size()) / kb.size();
    const double precision = matches.empty() ? 0 : double(correct) / matches.size();
    std::printf("features %zu / %zu, matches %zu\n", ka.size(), kb.size(), matches.size());
    std::printf("checks:\n");
    // The feature grid is fixed to the image while the content moves, so each
    // cell keeps a partly different set of corners; ~50% overlap is expected.
    check(matchRate > 0.4, "more than 40% of features matched", matchRate);
    check(precision > 0.98, "more than 98% of matches recover the known shift", precision);

    // Exhaustive search must find the same matches here (the shift is well inside the radius).
    vg::MatchParams exhaustive = params;
    exhaustive.searchRadius = 0;
    std::vector<vg::Match> all;
    vg::matchFeatures(kb, db, ka, da, exhaustive, all);
    int common = 0;
    for (const vg::Match& m : matches) {
        for (const vg::Match& n : all) {
            if (n.query == m.query && n.train == m.train) {
                ++common;
                break;
            }
        }
    }
    const double overlap = matches.empty() ? 0 : double(common) / matches.size();
    check(overlap > 0.95, "windowed matches agree with exhaustive search", overlap);

    vg::ThreadPool pool(8);
    bool identical = true;
    for (int run = 0; run < 3; ++run) {
        std::vector<vg::Match> parallel;
        vg::matchFeatures(kb, db, ka, da, params, parallel, nullptr, &pool);
        identical = identical && parallel.size() == matches.size();
        for (std::size_t i = 0; identical && i < matches.size(); ++i) {
            identical = parallel[i].query == matches[i].query && parallel[i].train == matches[i].train &&
                        parallel[i].distance == matches[i].distance;
        }
    }
    check(identical, "8-thread matches identical to single-threaded (3 runs)", identical ? 1.0 : 0.0);

    std::printf("%s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
