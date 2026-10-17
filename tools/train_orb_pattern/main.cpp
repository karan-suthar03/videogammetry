// Learns the 256 BRIEF tests used by the ORB descriptor (rBRIEF, Rublee et al. 2011).
//
// 1. Run ORB detection on frames from the given videos.
// 2. Evaluate a large pool of random candidate tests on every keypoint.
// 3. Greedily pick tests whose mean is closest to 0.5 and whose correlation
//    with all previously picked tests stays below a threshold.
// 4. Write the result as core/src/orb_pattern.inc.
//
// Usage: train_orb_pattern [options] <video>... -o <orb_pattern.inc>
//   --start <sec>   skip frames before this time (default 0)
//   --step <n>      use every n-th frame (default 5)
//   --candidates <n> size of the candidate pool (default 8192)

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "video_reader.h"
#include "videogammetry/image.h"
#include "videogammetry/orb.h"
#include "videogammetry/orb_training.h"

namespace {

// Bit matrix stored per candidate test (column-major), packed into uint64 words.
struct TestBits {
    std::vector<std::vector<std::uint64_t>> columns;  // [candidate][word]
    std::size_t samples = 0;

    void addSample(const std::vector<bool>& values) {
        if (samples % 64 == 0) {
            for (auto& c : columns) c.push_back(0);
        }
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (values[i]) columns[i].back() |= std::uint64_t{1} << (samples % 64);
        }
        ++samples;
    }

    std::size_t ones(std::size_t c) const {
        std::size_t n = 0;
        for (std::uint64_t w : columns[c]) n += std::popcount(w);
        return n;
    }

    std::size_t bothOnes(std::size_t a, std::size_t b) const {
        std::size_t n = 0;
        const auto& ca = columns[a];
        const auto& cb = columns[b];
        for (std::size_t i = 0; i < ca.size(); ++i) n += std::popcount(ca[i] & cb[i]);
        return n;
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> inputs;
    std::string output;
    double startSec = 0.0;
    int step = 5;
    int candidateCount = 8192;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (std::strcmp(argv[i], "--start") == 0 && i + 1 < argc) {
            startSec = std::strtod(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--step") == 0 && i + 1 < argc) {
            step = std::max(1, static_cast<int>(std::strtol(argv[++i], nullptr, 10)));
        } else if (std::strcmp(argv[i], "--candidates") == 0 && i + 1 < argc) {
            candidateCount = std::max(256, static_cast<int>(std::strtol(argv[++i], nullptr, 10)));
        } else {
            inputs.push_back(argv[i]);
        }
    }
    if (inputs.empty() || output.empty()) {
        std::fprintf(stderr, "usage: train_orb_pattern [--start s] [--step n] [--candidates n] <video>... -o <file.inc>\n");
        return 1;
    }

    const std::vector<vg::BriefTest> candidates = vg::randomBriefTests(candidateCount, 0xb41efu);
    TestBits bits;
    bits.columns.resize(candidates.size());

    // Same preprocessing as the pipeline: gray, downscale to 960 px wide.
    vg::OrbParams orbParams;
    orbParams.subpixel = false;  // patches are sampled at the corner's whole level pixel
    vg::OrbExtractor orb(orbParams);
    vg::Image gray, small;
    std::vector<vg::Keypoint> keypoints;
    std::vector<vg::Descriptor> descriptors;
    std::vector<bool> values(candidates.size());
    long framesUsed = 0;

    for (const std::string& path : inputs) {
        VideoReader reader;
        std::string error;
        if (!reader.open(path, error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        vg::FrameView frame;
        long index = 0;
        while (reader.read(frame)) {
            const bool use = frame.timestamp >= startSec && index % step == 0;
            ++index;
            if (!use) continue;

            vg::toGray(frame, gray);
            const vg::Image* input = &gray;
            if (gray.width > 960) {
                vg::resizeBilinear(gray, 960, static_cast<int>(std::lround(gray.height * 960.0 / gray.width)), small);
                input = &small;
            }
            orb.extract(*input, keypoints, descriptors);

            for (const vg::Keypoint& kp : keypoints) {
                const vg::Image& blurred = orb.blurredLevel(kp.level);
                const int x = static_cast<int>(std::lround((kp.x + 0.5f) / kp.scale - 0.5f));
                const int y = static_cast<int>(std::lround((kp.y + 0.5f) / kp.scale - 0.5f));
                const float ca = std::cos(kp.angle), sa = std::sin(kp.angle);
                for (std::size_t c = 0; c < candidates.size(); ++c) {
                    values[c] = vg::evaluateBriefTest(blurred, x, y, ca, sa, candidates[c]);
                }
                bits.addSample(values);
            }
            ++framesUsed;
        }
        if (!reader.lastError().empty()) {
            std::fprintf(stderr, "error: %s\n", reader.lastError().c_str());
            return 1;
        }
    }

    const double n = static_cast<double>(bits.samples);
    std::printf("training set: %ld frames, %zu keypoints, %zu candidate tests\n",
                framesUsed, bits.samples, candidates.size());
    if (bits.samples < 1000) {
        std::fprintf(stderr, "error: too few keypoints to train on\n");
        return 1;
    }

    // Order candidates by how close their mean is to 0.5.
    std::vector<double> mean(candidates.size());
    for (std::size_t c = 0; c < candidates.size(); ++c) mean[c] = bits.ones(c) / n;
    std::vector<std::size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return std::abs(mean[a] - 0.5) < std::abs(mean[b] - 0.5);
    });

    const auto correlation = [&](std::size_t a, std::size_t b) {
        const double pa = mean[a], pb = mean[b];
        const double pab = bits.bothOnes(a, b) / n;
        const double denom = std::sqrt(pa * (1 - pa) * pb * (1 - pb));
        return denom > 0 ? (pab - pa * pb) / denom : 1.0;
    };

    // Greedy selection; relax the correlation threshold until 256 tests fit.
    std::vector<std::size_t> chosen;
    double threshold = 0.2;
    while (true) {
        chosen.clear();
        for (std::size_t c : order) {
            bool ok = true;
            for (std::size_t r : chosen) {
                if (std::abs(correlation(c, r)) > threshold) {
                    ok = false;
                    break;
                }
            }
            if (ok) chosen.push_back(c);
            if (chosen.size() == 256) break;
        }
        if (chosen.size() == 256) break;
        std::printf("threshold %.2f: only %zu tests, relaxing\n", threshold, chosen.size());
        threshold += 0.05;
    }

    double worstBias = 0, maxCorr = 0, sumAbsCorr = 0;
    for (std::size_t i = 0; i < chosen.size(); ++i) {
        worstBias = std::max(worstBias, std::abs(mean[chosen[i]] - 0.5));
        for (std::size_t j = 0; j < i; ++j) {
            const double r = std::abs(correlation(chosen[i], chosen[j]));
            maxCorr = std::max(maxCorr, r);
            sumAbsCorr += r;
        }
    }
    std::printf("selected 256 tests at correlation threshold %.2f: worst bias %.3f, max |corr| %.3f, mean |corr| %.3f\n",
                threshold, worstBias, maxCorr, sumAbsCorr / (256.0 * 255.0 / 2.0));

    std::ofstream out(output);
    if (!out) {
        std::fprintf(stderr, "error: cannot write %s\n", output.c_str());
        return 1;
    }
    out << "// Generated by tools/train_orb_pattern. Do not edit by hand.\n"
        << "// Training set: " << framesUsed << " frames, " << bits.samples << " keypoints, "
        << candidates.size() << " candidates, correlation threshold " << threshold << ".\n"
        << "// Each entry: {x1, y1, x2, y2}; bit = I(x1, y1) < I(x2, y2).\n"
        << "constexpr std::array<BriefTest, 256> kLearnedBriefPattern = {{\n";
    for (std::size_t i = 0; i < chosen.size(); ++i) {
        const vg::BriefTest& t = candidates[chosen[i]];
        char line[64];
        std::snprintf(line, sizeof(line), "    {%3d, %3d, %3d, %3d},%s", t.x1, t.y1, t.x2, t.y2,
                      (i % 4 == 3) ? "\n" : "");
        out << line;
    }
    out << "}};\n";
    std::printf("wrote %s\n", output.c_str());
    return 0;
}
