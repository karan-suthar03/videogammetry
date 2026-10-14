#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <string>
#include <vector>

#include "frame_prefetcher.h"
#include "image_io.h"
#include "video_reader.h"
#include "videogammetry/reconstructor.h"
#include "videogammetry/version.h"

namespace {

void printUsage() {
    std::printf(
        "Usage: videogammetry <input-video> [options]\n"
        "       videogammetry --version\n"
        "\n"
        "Options:\n"
        "  -o <file.ply>               output point cloud (default: cloud.ply)\n"
        "  --width <px>                processing width, 0 = full resolution (default 960)\n"
        "  --threads <n>               worker threads, 0 = all hardware threads (default 0)\n"
        "  --match-gap <n>             match frame t against frame t-n (default 1)\n"
        "  --search-radius <px>        match search radius in processed pixels, 0 = everywhere (default 100)\n"
        "  --dump <dir>                write debug images to <dir>/features and <dir>/matches\n"
        "  --dump-every <n>            frame interval for --dump (default 25)\n"
        "  --save-frame <n> <file>     write decoded frame n (0-based) as .bmp or .ppm\n");
}

bool ensureDir(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "error: cannot create %s: %s\n", dir.string().c_str(), ec.message().c_str());
        return false;
    }
    return true;
}

std::string numbered(const std::filesystem::path& dir, const char* prefix, long index) {
    char name[64];
    std::snprintf(name, sizeof(name), "%s_%05ld.bmp", prefix, index);
    return (dir / name).string();
}

}  // namespace

int main(int argc, char** argv) {
    std::string input;
    std::string output = "cloud.ply";
    long saveFrameIndex = -1;
    std::string saveFramePath;
    vg::Reconstructor::Options options;
    std::string dumpDir;
    long dumpEvery = 25;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0) {
            std::printf("videogammetry %s\n", vg::version());
            return 0;
        }
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            printUsage();
            return 0;
        }
        if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            options.processWidth = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            options.threads = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--match-gap") == 0 && i + 1 < argc) {
            options.matchGap = std::max(1, static_cast<int>(std::strtol(argv[++i], nullptr, 10)));
        } else if (std::strcmp(argv[i], "--search-radius") == 0 && i + 1 < argc) {
            options.match.searchRadius = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dumpDir = argv[++i];
        } else if (std::strcmp(argv[i], "--dump-every") == 0 && i + 1 < argc) {
            dumpEvery = std::max(1L, std::strtol(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--save-frame") == 0 && i + 2 < argc) {
            saveFrameIndex = std::strtol(argv[++i], nullptr, 10);
            saveFramePath = argv[++i];
        } else if (input.empty()) {
            input = argv[i];
        } else {
            printUsage();
            return 1;
        }
    }

    if (input.empty()) {
        printUsage();
        return 1;
    }

    VideoReader reader;
    std::string error;
    if (!reader.open(input, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    const VideoReader::Info& info = reader.info();
    std::printf("input:   %s\n", input.c_str());
    std::printf("video:   %dx%d, %.2f fps, %.2f s, codec %s",
                info.width, info.height, info.fps, info.durationSec, info.codec.c_str());
    if (info.frameCountEstimate > 0) {
        std::printf(", ~%lld frames", static_cast<long long>(info.frameCountEstimate));
    }
    std::printf("\n");

    const bool downscaled = options.processWidth > 0 && info.width > options.processWidth;
    std::printf("process: %d px wide%s, matching frame t vs t-%d within %.0f px\n",
                downscaled ? options.processWidth : info.width,
                downscaled ? " (downscaled)" : " (full resolution)",
                options.matchGap, options.match.searchRadius);

    const std::filesystem::path featuresDir = std::filesystem::path(dumpDir) / "features";
    const std::filesystem::path matchesDir = std::filesystem::path(dumpDir) / "matches";
    if (!dumpDir.empty() && (!ensureDir(featuresDir) || !ensureDir(matchesDir))) return 1;

    vg::Reconstructor reconstructor(options);
    std::printf("threads: %d of %d hardware threads (+ background decode)\n",
                reconstructor.threadCount(), vg::ThreadPool::hardwareThreads());

    // Processed-size RGB copies of the last matchGap + 1 frames, for match images.
    std::deque<RgbImage> recentRgb;

    long dumped = 0;
    vg::FrameView frame;
    long index = 0;
    double lastTimestamp = 0.0;
    double decodeWaitMs = 0.0;
    long long totalFeatures = 0;
    long matchedFrames = 0;
    long long sumCandidates = 0, sumDistance = 0, sumCross = 0, sumFinal = 0;
    int minMatches = -1;

    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    FramePrefetcher frames(reader);
    while (true) {
        const auto waitStart = Clock::now();
        if (!frames.next(frame)) break;
        decodeWaitMs += std::chrono::duration<double, std::milli>(Clock::now() - waitStart).count();

        if (index == saveFrameIndex) {
            if (writeImage(frame, saveFramePath)) {
                std::printf("saved frame %ld (t=%.3f s) to %s\n", index, frame.timestamp, saveFramePath.c_str());
            } else {
                std::fprintf(stderr, "warning: could not write %s\n", saveFramePath.c_str());
            }
        }

        reconstructor.addFrame(frame);
        totalFeatures += static_cast<long long>(reconstructor.lastKeypoints().size());

        const auto& matches = reconstructor.lastMatches();
        const vg::MatchStats& ms = reconstructor.lastMatchStats();
        if (ms.candidates > 0) {
            ++matchedFrames;
            sumCandidates += ms.candidates;
            sumDistance += ms.afterDistance;
            sumCross += ms.afterCrossCheck;
            sumFinal += ms.afterOrientation;
            if (minMatches < 0 || ms.afterOrientation < minMatches) minMatches = ms.afterOrientation;
        }

        if (!dumpDir.empty()) {
            const float s = reconstructor.processScale();
            recentRgb.push_back(resizeRgb(frame, static_cast<int>(std::lround(frame.width / s)),
                                          static_cast<int>(std::lround(frame.height / s))));
            if (recentRgb.size() > static_cast<std::size_t>(options.matchGap) + 1) recentRgb.pop_front();

            if (index % dumpEvery == 0) {
                std::vector<std::uint8_t> drawn = drawKeypoints(frame, reconstructor.lastKeypoints(), s);
                vg::FrameView view = frame;
                view.data = drawn.data();
                view.stride = frame.width * 3;
                if (writeImage(view, numbered(featuresDir, "features", index))) ++dumped;

                if (!matches.empty() && recentRgb.size() == static_cast<std::size_t>(options.matchGap) + 1) {
                    const RgbImage img = drawMatches(recentRgb.front(), reconstructor.matchedFrameKeypoints(),
                                                     recentRgb.back(), reconstructor.lastKeypoints(), matches,
                                                     options.match.maxDistance);
                    if (writeImage(img.view(), numbered(matchesDir, "matches", index))) ++dumped;
                    const RgbImage flow = drawFlow(recentRgb.back(), reconstructor.matchedFrameKeypoints(),
                                                   reconstructor.lastKeypoints(), matches,
                                                   options.match.maxDistance);
                    if (writeImage(flow.view(), numbered(matchesDir, "flow", index))) ++dumped;
                }
            }
        }

        lastTimestamp = frame.timestamp;
        ++index;
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();

    if (!frames.error().empty()) {
        std::fprintf(stderr, "error: %s (after %ld frames)\n", frames.error().c_str(), index);
        return 1;
    }

    if (dumped) std::printf("dumped:  %ld images to %s\n", dumped, dumpDir.c_str());
    std::printf("frames:  %ld (last t=%.3f s), %.0f ORB features/frame avg\n",
                index, lastTimestamp, index ? double(totalFeatures) / index : 0.0);
    if (matchedFrames > 0) {
        const double n = static_cast<double>(matchedFrames);
        std::printf("matches: %.0f/frame avg (min %d) | funnel per frame: %.0f with candidates -> "
                    "%.0f distance+ratio -> %.0f cross-check -> %.0f orientation\n",
                    sumFinal / n, minMatches, sumCandidates / n, sumDistance / n, sumCross / n, sumFinal / n);
    }

    std::printf("\n  %-10s %10s %10s %10s\n", "stage", "total ms", "avg ms", "max ms");
    // Decoding runs on its own thread; this is only the time spent waiting for it.
    std::printf("  %-10s %10.1f %10.2f %10s\n", "decode*", decodeWaitMs, index ? decodeWaitMs / index : 0.0, "-");
    double coreMs = 0.0;
    for (const vg::Profiler::Stage& st : reconstructor.profiler().stages()) {
        std::printf("  %-10s %10.1f %10.2f %10.2f\n", st.name.c_str(), st.totalMs, st.avgMs(), st.maxMs);
        coreMs += st.totalMs;
    }
    std::printf("  %-10s %10.1f %10.2f\n", "all", seconds * 1000.0, index ? seconds * 1000.0 / index : 0.0);
    std::printf("\nthroughput: %.1f fps overall, %.1f fps core only (video is %.2f fps)\n",
                seconds > 0 ? index / seconds : 0.0,
                coreMs > 0 ? index / (coreMs / 1000.0) : 0.0, info.fps);

    if (!vg::writePly(reconstructor.pointCloud(), output)) {
        std::fprintf(stderr, "error: could not write %s\n", output.c_str());
        return 1;
    }
    std::printf("output:  %s (%zu points; reconstruction not implemented yet)\n",
                output.c_str(), reconstructor.pointCloud().points.size());
    return 0;
}
