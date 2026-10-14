#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

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
        "  --save-frame <n> <file>     also write decoded frame n (0-based) as a .ppm image\n"
        "  --save-features <n> <file>  write frame n with its ORB keypoints drawn, as .ppm\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string input;
    std::string output = "cloud.ply";
    long saveFrameIndex = -1;
    std::string saveFramePath;
    long saveFeaturesIndex = -1;
    std::string saveFeaturesPath;
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
        } else if (std::strcmp(argv[i], "--save-frame") == 0 && i + 2 < argc) {
            saveFrameIndex = std::strtol(argv[++i], nullptr, 10);
            saveFramePath = argv[++i];
        } else if (std::strcmp(argv[i], "--save-features") == 0 && i + 2 < argc) {
            saveFeaturesIndex = std::strtol(argv[++i], nullptr, 10);
            saveFeaturesPath = argv[++i];
        } else if (std::strcmp(argv[i], "--dump-features") == 0 && i + 1 < argc) {
            dumpDir = argv[++i];
        } else if (std::strcmp(argv[i], "--dump-every") == 0 && i + 1 < argc) {
            dumpEvery = std::max(1L, std::strtol(argv[++i], nullptr, 10));
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
    std::printf("process: %d px wide%s\n", downscaled ? options.processWidth : info.width,
                downscaled ? " (downscaled)" : " (full resolution)");

    if (!dumpDir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(dumpDir, ec);
        if (ec) {
            std::fprintf(stderr, "error: cannot create %s: %s\n", dumpDir.c_str(), ec.message().c_str());
            return 1;
        }
    }

    vg::Reconstructor reconstructor(options);
    long dumped = 0;
    vg::FrameView frame;
    long index = 0;
    double lastTimestamp = 0.0;
    double decodeMs = 0.0;
    long long totalFeatures = 0;

    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    while (true) {
        const auto decodeStart = Clock::now();
        if (!reader.read(frame)) break;
        decodeMs += std::chrono::duration<double, std::milli>(Clock::now() - decodeStart).count();

        if (index == saveFrameIndex) {
            if (writeImage(frame, saveFramePath)) {
                std::printf("saved frame %ld (t=%.3f s) to %s\n", index, frame.timestamp, saveFramePath.c_str());
            } else {
                std::fprintf(stderr, "warning: could not write %s\n", saveFramePath.c_str());
            }
        }

        reconstructor.addFrame(frame);
        totalFeatures += static_cast<long long>(reconstructor.lastKeypoints().size());

        const bool dumpThis = !dumpDir.empty() && index % dumpEvery == 0;
        if (index == saveFeaturesIndex || dumpThis) {
            std::vector<std::uint8_t> drawn =
                drawKeypoints(frame, reconstructor.lastKeypoints(), reconstructor.processScale());
            vg::FrameView view = frame;
            view.data = drawn.data();
            view.stride = frame.width * 3;
            if (index == saveFeaturesIndex) {
                if (writeImage(view, saveFeaturesPath)) {
                    std::printf("saved %zu keypoints on frame %ld to %s\n",
                                reconstructor.lastKeypoints().size(), index, saveFeaturesPath.c_str());
                } else {
                    std::fprintf(stderr, "warning: could not write %s\n", saveFeaturesPath.c_str());
                }
            }
            if (dumpThis) {
                char name[64];
                std::snprintf(name, sizeof(name), "features_%05ld.bmp", index);
                const std::string path = (std::filesystem::path(dumpDir) / name).string();
                if (writeImage(view, path)) {
                    ++dumped;
                } else {
                    std::fprintf(stderr, "warning: could not write %s\n", path.c_str());
                }
            }
        }

        lastTimestamp = frame.timestamp;
        ++index;
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();

    if (!reader.lastError().empty()) {
        std::fprintf(stderr, "error: %s (after %ld frames)\n", reader.lastError().c_str(), index);
        return 1;
    }

    if (dumped) std::printf("dumped:  %ld keypoint images to %s\n", dumped, dumpDir.c_str());
    std::printf("frames:  %ld (last t=%.3f s), %.0f ORB features/frame avg\n",
                index, lastTimestamp, index ? double(totalFeatures) / index : 0.0);

    std::printf("\n  %-10s %10s %10s %10s\n", "stage", "total ms", "avg ms", "max ms");
    std::printf("  %-10s %10.1f %10.2f %10s\n", "decode", decodeMs, index ? decodeMs / index : 0.0, "-");
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
