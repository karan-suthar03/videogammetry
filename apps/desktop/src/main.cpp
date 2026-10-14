#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

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
        "  -o <file.ply>              output point cloud (default: cloud.ply)\n"
        "  --save-frame <n> <file>    also write decoded frame n (0-based) as a .ppm image\n");
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

int main(int argc, char** argv) {
    std::string input;
    std::string output = "cloud.ply";
    long saveFrameIndex = -1;
    std::string saveFramePath;

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
    std::printf("input:  %s\n", input.c_str());
    std::printf("video:  %dx%d, %.2f fps, %.2f s, codec %s",
                info.width, info.height, info.fps, info.durationSec, info.codec.c_str());
    if (info.frameCountEstimate > 0) {
        std::printf(", ~%lld frames", static_cast<long long>(info.frameCountEstimate));
    }
    std::printf("\n");

    vg::Reconstructor reconstructor;
    vg::FrameView frame;
    long index = 0;
    double lastTimestamp = 0.0;

    const auto start = std::chrono::steady_clock::now();
    while (reader.read(frame)) {
        if (index == saveFrameIndex) {
            if (writePpm(frame, saveFramePath)) {
                std::printf("saved frame %ld (t=%.3f s) to %s\n", index, frame.timestamp, saveFramePath.c_str());
            } else {
                std::fprintf(stderr, "warning: could not write %s\n", saveFramePath.c_str());
            }
        }
        reconstructor.addFrame(frame);
        lastTimestamp = frame.timestamp;
        ++index;
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    if (!reader.lastError().empty()) {
        std::fprintf(stderr, "error: %s (after %ld frames)\n", reader.lastError().c_str(), index);
        return 1;
    }

    std::printf("decoded: %ld frames (last t=%.3f s) in %.2f s = %.1f fps\n",
                index, lastTimestamp, seconds, seconds > 0 ? index / seconds : 0.0);
    std::printf("core:   %zu frames received\n", reconstructor.frameCount());

    if (!vg::writePly(reconstructor.pointCloud(), output)) {
        std::fprintf(stderr, "error: could not write %s\n", output.c_str());
        return 1;
    }
    std::printf("output: %s (%zu points; reconstruction not implemented yet)\n",
                output.c_str(), reconstructor.pointCloud().points.size());
    return 0;
}
