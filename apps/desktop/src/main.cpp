#include <cstdio>
#include <cstring>
#include <string>

#include "videogammetry/reconstructor.h"
#include "videogammetry/version.h"

namespace {

void printUsage() {
    std::printf(
        "Usage: videogammetry <input-video> [-o <output.ply>]\n"
        "       videogammetry --version\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string input;
    std::string output = "cloud.ply";

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

    // TODO(phase 1): decode `input` into frames and push them to the reconstructor.
    vg::Reconstructor reconstructor;

    if (!vg::writePly(reconstructor.pointCloud(), output)) {
        std::fprintf(stderr, "error: could not write %s\n", output.c_str());
        return 1;
    }
    std::printf("video decoding not implemented yet; wrote empty cloud to %s\n", output.c_str());
    return 0;
}
