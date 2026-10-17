// Lists the cameras and reads frames from one: size, frame rate, and one
// frame saved as an image to check colors and orientation.
//
// Usage: camera_check [camera index] [--size N] [--frames N] [--save frame.png]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "camera_reader.h"
#include "image_io.h"

int main(int argc, char** argv) {
    int index = 0, size = 960, frames = 60;
    std::string save;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            size = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            frames = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save = argv[++i];
        } else {
            index = std::atoi(argv[i]);
        }
    }

    const std::vector<CameraReader::Device> cameras = CameraReader::list();
    for (std::size_t i = 0; i < cameras.size(); ++i) std::printf("%zu: %s\n", i, cameras[i].name.c_str());
    if (index < 0 || index >= static_cast<int>(cameras.size())) {
        std::fprintf(stderr, "no camera %d\n", index);
        return 1;
    }

    CameraReader camera;
    std::string error;
    if (!camera.open(cameras[index].id, size, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    vg::FrameView frame;
    double first = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < frames; ++i) {
        if (!camera.read(frame)) {
            std::fprintf(stderr, "read failed: %s\n", camera.lastError().c_str());
            return 1;
        }
        if (i == 0) first = frame.timestamp;
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("%s: %dx%d, %d frames, %.1f fps (camera clock %.1f fps)\n", cameras[index].name.c_str(), frame.width,
                frame.height, frames, frames / wall, (frames - 1) / (frame.timestamp - first));
    if (!save.empty()) {
        if (!writeImage(frame, save)) {
            std::fprintf(stderr, "cannot write %s\n", save.c_str());
            return 1;
        }
        std::printf("saved %s\n", save.c_str());
    }
    return 0;
}
