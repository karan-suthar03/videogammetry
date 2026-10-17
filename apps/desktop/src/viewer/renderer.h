#pragma once

#include <cstdint>
#include <vector>

#include "image_io.h"
#include "live_session.h"
#include "videogammetry/linalg.h"

// 32-bit pixels, 0x00RRGGBB (BGRA byte order in memory), rows top-down.
struct Framebuffer {
    std::uint32_t* pixels = nullptr;
    int width = 0, height = 0;
    int stride = 0;  // in pixels
};

// Orbit camera of the 3D view. Distances are in units of the scene scale
// (the typical camera-to-point distance), because a monocular map has no
// absolute scale and a restarted map gets a different one.
//
// World axes follow the first camera: x right, y down, z forward, so -y is
// "up" for orbiting.
struct OrbitView {
    double yaw = 0.0;      // around the up axis; 0 = looking along +z
    double pitch = 0.45;   // > 0 looks down
    double zoom = 2.5;     // eye distance from the target, in scene scales
    double fovDeg = 60.0;  // horizontal field of view of the viewer
    vg::Vec3 target;       // orbit center (world)
    bool follow = true;    // target tracks the current camera
    bool attached = false; // look through the current camera (its pose and field of view)

    void orbit(double dxPx, double dyPx);
    // Moves the target in the view plane; stops following.
    void pan(double dxPx, double dyPx, double sceneScale, int viewportWidth);
    void zoomBy(double wheelSteps);
    // Eases the target toward the current camera when following.
    void update(const LiveSnapshot& scene);
    void reset();

    // World -> viewer-camera transform (same axes as the video camera).
    vg::Pose pose(double sceneScale) const;
};

struct RenderOptions {
    int pointSize = 2;            // pixels
    bool highlightTracked = true; // points seen in the current frame drawn green
    bool colorByNoise = true;     // points colored by depth uncertainty instead of their own color
    bool showKeyframes = true;
    bool showDense = true;        // background dense cloud
};

// Draws the map, keyframe and current-camera frustums and the camera path
// with a z-buffered point splatter. CPU only.
class SceneRenderer {
public:
    void render(const LiveSnapshot& scene, const OrbitView& view, const RenderOptions& options, Framebuffer& fb);

private:
    std::vector<float> depth_;
};

// Copies an RGB8 image into the framebuffer at (x, y), with a 1 px border.
void blitImage(const RgbImage& image, int x, int y, Framebuffer& fb);
