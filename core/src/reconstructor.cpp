#include "videogammetry/reconstructor.h"

namespace vg {

void Reconstructor::addFrame(const FrameView& frame) {
    if (frame.data == nullptr || frame.width <= 0 || frame.height <= 0) return;

    // TODO(phase 1): feature detection, matching, pose estimation, triangulation.
    ++frameCount_;
}

}  // namespace vg
