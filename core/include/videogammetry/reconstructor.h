#pragma once

#include <cstddef>

#include "videogammetry/frame.h"
#include "videogammetry/point_cloud.h"

namespace vg {

// Entry point of the pipeline. Apps push frames in order, then read the cloud.
// Offline and real-time modes share this interface.
class Reconstructor {
public:
    void addFrame(const FrameView& frame);

    std::size_t frameCount() const { return frameCount_; }
    const PointCloud& pointCloud() const { return cloud_; }

private:
    std::size_t frameCount_ = 0;
    PointCloud cloud_;
};

}  // namespace vg
