#pragma once

#include <cstddef>
#include <vector>

#include "videogammetry/frame.h"
#include "videogammetry/image.h"
#include "videogammetry/orb.h"
#include "videogammetry/point_cloud.h"
#include "videogammetry/profiler.h"

namespace vg {

// Entry point of the pipeline. Apps push frames in order, then read the cloud.
// Offline and real-time modes share this interface.
class Reconstructor {
public:
    struct Options {
        int processWidth = 960;  // frames wider than this are downscaled first; 0 = full size
        OrbParams orb;
    };

    Reconstructor() : Reconstructor(Options{}) {}
    explicit Reconstructor(const Options& options);

    void addFrame(const FrameView& frame);

    std::size_t frameCount() const { return frameCount_; }
    const PointCloud& pointCloud() const { return cloud_; }
    const Profiler& profiler() const { return profiler_; }

    // Features of the most recent frame, in processed-image pixels.
    const std::vector<Keypoint>& lastKeypoints() const { return keypoints_; }
    const std::vector<Descriptor>& lastDescriptors() const { return descriptors_; }
    // Input-frame pixels per processed-image pixel (>= 1).
    float processScale() const { return processScale_; }

private:
    Options options_;
    OrbExtractor orb_;
    Profiler profiler_;

    Image gray_;
    Image processed_;
    std::vector<Keypoint> keypoints_;
    std::vector<Descriptor> descriptors_;
    float processScale_ = 1.0f;

    std::size_t frameCount_ = 0;
    PointCloud cloud_;
};

}  // namespace vg
