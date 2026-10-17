#pragma once

#include <memory>
#include <string>

#include "videogammetry/features.h"
#include "videogammetry/gpu/vulkan_context.h"

namespace vg::gpu {

// vg::FeatureExtractor on a Vulkan compute device: gray conversion,
// downscaling, the pyramid, FAST, Harris, non-maximum suppression,
// orientation and descriptors run on the GPU in one submission; the host then
// spreads the corners over the grid with the same code as the CPU. Keypoints
// (positions, levels, responses, angles) are identical to the CPU's; a
// descriptor bit can rarely differ, because GPU sine/cosine are not exactly
// rounded. Not thread-safe; one per thread (they share the context).
class GpuFeatureExtractor final : public FrameExtractor {
public:
    GpuFeatureExtractor(std::shared_ptr<VulkanContext> context, int processSize, const OrbParams& orb);
    ~GpuFeatureExtractor() override;

    void extract(const FrameView& frame, long index, FrameFeatures& out) override;

    // Configurations the GPU path does not handle (e.g. thousands of pyramid
    // levels) are extracted on the CPU instead; true if that happened.
    bool usedCpuFallback() const { return usedCpu_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool usedCpu_ = false;
};

// Creates extractors for FeaturePipeline workers on one shared Vulkan device,
// or returns an empty factory (CPU extraction) when there is no usable device;
// `status` says which and why.
FrameExtractorFactory makeGpuExtractorFactory(int processSize, const OrbParams& orb, std::string& status);

}  // namespace vg::gpu
