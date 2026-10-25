#pragma once

#include <string>

#include "videogammetry/mvs.h"

namespace vg::gpu {

// vg::patchMatch on a Vulkan compute device (shaders/patch_match.comp): the
// same steps and constants, so depth maps match the CPU's closely (random
// numbers and float rounding differ, so not bit for bit). Returns an empty
// solver (the CPU is used) when there is no usable device; `status` names the
// device or why there is none. The solver is not thread-safe.
DepthMapSolver makeGpuDepthSolver(std::string& status);

}  // namespace vg::gpu
