# videogammetry GPU backend: Vulkan compute implementations of core stages.
# The core stays GPU-free; apps plug these in (e.g. FeaturePipeline's
# extractor factory) and fall back to the CPU when there is no device.

include(${PROJECT_SOURCE_DIR}/cmake/Vulkan.cmake)

add_library(videogammetry_gpu STATIC
    src/vulkan_context.cpp
    src/gpu_orb.cpp
)
add_library(videogammetry::gpu ALIAS videogammetry_gpu)

target_include_directories(videogammetry_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(videogammetry_gpu PUBLIC videogammetry::core volk::volk)

set(VG_SHADER_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/common.glsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/fast.glsl)
vg_add_shaders(videogammetry_gpu
    shaders/gray.comp
    shaders/resize.comp
    shaders/blur_h.comp
    shaders/blur_v.comp
    shaders/fast_cells.comp
    shaders/fast_collect.comp
    shaders/describe.comp
)

if(MSVC)
    target_compile_options(videogammetry_gpu PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(videogammetry_gpu PRIVATE -Wall -Wextra -Wpedantic)
endif()
set_target_properties(videogammetry_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
