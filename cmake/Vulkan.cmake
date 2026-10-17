# Vulkan for the GPU backend (gpu/), without needing the Vulkan SDK installed:
#   VulkanHeaders::Headers  the API headers (header-only)
#   volk::volk              loads vulkan-1.dll / libvulkan at run time, so the apps
#                           start (and fall back to the CPU) on machines without Vulkan
#   VG_GLSLANG              host shader compiler (GLSL -> SPIR-V), for vg_add_shaders()
# All pinned to one Vulkan SDK release and checked against SHA-256.

include(FetchContent)
enable_language(C)  # volk is C

set(_vk_tag "vulkan-sdk-1.4.363.0")

# SOURCE_SUBDIR points at a directory without a CMakeLists.txt: fetch only,
# the targets below are our own.
FetchContent_Declare(vg_vulkan_headers
    URL "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/${_vk_tag}.tar.gz"
    URL_HASH SHA256=4a078be12bef21cfebc09d878b77a63cff9d68f899254a0b00d0e37ef73e7f7e
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR fetch-only)
FetchContent_Declare(vg_volk
    URL "https://github.com/zeux/volk/archive/refs/tags/${_vk_tag}.tar.gz"
    URL_HASH SHA256=1547d8d74395d4048fb3f4a6313da56db626e89b55db238d0d7e8944c8a645f3
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR fetch-only)

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(_glslang_asset "windows-x86_64-release.zip")
    set(_glslang_hash 82bf434e69b9bb4829de7e2b4bc2c5e7a7861e53d66cf75e5cc70f5f694a8d9b)
    set(_glslang_exe "glslang.exe")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(_glslang_asset "macos-universal-release.zip")
    set(_glslang_hash 1086bc6c7ce8ff188e4e1d1a69078e9e38156a4c2c3f13e4f91b9289a1d90453)
    set(_glslang_exe "glslang")
else()
    set(_glslang_asset "linux-x86_64-release.zip")
    set(_glslang_hash a3fc4f083b1793eb53e55fa3577ac9649ffbe0340715e30d116290fb5382393f)
    set(_glslang_exe "glslang")
endif()
FetchContent_Declare(vg_glslang
    URL "https://github.com/KhronosGroup/glslang/releases/download/16.6.0/glslang-16.6.0-${_glslang_asset}"
    URL_HASH SHA256=${_glslang_hash}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR fetch-only)

FetchContent_MakeAvailable(vg_vulkan_headers vg_volk vg_glslang)

if(NOT TARGET VulkanHeaders::Headers)
    add_library(vg_vulkan_headers INTERFACE)
    target_include_directories(vg_vulkan_headers SYSTEM INTERFACE "${vg_vulkan_headers_SOURCE_DIR}/include")
    add_library(VulkanHeaders::Headers ALIAS vg_vulkan_headers)
endif()

if(NOT TARGET volk::volk)
    add_library(vg_volk STATIC "${vg_volk_SOURCE_DIR}/volk.c")
    set_target_properties(vg_volk PROPERTIES LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON)
    target_include_directories(vg_volk SYSTEM PUBLIC "${vg_volk_SOURCE_DIR}")
    target_compile_definitions(vg_volk PUBLIC VK_NO_PROTOTYPES)  # compute only: no window-system headers
    target_link_libraries(vg_volk PUBLIC VulkanHeaders::Headers ${CMAKE_DL_LIBS})
    add_library(volk::volk ALIAS vg_volk)
endif()

set(VG_GLSLANG "${vg_glslang_SOURCE_DIR}/bin/${_glslang_exe}" CACHE FILEPATH "GLSL to SPIR-V compiler" FORCE)
if(NOT CMAKE_HOST_WIN32)
    file(CHMOD "${VG_GLSLANG}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
         WORLD_READ WORLD_EXECUTE)
endif()

# vg_add_shaders(<target> <shader.comp>...): compiles each compute shader to
# SPIR-V embedded as a C array (k<Name>Spv, e.g. orb_fast.comp -> kOrbFastSpv)
# in <binary dir>/shaders/<name>.h, on the target's include path.
function(vg_add_shaders target)
    set(_out_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${_out_dir}")
    set(_headers "")
    foreach(_src IN LISTS ARGN)
        get_filename_component(_abs "${_src}" ABSOLUTE)
        get_filename_component(_name "${_src}" NAME_WE)
        string(REPLACE "_" ";" _parts "${_name}")
        set(_var "k")
        foreach(_p IN LISTS _parts)
            string(SUBSTRING "${_p}" 0 1 _first)
            string(SUBSTRING "${_p}" 1 -1 _rest)
            string(TOUPPER "${_first}" _first)
            string(APPEND _var "${_first}${_rest}")
        endforeach()
        string(APPEND _var "Spv")
        set(_header "${_out_dir}/${_name}.h")
        add_custom_command(
            OUTPUT "${_header}"
            COMMAND "${VG_GLSLANG}" -V --target-env vulkan1.1 --quiet --vn "${_var}"
                    -I${CMAKE_CURRENT_SOURCE_DIR}/shaders -o "${_header}" "${_abs}"
            DEPENDS "${_abs}" ${VG_SHADER_INCLUDES}
            COMMENT "SPIR-V ${_name}"
            VERBATIM)
        list(APPEND _headers "${_header}")
    endforeach()
    target_sources(${target} PRIVATE ${_headers})
    target_include_directories(${target} PRIVATE "${_out_dir}")
endfunction()
