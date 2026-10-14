cmake_minimum_required(VERSION 3.22)

project(videogammetry
    VERSION 0.1.0
    DESCRIPTION "Video to point cloud reconstruction"
    LANGUAGES CXX)

# Desktop app is built by default on Windows/Linux/macOS; Android builds only
# the core (pulled in from apps/android via its own CMakeLists).
option(VG_BUILD_DESKTOP "Build the desktop CLI app" ON)
option(VG_BUILD_TESTS "Build unit tests" ON)

add_subdirectory(core)

if(VG_BUILD_DESKTOP AND NOT ANDROID)
    include(cmake/FFmpeg.cmake)  # defines FFmpeg::FFmpeg and FFMPEG_DLLS for the apps and tools
    add_subdirectory(apps/desktop)
    add_subdirectory(tools)
endif()

if(VG_BUILD_TESTS AND NOT ANDROID)
    enable_testing()
    add_subdirectory(tests)
endif()
