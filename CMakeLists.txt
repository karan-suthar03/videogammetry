cmake_minimum_required(VERSION 3.22)

project(videogammetry
    VERSION 0.1.0
    DESCRIPTION "Video to point cloud reconstruction"
    LANGUAGES CXX)

# Desktop app is built by default on Windows/Linux/macOS; Android builds only
# the core (pulled in from apps/android via its own CMakeLists).
option(VG_BUILD_DESKTOP "Build the desktop CLI app" ON)
option(VG_BUILD_TESTS "Build unit tests" ON)
# Self-contained Windows exes: the MSVC runtime and FFmpeg linked in statically,
# so the exe runs on its own with no DLLs next to it (see cmake/FFmpeg.cmake).
option(VG_STATIC "Link the C++ runtime and FFmpeg statically (Windows)" OFF)

if(VG_STATIC AND MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

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
