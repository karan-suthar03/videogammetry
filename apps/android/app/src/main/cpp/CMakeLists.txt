cmake_minimum_required(VERSION 3.22)

project(videogammetry_android VERSION 0.1.0 LANGUAGES CXX)

# The reconstruction is far too slow unoptimized; optimize native code even
# in debug builds of the app.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O2)
endif()

# Pull in the shared core from the repo root.
set(VG_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../..)
add_subdirectory(${VG_ROOT}/core ${CMAKE_CURRENT_BINARY_DIR}/core)

add_library(videogammetry_jni SHARED native_bridge.cpp)

target_link_libraries(videogammetry_jni PRIVATE videogammetry::core jnigraphics log)
