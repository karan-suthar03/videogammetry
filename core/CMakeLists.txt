# videogammetry core: platform-independent reconstruction library.
# No OS, UI, or video-decoding code lives here. Apps feed it frames.
# Included both from the repo root (desktop) and from apps/android.

set(VG_CORE_VERSION 0.1.0)

add_library(videogammetry_core STATIC
    src/version.cpp
    src/bundle_adjustment.cpp
    src/features.cpp
    src/image.cpp
    src/linalg.cpp
    src/map_quality.cpp
    src/matcher.cpp
    src/mvs.cpp
    src/orb.cpp
    src/pnp.cpp
    src/point_cloud.cpp
    src/profiler.cpp
    src/reconstructor.cpp
    src/thread_pool.cpp
    src/two_view.cpp
)
add_library(videogammetry::core ALIAS videogammetry_core)

target_include_directories(videogammetry_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(videogammetry_core PUBLIC cxx_std_20)

target_compile_definitions(videogammetry_core PRIVATE
    VG_VERSION_STRING="${VG_CORE_VERSION}")

if(MSVC)
    target_compile_options(videogammetry_core PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(videogammetry_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(videogammetry_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
target_link_libraries(videogammetry_core PUBLIC Threads::Threads)
