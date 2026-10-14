# Provides the FFmpeg::FFmpeg target (avformat, avcodec, avutil, swscale).
#
# Windows: uses the prebuilt package in FFMPEG_ROOT
#          (default third_party/ffmpeg, fetched by scripts/fetch_ffmpeg_windows.ps1).
# Other:   uses pkg-config (e.g. apt install libavformat-dev libavcodec-dev libswscale-dev).

if(TARGET FFmpeg::FFmpeg)
    return()
endif()

set(_ffmpeg_components avformat avcodec avutil swscale)

if(WIN32)
    set(FFMPEG_ROOT "${PROJECT_SOURCE_DIR}/third_party/ffmpeg" CACHE PATH "FFmpeg dev package root")

    if(NOT EXISTS "${FFMPEG_ROOT}/include/libavcodec/avcodec.h")
        message(FATAL_ERROR
            "FFmpeg not found in ${FFMPEG_ROOT}.\n"
            "Run: powershell -ExecutionPolicy Bypass -File scripts/fetch_ffmpeg_windows.ps1\n"
            "or pass -DFFMPEG_ROOT=<path to FFmpeg shared dev package>.")
    endif()

    add_library(FFmpeg::FFmpeg INTERFACE IMPORTED)
    target_include_directories(FFmpeg::FFmpeg INTERFACE "${FFMPEG_ROOT}/include")
    foreach(_c IN LISTS _ffmpeg_components)
        target_link_libraries(FFmpeg::FFmpeg INTERFACE "${FFMPEG_ROOT}/lib/${_c}.lib")
    endforeach()

    file(GLOB FFMPEG_DLLS "${FFMPEG_ROOT}/bin/*.dll")
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswscale)
    add_library(FFmpeg::FFmpeg INTERFACE IMPORTED)
    target_link_libraries(FFmpeg::FFmpeg INTERFACE PkgConfig::FFMPEG)
endif()
