# Provides the FFmpeg::FFmpeg target (avformat, avcodec, avutil, swscale).
#
# Windows: uses the prebuilt package in FFMPEG_ROOT
#          (default third_party/ffmpeg, fetched by scripts/fetch_ffmpeg_windows.ps1).
#          With VG_STATIC, static libraries instead (default
#          third_party/ffmpeg-static, built by scripts/build_ffmpeg_static_windows.ps1).
# Other:   uses pkg-config (e.g. apt install libavformat-dev libavcodec-dev libswscale-dev).

if(TARGET FFmpeg::FFmpeg)
    return()
endif()

set(_ffmpeg_components avformat avcodec swscale avutil)  # dependents first, for static linking

if(WIN32)
    if(VG_STATIC)
        set(_ffmpeg_default "${PROJECT_SOURCE_DIR}/third_party/ffmpeg-static")
        set(_ffmpeg_hint "scripts/build_ffmpeg_static_windows.ps1")
    else()
        set(_ffmpeg_default "${PROJECT_SOURCE_DIR}/third_party/ffmpeg")
        set(_ffmpeg_hint "scripts/fetch_ffmpeg_windows.ps1")
    endif()
    set(FFMPEG_ROOT "${_ffmpeg_default}" CACHE PATH "FFmpeg dev package root")

    if(NOT EXISTS "${FFMPEG_ROOT}/include/libavcodec/avcodec.h")
        message(FATAL_ERROR
            "FFmpeg not found in ${FFMPEG_ROOT}.\n"
            "Run: powershell -ExecutionPolicy Bypass -File ${_ffmpeg_hint}\n"
            "or pass -DFFMPEG_ROOT=<path to FFmpeg dev package>.")
    endif()

    add_library(FFmpeg::FFmpeg INTERFACE IMPORTED)
    target_include_directories(FFmpeg::FFmpeg INTERFACE "${FFMPEG_ROOT}/include")
    foreach(_c IN LISTS _ffmpeg_components)
        target_link_libraries(FFmpeg::FFmpeg INTERFACE "${FFMPEG_ROOT}/lib/${_c}.lib")
    endforeach()

    if(VG_STATIC)
        # What the static libraries need from Windows itself (Libs.private in
        # their .pc files): all system DLLs, present on every machine.
        target_link_libraries(FFmpeg::FFmpeg INTERFACE
            mfuuid strmiids ole32 user32 secur32 ncrypt crypt32 ws2_32 bcrypt)
        set(FFMPEG_DLLS "")
    else()
        file(GLOB FFMPEG_DLLS "${FFMPEG_ROOT}/bin/*.dll")
    endif()
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswscale)
    add_library(FFmpeg::FFmpeg INTERFACE IMPORTED)
    target_link_libraries(FFmpeg::FFmpeg INTERFACE PkgConfig::FFMPEG)
endif()
