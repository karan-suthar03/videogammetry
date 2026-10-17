# Desktop apps: the CLI (Windows, Linux and macOS, same source) and the live viewer.

# Video decoding, image file helpers and the options table, shared by the CLI,
# the viewer and the dev tools.
add_library(videogammetry_video STATIC
    src/video_reader.cpp
    src/image_io.cpp
    src/frame_prefetcher.cpp
    src/power.cpp
    src/options_table.cpp
)
target_include_directories(videogammetry_video PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(videogammetry_video PUBLIC videogammetry::core FFmpeg::FFmpeg)

add_executable(videogammetry_cli src/main.cpp)
target_link_libraries(videogammetry_cli PRIVATE videogammetry::core videogammetry_video)
if(TARGET videogammetry_gpu)
    target_link_libraries(videogammetry_cli PRIVATE videogammetry::gpu)
    target_compile_definitions(videogammetry_cli PRIVATE VG_HAS_GPU)
endif()
set_target_properties(videogammetry_cli PROPERTIES OUTPUT_NAME videogammetry)

set(_targets videogammetry_video videogammetry_cli)

# Live viewer: the point cloud and cameras as they are reconstructed.
# The window layer is Win32 for now; live_session and renderer are portable.
if(WIN32)
    add_executable(videogammetry_viewer WIN32
        src/viewer/main_win32.cpp
        src/viewer/live_session.cpp
        src/viewer/renderer.cpp
    )
    target_include_directories(videogammetry_viewer PRIVATE src/viewer)
    target_sources(videogammetry_viewer PRIVATE src/viewer/settings_table.cpp)
    target_link_libraries(videogammetry_viewer PRIVATE videogammetry::core videogammetry_video comdlg32 winmm)
    if(TARGET videogammetry_gpu)
        target_link_libraries(videogammetry_viewer PRIVATE videogammetry::gpu)
        target_compile_definitions(videogammetry_viewer PRIVATE VG_HAS_GPU)
    endif()
    target_compile_definitions(videogammetry_viewer PRIVATE UNICODE _UNICODE)
    list(APPEND _targets videogammetry_viewer)
endif()

foreach(_t ${_targets})
    if(MSVC)
        target_compile_options(${_t} PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(${_t} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()

# Put the FFmpeg DLLs next to the exe so it runs straight from the build folder.
if(WIN32 AND FFMPEG_DLLS)
    foreach(_t videogammetry_cli videogammetry_viewer)
        add_custom_command(TARGET ${_t} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FFMPEG_DLLS} $<TARGET_FILE_DIR:${_t}>
            COMMAND_EXPAND_LISTS)
    endforeach()
endif()
