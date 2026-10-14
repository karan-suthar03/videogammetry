# Desktop CLI for Windows, Linux and macOS (same source on all three).

# Video decoding + image file helpers, shared by the CLI and the dev tools.
add_library(videogammetry_video STATIC
    src/video_reader.cpp
    src/image_io.cpp
    src/frame_prefetcher.cpp
)
target_include_directories(videogammetry_video PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(videogammetry_video PUBLIC videogammetry::core FFmpeg::FFmpeg)

add_executable(videogammetry_cli src/main.cpp)
target_link_libraries(videogammetry_cli PRIVATE videogammetry::core videogammetry_video)
set_target_properties(videogammetry_cli PROPERTIES OUTPUT_NAME videogammetry)

foreach(_t videogammetry_video videogammetry_cli)
    if(MSVC)
        target_compile_options(${_t} PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(${_t} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()

# Put the FFmpeg DLLs next to the exe so it runs straight from the build folder.
if(WIN32 AND FFMPEG_DLLS)
    add_custom_command(TARGET videogammetry_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FFMPEG_DLLS} $<TARGET_FILE_DIR:videogammetry_cli>
        COMMAND_EXPAND_LISTS)
endif()
