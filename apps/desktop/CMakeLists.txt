# Desktop CLI for Windows, Linux and macOS (same source on all three).

include(${PROJECT_SOURCE_DIR}/cmake/FFmpeg.cmake)

add_executable(videogammetry_cli
    src/main.cpp
    src/video_reader.cpp
)

target_link_libraries(videogammetry_cli PRIVATE videogammetry::core FFmpeg::FFmpeg)

set_target_properties(videogammetry_cli PROPERTIES OUTPUT_NAME videogammetry)

if(MSVC)
    target_compile_options(videogammetry_cli PRIVATE /W4 /permissive-)
else()
    target_compile_options(videogammetry_cli PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Put the FFmpeg DLLs next to the exe so it runs straight from the build folder.
if(WIN32 AND FFMPEG_DLLS)
    add_custom_command(TARGET videogammetry_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FFMPEG_DLLS} $<TARGET_FILE_DIR:videogammetry_cli>
        COMMAND_EXPAND_LISTS)
endif()
