# Offline developer tools (desktop only).

add_executable(train_orb_pattern train_orb_pattern/main.cpp)
target_link_libraries(train_orb_pattern PRIVATE videogammetry::core videogammetry_video)

if(MSVC)
    target_compile_options(train_orb_pattern PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(train_orb_pattern PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32 AND FFMPEG_DLLS)
    add_custom_command(TARGET train_orb_pattern POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FFMPEG_DLLS} $<TARGET_FILE_DIR:train_orb_pattern>
        COMMAND_EXPAND_LISTS)
endif()
