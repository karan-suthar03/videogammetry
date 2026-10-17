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

# CPU vs GPU feature extraction: identity and timing.
if(TARGET videogammetry_gpu)
    add_executable(gpu_check gpu_check/main.cpp)
    target_link_libraries(gpu_check PRIVATE videogammetry::core videogammetry::gpu videogammetry_video)
    if(MSVC)
        target_compile_options(gpu_check PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(gpu_check PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    if(WIN32 AND FFMPEG_DLLS)
        add_custom_command(TARGET gpu_check POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FFMPEG_DLLS} $<TARGET_FILE_DIR:gpu_check>
            COMMAND_EXPAND_LISTS)
    endif()
endif()
