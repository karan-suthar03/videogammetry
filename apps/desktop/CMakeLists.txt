# Desktop CLI for Windows, Linux and macOS (same source on all three).

add_executable(videogammetry_cli src/main.cpp)

target_link_libraries(videogammetry_cli PRIVATE videogammetry::core)

set_target_properties(videogammetry_cli PROPERTIES OUTPUT_NAME videogammetry)

if(MSVC)
    target_compile_options(videogammetry_cli PRIVATE /W4 /permissive-)
else()
    target_compile_options(videogammetry_cli PRIVATE -Wall -Wextra -Wpedantic)
endif()
