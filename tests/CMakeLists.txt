add_executable(test_orb test_orb.cpp)
target_link_libraries(test_orb PRIVATE videogammetry::core)
add_test(NAME orb COMMAND test_orb)
