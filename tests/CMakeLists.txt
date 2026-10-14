add_executable(test_orb test_orb.cpp)
target_link_libraries(test_orb PRIVATE videogammetry::core)
add_test(NAME orb COMMAND test_orb)

add_executable(test_matcher test_matcher.cpp)
target_link_libraries(test_matcher PRIVATE videogammetry::core)
add_test(NAME matcher COMMAND test_matcher)
