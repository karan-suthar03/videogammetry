add_executable(test_orb test_orb.cpp)
target_link_libraries(test_orb PRIVATE videogammetry::core)
add_test(NAME orb COMMAND test_orb)

add_executable(test_matcher test_matcher.cpp)
target_link_libraries(test_matcher PRIVATE videogammetry::core)
add_test(NAME matcher COMMAND test_matcher)

add_executable(test_two_view test_two_view.cpp)
target_link_libraries(test_two_view PRIVATE videogammetry::core)
add_test(NAME two_view COMMAND test_two_view)

add_executable(test_pnp test_pnp.cpp)
target_link_libraries(test_pnp PRIVATE videogammetry::core)
add_test(NAME pnp COMMAND test_pnp)

add_executable(test_bundle_adjustment test_bundle_adjustment.cpp)
target_link_libraries(test_bundle_adjustment PRIVATE videogammetry::core)
add_test(NAME bundle_adjustment COMMAND test_bundle_adjustment)

add_executable(test_mvs test_mvs.cpp)
target_link_libraries(test_mvs PRIVATE videogammetry::core)
add_test(NAME mvs COMMAND test_mvs)
if(TARGET videogammetry::gpu)
    target_link_libraries(test_mvs PRIVATE videogammetry::gpu)
    target_compile_definitions(test_mvs PRIVATE VG_HAS_GPU)
endif()
