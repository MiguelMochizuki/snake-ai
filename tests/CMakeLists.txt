add_executable(snake_ai_tests
	test_tiles.cpp
	test_modes.cpp
	test_persistence.cpp
)
target_link_libraries(snake_ai_tests PRIVATE snake_ai_core doctest::doctest)
snake_ai_set_warnings(snake_ai_tests)

doctest_discover_tests(snake_ai_tests)

if(SNAKE_AI_BUILD_GUI)
	target_sources(snake_ai_tests PRIVATE test_gui_player.cpp)
	target_link_libraries(snake_ai_tests PRIVATE snake_gui_core)
	target_compile_definitions(snake_ai_tests PRIVATE SNAKE_AI_HAVE_GUI)
endif()
