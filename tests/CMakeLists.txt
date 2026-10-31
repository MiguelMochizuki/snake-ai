add_executable(snake_ai_tests
	test_tiles.cpp
	test_modes.cpp
	test_persistence.cpp
)
target_link_libraries(snake_ai_tests PRIVATE snake_ai_core doctest::doctest)
snake_ai_set_warnings(snake_ai_tests)

doctest_discover_tests(snake_ai_tests)
