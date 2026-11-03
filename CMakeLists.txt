cmake_minimum_required(VERSION 3.16)
project(snake_ai LANGUAGES C CXX) # C is needed by SDL3 (GUI build)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(SNAKE_AI_BUILD_APPS "Build the command-line apps" ON)
option(SNAKE_AI_BUILD_EXPERIMENTS "Build the research experiments" ON)
option(SNAKE_AI_BUILD_TESTS "Build unit tests" ON)
option(SNAKE_AI_BUILD_GUI "Let watch draw through snake-game's SDL3 renderer (fetches SDL3 and miniaudio when configuring)" OFF)

# Dependencies are git submodules under deps/ (git submodule update --init)
set(SNAKE_DIR ${CMAKE_SOURCE_DIR}/deps/snake-game)
set(NN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NN_BUILD_APPS OFF CACHE BOOL "" FORCE)
add_subdirectory(deps/nn-cpp)

find_package(Threads REQUIRED)

# Compiler warnings, shared by all targets
function(snake_ai_set_warnings target)
	if(MSVC)
		target_compile_options(${target} PRIVATE /W4)
	else()
		target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
	endif()
endfunction()

if(SNAKE_AI_BUILD_GUI)
	# The whole of snake-game, so snake_gui_core can be linked; its own apps and tests stay off
	set(SNAKE_GAME_BUILD_APPS OFF CACHE BOOL "" FORCE)
	set(SNAKE_GAME_BUILD_TESTS OFF CACHE BOOL "" FORCE)
	set(SNAKE_GAME_BUILD_GUI ON CACHE BOOL "" FORCE)
	add_subdirectory(deps/snake-game)
	add_library(snake_logic INTERFACE)
	target_link_libraries(snake_logic INTERFACE snake_core)
else()
	# Only the headless game logic of snake-game; no terminal code
	add_library(snake_logic STATIC
		${SNAKE_DIR}/src/game/entity.cpp
		${SNAKE_DIR}/src/game/food.cpp
		${SNAKE_DIR}/src/game/snake_segment.cpp
		${SNAKE_DIR}/src/game/snake.cpp
		${SNAKE_DIR}/src/game/game_state.cpp
	)
	target_include_directories(snake_logic PUBLIC ${SNAKE_DIR}/include)
endif()

add_library(snake_ai_core STATIC
	src/env/cycle.cpp
	src/env/snake_env.cpp
	src/planning/tiles.cpp
	src/planning/modes.cpp
	src/teachers/expert.cpp
	src/teachers/heuristic.cpp
)
target_include_directories(snake_ai_core PUBLIC include)
target_link_libraries(snake_ai_core PUBLIC snake_logic nn_core Threads::Threads)
snake_ai_set_warnings(snake_ai_core)

if(SNAKE_AI_BUILD_APPS)
	add_subdirectory(apps)
endif()

if(SNAKE_AI_BUILD_EXPERIMENTS)
	add_subdirectory(experiments)
endif()

if(SNAKE_AI_BUILD_TESTS)
	enable_testing()
	include(${CMAKE_SOURCE_DIR}/cmake/FetchDoctest.cmake)
	include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
	add_subdirectory(tests)
endif()
