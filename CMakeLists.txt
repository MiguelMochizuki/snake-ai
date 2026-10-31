cmake_minimum_required(VERSION 3.16)
project(snake_ai CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Dependencies are git submodules under deps/ (git submodule update --init)
set(SNAKE_DIR ${CMAKE_SOURCE_DIR}/deps/snake-game)
set(NN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NN_BUILD_APPS OFF CACHE BOOL "" FORCE)
add_subdirectory(deps/nn-cpp)

# Only the headless game logic; no terminal code
add_library(snake_logic STATIC
	${SNAKE_DIR}/src/entity.cpp
	${SNAKE_DIR}/src/food.cpp
	${SNAKE_DIR}/src/snake_segment.cpp
	${SNAKE_DIR}/src/snake.cpp
	${SNAKE_DIR}/src/game_state.cpp
)
target_include_directories(snake_logic PUBLIC ${SNAKE_DIR}/include)

add_library(snake_ai_core STATIC src/cycle.cpp src/snake_env.cpp src/expert.cpp src/heuristic.cpp src/tiles.cpp src/modes.cpp)
target_include_directories(snake_ai_core PUBLIC include)
target_link_libraries(snake_ai_core PUBLIC snake_logic)
target_compile_options(snake_ai_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(compare apps/compare.cpp)
target_link_libraries(compare PRIVATE snake_ai_core nn_core)
target_compile_options(compare PRIVATE -Wall -Wextra -Wpedantic)

find_package(Threads REQUIRED)
add_executable(evolve apps/evolve.cpp)
target_link_libraries(evolve PRIVATE snake_ai_core Threads::Threads)
target_compile_options(evolve PRIVATE -Wall -Wextra -Wpedantic)

add_executable(watch apps/watch.cpp)
target_link_libraries(watch PRIVATE snake_ai_core nn_core)
target_compile_options(watch PRIVATE -Wall -Wextra -Wpedantic)

add_executable(bfs_safe experiments/bfs_safe.cpp)
target_link_libraries(bfs_safe PRIVATE snake_ai_core)

add_executable(safe_modes experiments/safe_modes.cpp)
target_link_libraries(safe_modes PRIVATE snake_ai_core Threads::Threads)
target_compile_options(safe_modes PRIVATE -Wall -Wextra -Wpedantic)

add_executable(selector apps/selector.cpp)
target_link_libraries(selector PRIVATE snake_ai_core nn_core Threads::Threads)
target_compile_options(selector PRIVATE -Wall -Wextra -Wpedantic)

add_executable(imitate_astar apps/imitate_astar.cpp)
target_link_libraries(imitate_astar PRIVATE snake_ai_core nn_core Threads::Threads)
target_compile_options(imitate_astar PRIVATE -Wall -Wextra -Wpedantic)
