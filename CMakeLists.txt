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

add_library(snake_ai_core STATIC src/snake_env.cpp src/expert.cpp)
target_include_directories(snake_ai_core PUBLIC include)
target_link_libraries(snake_ai_core PUBLIC snake_logic)
target_compile_options(snake_ai_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(train_imitation apps/train_imitation.cpp)
target_link_libraries(train_imitation PRIVATE snake_ai_core nn_core)
target_compile_options(train_imitation PRIVATE -Wall -Wextra -Wpedantic)
