cmake_minimum_required(VERSION 3.16)
project(snake_ai CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Dependencies are fetched at pinned versions. For local development against
# your own checkouts, pass -DFETCHCONTENT_SOURCE_DIR_SNAKE_GAME=../snake-game
# and -DFETCHCONTENT_SOURCE_DIR_NN_CPP=../nn-cpp
include(FetchContent)
FetchContent_Declare(snake_game
	GIT_REPOSITORY https://github.com/MiguelMochizuki/snake-game.git
	GIT_TAG v2.0.0
	SOURCE_SUBDIR no-cmake  # only its sources are used, not its build
)
FetchContent_Declare(nn_cpp
	GIT_REPOSITORY https://github.com/MiguelMochizuki/nn-cpp.git
	GIT_TAG bc06012c7bdaabfcdcfc5e43cd2b034f98ffb616
)
set(NN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NN_BUILD_APPS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(snake_game nn_cpp)
set(SNAKE_DIR ${snake_game_SOURCE_DIR})

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
