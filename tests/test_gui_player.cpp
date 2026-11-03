/**
 * test_gui_player.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Tests that a game can be drawn through snake-game's SDL renderer off-screen (needs SNAKE_AI_BUILD_GUI)
 */

#include "doctest/doctest.h"
#include "snake_ai/gui/gui_player.hpp"
#include "snake_ai/planning/modes.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

/**
 * Makes a fresh empty directory for one test
 *
 * Parameters:
 * const std::string& name: Directory name inside the system temp folder
 *
 * Returns std::filesystem::path: The directory
 */
std::filesystem::path freshDir(const std::string& name) {
	std::filesystem::path dir = std::filesystem::temp_directory_path() / name;
	std::filesystem::remove_all(dir);
	std::filesystem::create_directories(dir);
	return dir;
}

/**
 * Reads a whole file
 *
 * Parameters:
 * const std::filesystem::path& path: File to read
 *
 * Returns std::string: Its bytes
 */
std::string slurp(const std::filesystem::path& path) {
	std::ifstream in(path, std::ios::binary);
	return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

}  // namespace

TEST_CASE("playing off-screen writes PPM frames of the real renderer") {
	std::filesystem::path dir = freshDir("snake_ai_gui_player_test");
	SnakeEnv env(1, 10);
	SafeAgent agent;
	int frames = playToFrames(env, [&](SnakeEnv& e) { return agent.act(e, CYCLE); }, dir.string(), 5, 400);
	REQUIRE(frames > 3);
	std::string first = slurp(dir / "frame-00000.ppm");
	REQUIRE(first.rfind("P6\n", 0) == 0);
	CHECK(slurp(dir / "frame-00001.ppm") != first);  // the snake moved between frames
	char last[32];
	std::snprintf(last, sizeof last, "frame-%05d.ppm", frames - 1);
	CHECK(std::filesystem::exists(dir / last));
	std::snprintf(last, sizeof last, "frame-%05d.ppm", frames);
	CHECK(!std::filesystem::exists(dir / last));
	std::filesystem::remove_all(dir);
}

TEST_CASE("the last state is always written, whatever the stride") {
	std::filesystem::path dir = freshDir("snake_ai_gui_player_last");
	SnakeEnv env(1, 10);
	int steps = 0;
	int frames = playToFrames(env, [&](SnakeEnv&) { ++steps; return 0; }, dir.string(), 1000, 3);  // goes straight for three steps, before the wall
	CHECK(steps == 3);
	CHECK(frames == 2);  // the first state and the last one
	std::filesystem::remove_all(dir);
}
