/**
 * test_sim.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Tests for the rollout simulator
 */

#include "doctest/doctest.h"
#include "snake_ai/planning/modes.hpp"
#include "snake_ai/planning/sim.hpp"

#include <algorithm>

TEST_CASE("Sim matches SnakeEnv step by step") {
	SnakeEnv env(1, 10);
	SafeAgent agent;
	for (int i = 0; i < 20000 && !env.dead(); ++i) {
		int a = teacherAction(snapshotOf(env), agent);
		Sim sim(snapshotOf(env), 99);
		bool simAte = sim.step(a);
		bool envAte = env.step(a);
		REQUIRE(simAte == envAte);
		REQUIRE(sim.dead() == env.dead());
		if (!env.dead()) {
			// After an eat the foods differ (each side draws its own), but the body does not depend on the new food
			REQUIRE(sim.snapshot().body == snapshotOf(env).body);
			REQUIRE(sim.snapshot().sinceFood == env.sinceFood());
			REQUIRE(sim.snapshot().dir == env.dir());
		}
	}
	CHECK(env.score() == env.maxScore());
}

TEST_CASE("Sim keeps the food off the body") {
	SnakeEnv env(2, 10);
	Sim sim(snapshotOf(env), 5);
	SafeAgent agent;
	while (!sim.dead() && sim.eaten() < 60) {
		sim.step(teacherAction(sim.snapshot(), agent));
		if (sim.dead()) break;
		Snapshot s = sim.snapshot();
		CHECK(std::find(s.body.begin(), s.body.end(), s.food) == s.body.end());
	}
	CHECK(sim.eaten() == 60);
}

TEST_CASE("Sim ends the game as won on the last free cell") {
	// 4x4 board, 15 cells serpentine from the tail at (0,0) to the head at (1,3); the food is the one free cell
	std::vector<Cell> path;
	for (int x = 0; x < 4; ++x) path.push_back({x, 0});
	for (int x = 3; x >= 0; --x) path.push_back({x, 1});
	for (int x = 0; x < 4; ++x) path.push_back({x, 2});
	for (int x = 3; x >= 1; --x) path.push_back({x, 3});
	std::reverse(path.begin(), path.end());  // head first
	Snapshot s{4, 2, path, {0, 3}, 0};
	Sim sim(s, 1);
	CHECK(sim.step(STRAIGHT));
	CHECK(sim.won());
	CHECK(sim.dead());
	CHECK(sim.eaten() == 1);
}

TEST_CASE("Sim dies on a wall and on starvation") {
	Snapshot wall{4, 2, {{0, 0}, {1, 0}, {2, 0}}, {3, 3}, 0};
	Sim a(wall, 1);
	a.step(STRAIGHT);
	CHECK(a.dead());
	CHECK_FALSE(a.won());

	Snapshot hungry{4, 3, {{1, 1}, {0, 1}, {0, 0}}, {3, 3}, 4 * 4 * 3 / 2 - 1};
	Sim b(hungry, 1);
	b.step(STRAIGHT);
	CHECK(b.dead());
}

TEST_CASE("Sim is deterministic for a snapshot and a seed") {
	SnakeEnv env(3, 10);
	Snapshot s = snapshotOf(env);
	auto run = [&](unsigned seed) {
		Sim sim(s, seed);
		SafeAgent agent;
		for (int i = 0; i < 300 && !sim.dead(); ++i) sim.step(teacherAction(sim.snapshot(), agent));
		return sim.snapshot();
	};
	Snapshot x = run(7), y = run(7);
	CHECK(x.body == y.body);
	CHECK(x.food == y.food);
}
