/**
 * test_snapshot.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Tests for the Snapshot and for SafeAgent deciding from it
 */

#include "doctest/doctest.h"
#include "snake_ai/planning/modes.hpp"
#include "snake_ai/planning/snapshot.hpp"

#include <cstdint>

namespace {

/**
 * Plays a full game with one mode and folds the chosen actions into a hash
 *
 * Parameters:
 * unsigned seed: Food seed
 * int size: Board side
 * Mode mode: Mode to play
 *
 * Returns uint64_t: FNV-1a hash of the action sequence, with the number of steps mixed in
 */
uint64_t actionHash(unsigned seed, int size, Mode mode) {
	SnakeEnv env(seed, size);
	SafeAgent agent;
	uint64_t h = 1469598103934665603ULL;
	int steps = 0;
	for (; steps < 200 * size * size && !env.dead(); ++steps) {
		int a = agent.act(env, mode);
		h = (h ^ static_cast<uint64_t>(a + 1)) * 1099511628211ULL;
		env.step(a);
	}
	return (h ^ static_cast<uint64_t>(steps)) * 1099511628211ULL;
}

}  // namespace

TEST_CASE("golden: env-based act reproduces the recorded action hashes") {
	// Recorded on the code before the Snapshot refactor, so the refactor is proven behaviour-preserving
	struct Row { unsigned seed; Mode mode; uint64_t hash; };
	const Row rows[] = {
		{1, ASTAR, 4656895045291497072ULL}, {2, ASTAR, 18367596375677857797ULL}, {3, ASTAR, 7628100582265230092ULL},
		{1, SHORTCUT, 12459030633572056041ULL}, {2, SHORTCUT, 11551195966136740348ULL}, {3, SHORTCUT, 16390811268619313424ULL},
		{1, CYCLE, 7733138758415757549ULL}, {2, CYCLE, 10027401117844486575ULL}, {3, CYCLE, 15449667583441357181ULL},
	};
	for (const Row& r : rows) {
		CAPTURE(r.seed);
		CAPTURE(static_cast<int>(r.mode));
		CHECK(actionHash(r.seed, 10, r.mode) == r.hash);
	}
}

TEST_CASE("snapshotOf mirrors the env") {
	SnakeEnv env(1, 10);
	SafeAgent agent;
	for (int i = 0; i < 30 && !env.dead(); ++i) env.step(agent.act(env, ASTAR));
	Snapshot s = snapshotOf(env);
	CHECK(s.body == bodyCells(env));
	CHECK(s.size == 10);
	CHECK(s.dir == env.dir());
	CHECK(s.sinceFood == env.sinceFood());
	CHECK(s.food == Cell{env.state().getFood().getX(), env.state().getFood().getY()});
	CHECK(s.loopGuard() == env.loopGuard());
	CHECK(s.starveLimit() == env.starveLimit());
	for (int y = -1; y <= 10; ++y) {
		for (int x = -1; x <= 10; ++x) {
			CAPTURE(x);
			CAPTURE(y);
			CHECK(s.blocked(x, y) == env.blocked(x, y));
		}
	}
}

TEST_CASE("snapshot act and features equal the env versions") {
	for (int m = 0; m < MODES; ++m) {
		SnakeEnv env(1, 10);
		SafeAgent viaEnv, viaSnap;
		for (int i = 0; i < 300 && !env.dead(); ++i) {
			Snapshot s = snapshotOf(env);
			CHECK(viaSnap.actionFeatures(s) == viaEnv.actionFeatures(env));
			CHECK(viaSnap.features(s) == viaEnv.features(env));
			int a = viaEnv.act(env, static_cast<Mode>(m));
			REQUIRE(viaSnap.act(s, static_cast<Mode>(m)) == a);
			env.step(a);
		}
	}
}

TEST_CASE("teacherAction follows the cycle after the loop guard") {
	SnakeEnv env(1, 10);
	SafeAgent a, b;
	Snapshot s = snapshotOf(env);
	CHECK(teacherAction(s, a) == b.act(s, ASTAR));
	s.sinceFood = s.loopGuard() + 1;
	CHECK(teacherAction(s, a) == b.act(s, CYCLE));
}
