/**
 * test_lookahead.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Tests for the lookahead chooser
 */

#include "doctest/doctest.h"
#include "snake_ai/planning/lookahead.hpp"
#include "snake_ai/planning/modes.hpp"

namespace {

/**
 * Tells whether an action was one of the safe moves of a state. In real games a safe move always exists, so the lookahead's
 * "no safe move" fallback is only a guard (SafeAgent itself requires the body to lie on a cycle, so such a state cannot be built by hand)
 *
 * Parameters:
 * const std::vector<float>& f: SafeAgent::actionFeatures of the state
 * int a: The chosen action
 *
 * Returns bool: true if a is safe
 */
bool isSafe(const std::vector<float>& f, int a) {
	return f[4 * a + 1] > 0.5f;
}

}  // namespace

TEST_CASE("honest lookahead fills a 10x10 board with no deaths, always on safe moves") {
	LookaheadConfig cfg;
	cfg.foods = 2;
	cfg.samples = 4;
	for (unsigned seed : {1u, 2u}) {
		SnakeEnv env(seed, 10);
		SafeAgent agent;
		for (int i = 0; i < 200 * 100 && !env.dead(); ++i) {
			Snapshot s = snapshotOf(env);
			int a = lookaheadHonest(s, agent, cfg);
			REQUIRE(isSafe(agent.actionFeatures(s), a));
			env.step(a);
		}
		CHECK(env.score() == env.maxScore());
	}
}

TEST_CASE("honest lookahead decides from the snapshot alone") {
	LookaheadConfig cfg;
	cfg.foods = 2;
	cfg.samples = 4;
	SnakeEnv env(1, 10);
	SafeAgent agent;
	for (int i = 0; i < 150 && !env.dead(); ++i) env.step(teacherAction(snapshotOf(env), agent));
	Snapshot s = snapshotOf(env);
	int first = lookaheadHonest(s, agent, cfg);
	// The game moves on and its food generator advances; the same snapshot must still give the same answer
	for (int i = 0; i < 40 && !env.dead(); ++i) env.step(teacherAction(snapshotOf(env), agent));
	CHECK(lookaheadHonest(s, agent, cfg) == first);
}

TEST_CASE("leaked lookahead fills a 10x10 board with no deaths") {
	SnakeEnv env(1, 10);
	SafeAgent agent;
	for (int i = 0; i < 200 * 100 && !env.dead(); ++i) {
		int a = lookaheadLeaked(env, agent, 2);
		REQUIRE(isSafe(agent.actionFeatures(env), a));
		env.step(a);
	}
	CHECK(env.score() == env.maxScore());
}

TEST_CASE("lookahead returns the teacher action after the loop guard") {
	SnakeEnv env(1, 10);
	SafeAgent agent, other;
	for (int i = 0; i < 60 && !env.dead(); ++i) env.step(teacherAction(snapshotOf(env), agent));
	Snapshot s = snapshotOf(env);
	s.sinceFood = s.loopGuard() + 1;
	CHECK(lookaheadHonest(s, agent, LookaheadConfig{}) == teacherAction(s, other));
}

namespace {

/**
 * Shortest path to the food over cells that are free right now, with the body as a fixed wall (what the baseline A* assumes)
 *
 * Parameters:
 * const Snapshot& s: Game
 * Cell from: Cell the head enters at step t0
 * int t0: Step at which the head enters from
 *
 * Returns int: Step at which the head reaches the food, or -1
 */
int staticDistance(const Snapshot& s, Cell from, int t0) {
	std::vector<int> dist(s.size * s.size, -1);
	std::vector<Cell> queue{from};
	dist[from.second * s.size + from.first] = t0;
	for (size_t i = 0; i < queue.size(); ++i) {
		Cell c = queue[i];
		if (c == s.food) return dist[c.second * s.size + c.first];
		for (int d = 0; d < 4; ++d) {
			Cell n{c.first + SnakeEnv::DX[d], c.second + SnakeEnv::DY[d]};
			if (s.blocked(n.first, n.second) || dist[n.second * s.size + n.first] >= 0) continue;
			dist[n.second * s.size + n.first] = dist[c.second * s.size + c.first] + 1;
			queue.push_back(n);
		}
	}
	return -1;
}

}  // namespace

TEST_CASE("timedDistance on a short snake is the plain distance") {
	Snapshot s{10, 3, {{5, 5}, {4, 5}, {3, 5}}, {8, 5}, 0};
	CHECK(timedDistance(s, {6, 5}, 1) == 3);
}

TEST_CASE("timedDistance finds a path the static wall hides because the tail moves away") {
	// The food at (0,0) is walled in by the body, but (0,1) is vacated by step 2 and the head can get there by step 5
	Snapshot s{4, 1, {{2, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 2}}, {0, 0}, 0};
	CHECK(staticDistance(s, {2, 1}, 1) == -1);
	CHECK(timedDistance(s, {2, 1}, 1) == 6);
}

TEST_CASE("timedDistance is never longer than the static one and is sometimes shorter") {
	int shorter = 0;
	for (unsigned seed : {1u, 2u}) {
		SnakeEnv env(seed, 20);
		SafeAgent agent;
		for (int i = 0; i < 6000 && !env.dead(); ++i) {
			Snapshot s = snapshotOf(env);
			for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
				int d = SnakeEnv::turn(s.dir, a);
				Cell to{s.body[0].first + SnakeEnv::DX[d], s.body[0].second + SnakeEnv::DY[d]};
				if (s.blocked(to.first, to.second)) continue;
				int st = staticDistance(s, to, 1), ti = timedDistance(s, to, 1);
				if (st >= 0) {
					REQUIRE(ti >= 0);
					REQUIRE(ti <= st);
				}
				shorter += (st < 0 && ti >= 0) || (st >= 0 && ti < st);
			}
			env.step(teacherAction(s, agent));
		}
	}
	CHECK(shorter > 0);
}

TEST_CASE("timed A* fills a 10x10 board with no deaths, always on safe moves") {
	for (unsigned seed : {1u, 2u}) {
		SnakeEnv env(seed, 10);
		SafeAgent agent;
		for (int i = 0; i < 200 * 100 && !env.dead(); ++i) {
			Snapshot s = snapshotOf(env);
			int a = timedAstarAction(s, agent);
			REQUIRE(isSafe(agent.actionFeatures(s), a));
			env.step(a);
		}
		CHECK(env.score() == env.maxScore());
	}
}
