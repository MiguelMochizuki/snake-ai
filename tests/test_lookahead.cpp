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
