/**
 * test_modes.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the safe playing modes
 */

#include "doctest/doctest.h"
#include "snake_ai/planning/modes.hpp"

TEST_CASE("every mode fills a 10x10 board without dying") {
	for (int m = 0; m < MODES; ++m) {
		SnakeEnv env(1, 10);
		SafeAgent agent;
		for (int i = 0; i < 200 * 100 && !env.dead(); ++i) {
			env.step(agent.act(env, static_cast<Mode>(m)));
		}
		CHECK(env.score() == env.maxScore());
	}
}

TEST_CASE("feature vectors have the declared sizes and stay in 0..1") {
	SnakeEnv env(2, 10);
	SafeAgent agent;
	for (int i = 0; i < 50 && !env.dead(); ++i) {
		std::vector<float> f = agent.features(env), a = agent.actionFeatures(env);
		REQUIRE(f.size() == static_cast<size_t>(SafeAgent::FEATURES));
		REQUIRE(a.size() == static_cast<size_t>(SafeAgent::ACTION_FEATURES));
		for (float x : f) CHECK((x >= 0.0f && x <= 1.0f));
		for (float x : a) CHECK((x >= 0.0f && x <= 1.0f));
		env.step(agent.act(env, ASTAR));
	}
}
