/**
 * test_features.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the extended action features and the widened network
 */

#include "doctest/doctest.h"
#include "snake_ai/env/cycle.hpp"
#include "snake_ai/planning/modes.hpp"
#include "snake_ai/policy/astar_policy.hpp"
#include "nn/checkpoint.hpp"

#include <cstdlib>
#include <stdexcept>

#ifndef SNAKE_AI_SOURCE_DIR
#define SNAKE_AI_SOURCE_DIR "."
#endif

namespace {

const char* const CKPT = SNAKE_AI_SOURCE_DIR "/models/imitation_astar.ckpt";

/** Plays the teacher for a number of steps on a 10x10 board and calls visit(snapshot) before each move */
template <typename F>
void playTeacher(int steps, F visit) {
	SnakeEnv env(7, 10);
	SafeAgent agent;
	for (int i = 0; i < steps && !env.dead(); ++i) {
		Snapshot s = snapshotOf(env);
		visit(s, agent);
		env.step(teacherAction(s, agent));
	}
}

}  // namespace

TEST_CASE("extended features have the declared size and stay in 0..1") {
	playTeacher(100, [](const Snapshot& s, SafeAgent& agent) {
		std::vector<float> f = agent.extendedFeatures(s);
		REQUIRE(f.size() == static_cast<size_t>(SafeAgent::EXT_FEATURES));
		for (float x : f) CHECK((x >= 0.0f && x <= 1.0f));
	});
}

TEST_CASE("extended prefix equals actionFeatures") {
	playTeacher(100, [](const Snapshot& s, SafeAgent& agent) {
		std::vector<float> base = agent.actionFeatures(s), ext = agent.extendedFeatures(s);
		CHECK(std::equal(base.begin(), base.end(), ext.begin()));
	});
}

TEST_CASE("widened network reproduces the old one") {
	nn::Sequential old = makeAstarNet(), wide = makeExtendedNet();
	nn::load_checkpoint(CKPT, old);
	widenFromAstar(wide, old);
	playTeacher(200, [&](const Snapshot& s, SafeAgent& agent) {
		std::vector<float> ext = agent.extendedFeatures(s);
		std::vector<float> base(ext.begin(), ext.begin() + SafeAgent::ACTION_FEATURES);
		nn::NoGradGuard noGrad;
		nn::Storage a = old.forward(nn::Value::leaf(nn::Tensor({1, base.size()}, base), false)).data().data();
		nn::Storage b = wide.forward(nn::Value::leaf(nn::Tensor({1, ext.size()}, ext), false)).data().data();
		REQUIRE(a.size() == b.size());
		for (size_t i = 0; i < a.size(); ++i) CHECK(b[i] == doctest::Approx(a[i]).epsilon(1e-6));
	});
}

TEST_CASE("loading the old checkpoint directly into the extended net throws") {
	nn::Sequential wide = makeExtendedNet();
	CHECK_THROWS(nn::load_checkpoint(CKPT, wide));
}

TEST_CASE("reachable cells after a move, checked by hand on a 4x4 board") {
	// Head (1,0) heading right, tail (0,0). Straight is (2,0), right is (1,1), left leaves the board.
	// Without eating, the tail cell is free again, so only the head cell is out: 15 of 16 cells are reachable.
	Snapshot s{4, 3, {{1, 0}, {0, 0}}, {3, 3}, 0};
	SafeAgent agent;
	std::vector<float> f = agent.extendedFeatures(s);
	const int base = SafeAgent::ACTION_FEATURES;  // per action: reachable, cycle distance to the tail, cycle distance to the food
	CHECK(f[base + 0] == doctest::Approx(15.0f / 16.0f));
	CHECK(f[base + 3] == doctest::Approx(15.0f / 16.0f));
	CHECK(f[base + 6] == 1.0f);  // blocked: 1, 0, 1
	CHECK(f[base + 7] == 0.0f);
	CHECK(f[base + 8] == 1.0f);

	// Eating at (2,0) keeps the tail in the body: 14 cells are reachable, and the cycle distance to the food is 0
	s.food = {2, 0};
	f = agent.extendedFeatures(s);
	CHECK(f[base + 0] == doctest::Approx(14.0f / 16.0f));
	CHECK(f[base + 2] == 0.0f);
}

TEST_CASE("cycleDist wraps around and cycleIndex walks to a neighbour each step") {
	CHECK(cycleDist(5, 5) == 0);
	CHECK(cycleDist(399, 0) == 1);
	CHECK(cycleDist(0, 399) == 399);
	std::vector<std::pair<int, int>> at(CYCLE_LEN);
	for (int y = 0; y < 20; ++y)
		for (int x = 0; x < 20; ++x) at[cycleIndex(x, y)] = {x, y};
	for (int i = 0; i < CYCLE_LEN; ++i) {
		const auto a = at[i], b = at[(i + 1) % CYCLE_LEN];
		CHECK(std::abs(a.first - b.first) + std::abs(a.second - b.second) == 1);
	}
}
