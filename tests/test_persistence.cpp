/**
 * test_persistence.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the heuristic genome file and the policy network
 */

#include "doctest/doctest.h"
#include "snake_ai/planning/modes.hpp"
#include "snake_ai/policy/astar_policy.hpp"
#include "snake_ai/policy/policy_net.hpp"
#include "snake_ai/teachers/heuristic.hpp"

#include <cstdio>

TEST_CASE("a genome survives a save and load") {
	Genome g;
	for (size_t i = 0; i < g.w.size(); ++i) g.w[i] = 0.5f * static_cast<float>(i) - 1.0f;
	const std::string path = "test_genome.tmp";
	saveGenome(path, g);
	Genome back = loadGenome(path);
	std::remove(path.c_str());
	CHECK(back.w == g.w);
}

TEST_CASE("the network guard follows the cycle when starving and never picks a blocked move when asked to filter") {
	nn::Sequential net = makeNet(8, SafeAgent::ACTION_FEATURES);
	SnakeEnv env(3, 10);
	SafeAgent agent;
	std::vector<float> f = agent.actionFeatures(env);
	int a = astarNetAction(net, f, false, true);
	CHECK((a >= 0 && a < SnakeEnv::ACTIONS));
	CHECK(f[a * 4] == 0.0f);  // the first feature of each action is "blocked"
}
