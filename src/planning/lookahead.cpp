/**
 * lookahead.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Lookahead over the safe moves
 */
#include "snake_ai/planning/lookahead.hpp"

#include <algorithm>

namespace {

/**
 * The moves a search may pick, or the forced answer
 *
 * Attributes:
 * std::vector<int> safe: Moves flagged safe by actionFeatures
 * int forced: A move to return without searching, or -1
 */
struct Candidates {
	std::vector<int> safe;
	int forced = -1;
};

/**
 * Parameters:
 * const Snapshot& s: Current game
 * SafeAgent& agent: The player's agent
 *
 * Returns Candidates: The safe moves; forced is set after the loop guard, with no safe move, or with only one
 */
Candidates candidatesOf(const Snapshot& s, SafeAgent& agent) {
	Candidates c;
	if (s.sinceFood > s.loopGuard()) {
		c.forced = teacherAction(s, agent);
		return c;
	}
	const std::vector<float> f = agent.actionFeatures(s);
	for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
		if (f[4 * a + 1] > 0.5f) c.safe.push_back(a);
	}
	if (c.safe.empty()) {
		c.forced = STRAIGHT;
		for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
			if (f[4 * a + 3] > 0.5f) c.forced = a;
		}
	} else if (c.safe.size() == 1) {
		c.forced = c.safe[0];
	}
	return c;
}

/**
 * Parameters:
 * const std::vector<double>& cost: Score of each candidate, in the order of safe
 * const std::vector<int>& safe: The candidates
 * int preferred: The baseline's move, which wins ties
 *
 * Returns int: The lowest-cost move
 */
int bestMove(const std::vector<double>& cost, const std::vector<int>& safe, int preferred) {
	size_t best = 0;
	for (size_t i = 1; i < safe.size(); ++i) {
		if (cost[i] < cost[best] || (cost[i] == cost[best] && safe[i] == preferred)) best = i;
	}
	return safe[best];
}

}  // namespace

int lookaheadHonest(const Snapshot& s, SafeAgent& agent, const LookaheadConfig& cfg) {
	Candidates c = candidatesOf(s, agent);
	if (c.forced >= 0) return c.forced;
	SafeAgent baseline = agent;
	const int preferred = teacherAction(s, baseline);
	const int cap = s.starveLimit();
	std::vector<double> cost;
	for (int a : c.safe) {
		double total = 0;
		for (int k = 0; k < cfg.samples; ++k) {
			SafeAgent rollout = agent;
			Sim sim(s, cfg.seed + static_cast<unsigned>(k));
			sim.step(a);
			while (!sim.dead() && sim.eaten() < cfg.foods && sim.steps() < cfg.foods * cap) {
				sim.step(teacherAction(sim.snapshot(), rollout));
			}
			// A rollout that ends early pays the starvation limit for every food it did not eat
			total += sim.steps() + std::max(0, cfg.foods - sim.eaten()) * cap;
		}
		cost.push_back(total / cfg.samples);
	}
	return bestMove(cost, c.safe, preferred);
}

int lookaheadLeaked(const SnakeEnv& env, SafeAgent& agent, int foods) {
	const Snapshot s = snapshotOf(env);
	Candidates c = candidatesOf(s, agent);
	if (c.forced >= 0) return c.forced;
	SafeAgent baseline = agent;
	const int preferred = teacherAction(s, baseline);
	const int cap = s.starveLimit();
	std::vector<double> cost;
	for (int a : c.safe) {
		SnakeEnv copy = env;  // copies the food generator too: the future is the real one
		SafeAgent rollout = agent;
		const int start = copy.score();
		int steps = 0;
		copy.step(a);
		++steps;
		while (!copy.dead() && copy.score() - start < foods && steps < foods * cap) {
			copy.step(teacherAction(snapshotOf(copy), rollout));
			++steps;
		}
		cost.push_back(steps + std::max(0, foods - (copy.score() - start)) * cap);
	}
	return bestMove(cost, c.safe, preferred);
}
