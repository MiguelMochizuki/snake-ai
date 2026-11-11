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

int timedDistance(const Snapshot& s, Cell from, int t0) {
	const int n = s.size, len = static_cast<int>(s.body.size());
	// freeFrom[c]: first step at which the head may enter the cell (0 for cells off the body)
	std::vector<int> freeFrom(n * n, 0), arrival(n * n, -1);
	for (int i = 0; i < len; ++i) freeFrom[s.body[i].second * n + s.body[i].first] = len - i;
	if (from.first < 0 || from.first >= n || from.second < 0 || from.second >= n || t0 < freeFrom[from.second * n + from.first]) return -1;
	std::vector<Cell> queue{from};
	arrival[from.second * n + from.first] = t0;
	for (size_t i = 0; i < queue.size(); ++i) {
		const Cell c = queue[i];
		const int t = arrival[c.second * n + c.first];
		if (c == s.food) return t;
		for (int d = 0; d < 4; ++d) {
			const Cell next{c.first + SnakeEnv::DX[d], c.second + SnakeEnv::DY[d]};
			if (next.first < 0 || next.first >= n || next.second < 0 || next.second >= n) continue;
			const int k = next.second * n + next.first;
			// Entering at the first arrival is exact: a cell that is free at step t stays free later
			if (arrival[k] >= 0 || t + 1 < freeFrom[k]) continue;
			arrival[k] = t + 1;
			queue.push_back(next);
		}
	}
	return -1;
}

int timedAstarAction(const Snapshot& s, SafeAgent& agent) {
	Candidates c = candidatesOf(s, agent);
	if (c.forced >= 0) return c.forced;
	SafeAgent baseline = agent;
	const int preferred = teacherAction(s, baseline);
	std::vector<double> cost;
	for (int a : c.safe) {
		const int d = SnakeEnv::turn(s.dir, a);
		const int t = timedDistance(s, {s.body[0].first + SnakeEnv::DX[d], s.body[0].second + SnakeEnv::DY[d]}, 1);
		cost.push_back(t < 0 ? s.starveLimit() : t);
	}
	return bestMove(cost, c.safe, preferred);
}
