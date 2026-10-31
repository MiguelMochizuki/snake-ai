/**
 * scale_test.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Plays the safe A* player and the network trained on 20x20 on other board sizes, without retraining
 */
#include "astar_policy.hpp"
#include "nn/checkpoint.hpp"

#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

struct Result { double score = 0, steps = 0; int notFull = 0; };

/**
 * Plays games on an n x n board
 *
 * Parameters:
 * int n: Board side
 * int games: Number of games
 * std::function<int(SnakeEnv&, SafeAgent&)> act: Chooses the action
 * bool parallel: Play the games on threads (not for the network)
 *
 * Returns Result: Means over the games; notFull counts games that did not fill the board
 */
template <class Act>
Result play(int n, int games, Act act, bool parallel) {
	std::vector<Result> parts(games);
	auto one = [&](int g) {
		SnakeEnv env(100000 + g, n);
		SafeAgent agent;
		int i = 0;
		for (; i < 200 * n * n && !env.dead(); ++i) {
			env.step(act(env, agent));
		}
		parts[g] = {static_cast<double>(env.score()), static_cast<double>(i), env.score() < env.maxScore()};
	};
	if (parallel) {
		std::vector<std::thread> pool;
		for (int g = 0; g < games; ++g) pool.emplace_back(one, g);
		for (auto& t : pool) t.join();
	} else {
		for (int g = 0; g < games; ++g) one(g);
	}
	Result r;
	for (auto& p : parts) {
		r.score += p.score / games;
		r.steps += p.steps / games;
		r.notFull += static_cast<int>(p.notFull);
	}
	return r;
}

void show(int n, int games, const char* name, const Result& r) {
	std::printf("%2dx%-2d %-24s score %7.1f / %-5d steps %8.0f (%.1f per cell)  not full %d/%d\n", n, n, name, r.score,
		n * n - 3, r.steps, r.steps / (n * n), r.notFull, games);
	std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
	nn::Sequential net = makeAstarNet();
	nn::load_checkpoint(argc > 1 ? argv[1] : "models/imitation_astar.ckpt", net);
	for (int n : {10, 20, 40}) {
		int games = n == 40 ? 4 : 12;
		show(n, games, "A* teacher", play(n, games, [](SnakeEnv& e, SafeAgent& a) {
			return a.act(e, e.sinceFood() > e.loopGuard() ? CYCLE : ASTAR);
		}, true));
		show(n, games, "cycle", play(n, games, [](SnakeEnv& e, SafeAgent& a) { return a.act(e, CYCLE); }, true));
		show(n, games, "network (20x20 weights)", play(n, games, [&](SnakeEnv& e, SafeAgent& a) {
			return astarNetAction(net, a.actionFeatures(e), e.sinceFood() > e.loopGuard(), true);
		}, false));
	}
	return 0;
}
