/**
 * safe_modes.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Checks the tile-tree cycle construction, then plays the three modes and some fixed switching schedules
 */
#include "snake_ai/planning/modes.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <thread>

namespace {

/**
 * Checks, for an n x n board, that random spanning trees give one Hamiltonian cycle and that treeFor accepts any prefix of it as a body
 */
bool checkConstruction(int n) {
	std::mt19937 rng(3);
	for (int trial = 0; trial < 50; ++trial) {
		// Random spanning tree by random edge order
		const int t = n / 2, edgeCount = tileEdgeCount(n);
		std::vector<int> edges(edgeCount);
		for (int i = 0; i < edgeCount; ++i) edges[i] = i;
		std::shuffle(edges.begin(), edges.end(), rng);
		Tree tree(edgeCount, 0);
		std::vector<int> p(t * t);
		for (int i = 0; i < t * t; ++i) p[i] = i;
		std::function<int(int)> find = [&](int x) { return p[x] == x ? x : p[x] = find(p[x]); };
		for (int e : edges) {
			int a, b;
			tileEdgeEnds(n, e, &a, &b);
			if (find(a) != find(b)) {
				p[find(a)] = find(b);
				tree[e] = 1;
			}
		}
		// Every cell must have exactly two cycle neighbours, and the walk from a cell must visit all 400
		static const int DX[4] = {0, 0, -1, 1}, DY[4] = {-1, 1, 0, 0};
		Cell start{0, 0}, neck{-1, -1};
		for (int d = 0; d < 4; ++d) {
			Cell c{DX[d], DY[d]};
			if (c.first >= 0 && c.second >= 0 && onCycle(tree, n, start, c)) neck = c;
		}
		for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) {
			int degree = 0;
			for (int d = 0; d < 4; ++d) {
				int nx = x + DX[d], ny = y + DY[d];
				if (nx >= 0 && nx < n && ny >= 0 && ny < n && onCycle(tree, n, {x, y}, {nx, ny})) ++degree;
			}
			if (degree != 2) { std::printf("trial %d: cell (%d,%d) has %d cycle neighbours\n", trial, x, y, degree); return false; }
		}
		std::vector<int> order = cycleOrder(tree, n, start, neck);
		for (int v : order) if (v < 0) { std::printf("trial %d: cycle misses cells\n", trial); return false; }
		// A body of the first k cells along the cycle must be accepted
		std::vector<Cell> cells(n * n);
		for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) cells[order[y * n + x]] = {x, y};
		for (int k : {2, 10, 77, 250, n * n}) {
			if (k > n * n) continue;
			std::vector<Cell> body(cells.begin(), cells.begin() + k);
			Tree t;
			if (!treeFor(body, n, t) || !consistent(t, n, body)) { std::printf("trial %d: prefix %d rejected\n", trial, k); return false; }
		}
	}
	return true;
}

struct Result { double score = 0, steps = 0; int died = 0; };

/**
 * Plays games with a schedule that picks the mode from the snake length
 *
 * Parameters:
 * std::function<Mode(const SnakeEnv&)> pick: Mode for the current state
 * int games: Number of games
 *
 * Returns Result: Totals over the games
 */
Result play(const std::function<Mode(const SnakeEnv&)>& pick, int games) {
	std::vector<Result> parts(games);
	std::vector<std::thread> pool;
	for (int g = 0; g < games; ++g) {
		pool.emplace_back([&, g] {
			SnakeEnv env(100000 + g);
			SafeAgent agent;
			int i = 0;
			for (; i < 200 * env.size() * env.size() && !env.dead(); ++i) {
				Mode m = env.sinceFood() > env.loopGuard() ? CYCLE : pick(env);  // guards against loops
				env.step(agent.act(env, m));
			}
			parts[g] = {static_cast<double>(env.score()), static_cast<double>(i), env.dead() && env.score() < 397};
		});
	}
	for (auto& t : pool) t.join();
	Result r;
	for (auto& p : parts) { r.score += p.score / games; r.steps += p.steps / games; r.died += p.died; }
	return r;
}

}  // namespace

int main(int argc, char** argv) {
	int games = argc > 1 ? std::atoi(argv[1]) : 12;
	for (int n : {10, 20, 40}) std::printf("construction %dx%d: %s\n", n, n, checkConstruction(n) ? "ok" : "FAILED");
	std::fflush(stdout);
	auto show = [&](const char* name, std::function<Mode(const SnakeEnv&)> pick) {
		Result r = play(pick, games);
		std::printf("%-28s score %6.1f  steps %7.0f  deaths %d/%d\n", name, r.score, r.steps, r.died, games);
		std::fflush(stdout);
	};
	show("cycle", [](const SnakeEnv&) { return CYCLE; });
	show("shortcuts", [](const SnakeEnv&) { return SHORTCUT; });
	show("A*", [](const SnakeEnv&) { return ASTAR; });
	for (int len : {100, 200, 300}) {
		char name[48];
		std::snprintf(name, sizeof name, "A* until %d, then shortcuts", len);
		show(name, [len](const SnakeEnv& e) { return e.state().getSnake().getLength() < static_cast<size_t>(len) ? ASTAR : SHORTCUT; });
		std::snprintf(name, sizeof name, "shortcuts until %d, then A*", len);
		show(name, [len](const SnakeEnv& e) { return e.state().getSnake().getLength() < static_cast<size_t>(len) ? SHORTCUT : ASTAR; });
		std::snprintf(name, sizeof name, "A* until %d, then cycle", len);
		show(name, [len](const SnakeEnv& e) { return e.state().getSnake().getLength() < static_cast<size_t>(len) ? ASTAR : CYCLE; });
	}
	return 0;
}
