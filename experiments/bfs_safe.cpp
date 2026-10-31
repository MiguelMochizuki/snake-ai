/**
 * bfs_safe.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Experiment: shortest path to the food, accepted only if a simulation shows the tail stays reachable; otherwise chase the tail
 */
#include "snake_ai/env/snake_env.hpp"
#include <algorithm>
#include <cstdio>
#include <queue>
#include <cstdlib>
static const int W = 20, H = 20;
/**
 * Breadth-first search over free cells
 *
 * Parameters:
 * const SnakeEnv& e: Current game
 * int sx, int sy: Start cell
 * int gx, int gy: Target cell
 * std::vector<int>* path: If not null, receives the heading of each step
 *
 * Returns int: Distance to the target, -1 if unreachable
 */
// BFS over non-blocked cells;
static int bfs(const SnakeEnv& e, int sx, int sy, int gx, int gy, std::vector<int>* path) {
	std::vector<int> par(W * H, -2), pd(W * H, -1);
	std::queue<int> q; q.push(sy * W + sx); par[sy * W + sx] = -1;
	while (!q.empty()) {
		int c = q.front(); q.pop();
		int x = c % W, y = c / W;
		if (x == gx && y == gy) {
			int n = 0; if (path) { path->clear(); for (int k = c; par[k] != -1; k = par[k]) { path->push_back(pd[k]); } std::reverse(path->begin(), path->end()); }
			for (int k = c; par[k] != -1; k = par[k]) ++n;
			return n;
		}
		for (int d = 0; d < 4; ++d) {
			int nx = x + SnakeEnv::DX[d], ny = y + SnakeEnv::DY[d];
			if (nx < 0 || nx >= W || ny < 0 || ny >= H || e.blocked(nx, ny) || par[ny * W + nx] != -2) continue;
			par[ny * W + nx] = c; pd[ny * W + nx] = d; q.push(ny * W + nx);
		}
	}
	return -1;
}
/**
 * Finds the relative action that turns the current heading into the wanted one
 *
 * Parameters:
 * int dir: Current heading
 * int want: Wanted heading
 *
 * Returns int: Action (straight, right or left), -1 if it would reverse
 */
static int actionFor(int dir, int want) { for (int a = 0; a < 3; ++a) if (SnakeEnv::turn(dir, a) == want) return a; return -1; }
/**
 * Distance from the head to the tail, a check that the snake can still escape
 *
 * Parameters:
 * const SnakeEnv& e: Current game
 *
 * Returns int: BFS distance, -1 if the tail is unreachable
 */
static int tailDist(const SnakeEnv& e) {
	auto h = e.state().getSnake().getHead(); auto& t = e.state().getSnake().getSegments().back();
	return bfs(e, h->getX(), h->getY(), t.getX(), t.getY(), nullptr);
}
/**
 * Takes the shortest path to the food if a simulation shows the tail stays reachable, otherwise chases the tail
 *
 * Parameters:
 * const SnakeEnv& e: Current game
 *
 * Returns int: Action to take
 */
int safeAction(const SnakeEnv& e) {
	auto h = e.state().getSnake().getHead(); auto& f = e.state().getFood();
	std::vector<int> path;
	if (bfs(e, h->getX(), h->getY(), f.getX(), f.getY(), &path) > 0) {
		SnakeEnv sim = e; bool ok = true;
		for (int d : path) { int a = actionFor(sim.dir(), d); if (a < 0) { ok = false; break; } sim.step(a); if (sim.dead()) { ok = false; break; } }
		if (ok && tailDist(sim) >= 0) { int a = actionFor(e.dir(), path[0]); if (a >= 0) return a; }
	}
	// Chase the tail: keep the tail reachable and stay as far from it as possible
	int best = -1, bestD = -1;
	for (int a = 0; a < 3; ++a) {
		int d = SnakeEnv::turn(e.dir(), a); if (e.blocked(h->getX() + SnakeEnv::DX[d], h->getY() + SnakeEnv::DY[d])) continue;
		SnakeEnv sim = e; sim.step(a); if (sim.dead()) continue;
		int td = tailDist(sim); if (td > bestD) { bestD = td; best = a; }
	}
	return best >= 0 ? best : 0;
}
/**
 * Plays the BFS-safe agent and prints each game's score, steps and cause of death
 *
 * Parameters:
 * int argc: Argument count
 * char** argv: argv[1] is the number of games (default 10)
 *
 * Returns int: 0
 */
int main(int argc, char** argv) {
	int games = argc > 1 ? atoi(argv[1]) : 10; double sc = 0, st = 0; int deaths = 0;
	for (int g = 0; g < games; ++g) {
		SnakeEnv e(100000 + g); int i = 0;
		for (; i < 60000 && !e.dead(); ++i) e.step(safeAction(e));
		sc += e.score(); st += i; deaths += e.dead() && e.score() < e.maxScore();
		std::printf("game %d score %d steps %d %s\n", g, e.score(), i, e.dead() && e.score() < e.maxScore() ? (e.sinceFood() >= 600 ? "starved" : "collided") : "");
		std::fflush(stdout);
	}
	std::printf("mean score %.1f  mean steps %.0f  deaths %d/%d\n", sc / games, st / games, deaths, games);
}
