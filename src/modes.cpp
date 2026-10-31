/**
 * modes.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Safe playing modes
 */
#include "modes.hpp"
#include <algorithm>
#include <cstdlib>
#include <queue>

std::vector<Cell> bodyCells(const SnakeEnv& env) {
	std::vector<Cell> body;
	for (const SnakeSegment& s : env.state().getSnake().getSegments()) {
		body.push_back({s.getX(), s.getY()});
	}
	return body;
}

int SafeAgent::act(const SnakeEnv& env, Mode mode) {
	const int W = 20;
	std::vector<Cell> body = bodyCells(env);
	const Cell head = body[0];
	const Food& food = env.state().getFood();

	if (!haveTree_ || !consistent(tree_, body)) {
		haveTree_ = treeFor(body, tree_);
	}
	// Cell the action leads to
	auto target = [&](int a) {
		int d = SnakeEnv::turn(env.dir(), a);
		return Cell{head.first + SnakeEnv::DX[d], head.second + SnakeEnv::DY[d]};
	};

	std::vector<int> order = cycleOrder(tree_, head, body[1]);
	auto cycleStep = [&]() {
		for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
			Cell c = target(a);
			if (c.first >= 0 && c.first < W && c.second >= 0 && c.second < W && order[c.second * W + c.first] == 1) return a;
		}
		return static_cast<int>(STRAIGHT);
	};
	if (mode == CYCLE) {
		return cycleStep();
	}

	// ASTAR and SHORTCUT: rank the moves that keep the body on a Hamiltonian cycle
	std::vector<int> dist(W * W, -1);
	std::queue<Cell> q;
	dist[food.getY() * W + food.getX()] = 0;
	q.push({food.getX(), food.getY()});
	while (!q.empty()) {
		Cell c = q.front();
		q.pop();
		for (int d = 0; d < 4; ++d) {
			int nx = c.first + SnakeEnv::DX[d], ny = c.second + SnakeEnv::DY[d];
			if (env.blocked(nx, ny) || dist[ny * W + nx] >= 0) continue;
			dist[ny * W + nx] = dist[c.second * W + c.first] + 1;
			q.push({nx, ny});
		}
	}
	const int rf = order[food.getY() * W + food.getX()];
	int best = -1, bestDist = 1 << 30;
	for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
		Cell c = target(a);
		if (env.blocked(c.first, c.second)) continue;
		// A* ranks by steps over free cells; SHORTCUT by steps along the current cycle, the classic jump-ahead rule
		int dd = mode == ASTAR ? dist[c.second * W + c.first] : ((rf - order[c.second * W + c.first]) % 400 + 400) % 400;
		if (dd < 0 || dd >= bestDist) continue;
		// Body after the move: the tail stays only when this move eats
		std::vector<Cell> next{c};
		next.insert(next.end(), body.begin(), body.end() - (c.first == food.getX() && c.second == food.getY() ? 0 : 1));
		Tree t;
		if (treeFor(next, t)) {
			best = a;
			bestDist = dd;
		}
	}
	return best >= 0 ? best : cycleStep();
}

std::vector<float> SafeAgent::features(const SnakeEnv& env) {
	const int W = 20;
	std::vector<Cell> body = bodyCells(env);
	const Cell head = body[0], tail = body.back();
	const Food& food = env.state().getFood();
	if (!haveTree_ || !consistent(tree_, body)) {
		haveTree_ = treeFor(body, tree_);
	}
	std::vector<int> order = cycleOrder(tree_, head, body[1]);

	std::vector<int> dist(W * W, -1);
	std::queue<Cell> q;
	dist[food.getY() * W + food.getX()] = 0;
	q.push({food.getX(), food.getY()});
	while (!q.empty()) {
		Cell c = q.front();
		q.pop();
		for (int d = 0; d < 4; ++d) {
			int nx = c.first + SnakeEnv::DX[d], ny = c.second + SnakeEnv::DY[d];
			if (env.blocked(nx, ny) || dist[ny * W + nx] >= 0) continue;
			dist[ny * W + nx] = dist[c.second * W + c.first] + 1;
			q.push({nx, ny});
		}
	}
	int safe = 0;
	for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
		int d = SnakeEnv::turn(env.dir(), a);
		Cell c{head.first + SnakeEnv::DX[d], head.second + SnakeEnv::DY[d]};
		if (env.blocked(c.first, c.second)) continue;
		std::vector<Cell> next{c};
		next.insert(next.end(), body.begin(), body.end() - (c.first == food.getX() && c.second == food.getY() ? 0 : 1));
		Tree t;
		safe += treeFor(next, t);
	}
	const int rf = order[food.getY() * W + food.getX()], dt = order[tail.second * W + tail.first];
	const int path = dist[head.second * W + head.first];
	return {static_cast<float>(body.size()) / 400.0f,
		static_cast<float>(std::abs(head.first - food.getX()) + std::abs(head.second - food.getY())) / 38.0f,
		path < 0 ? 1.0f : static_cast<float>(path) / 400.0f,
		static_cast<float>(rf) / 400.0f,
		static_cast<float>(dt) / 400.0f,
		std::min(1.0f, static_cast<float>(env.sinceFood()) / 250.0f),
		rf < dt ? 1.0f : 0.0f,
		static_cast<float>(safe) / 3.0f};
}
