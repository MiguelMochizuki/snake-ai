/**
 * modes.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Safe playing modes
 */
#include "snake_ai/planning/modes.hpp"
#include <algorithm>
#include <cstdlib>
#include <queue>

namespace {

/**
 * Everything the modes and the feature functions need to know about one state, computed once
 *
 * Attributes:
 * const SnakeEnv& env: The game
 * int side: Board side in cells
 * int cells: Cells on the board
 * std::vector<Cell> body: Body from head to tail
 * Cell head: Head cell
 * std::vector<int> order: Steps from the head along the cycle, per cell (y * side + x)
 */
struct View {
	const SnakeEnv& env;
	int side;
	int cells;
	std::vector<Cell> body;
	Cell head;
	std::vector<int> order;

	/**
	 * Parameters:
	 * const SnakeEnv& env: Current game
	 * Tree& tree: The agent's spanning tree; recomputed here if it no longer fits the body
	 * bool& haveTree: Whether tree holds a valid tree
	 */
	View(const SnakeEnv& e, Tree& tree, bool& haveTree) : env(e), side(e.size()), cells(side * side), body(bodyCells(e)), head(body[0]) {
		if (!haveTree || !consistent(tree, side, body)) {
			haveTree = treeFor(body, side, tree);
		}
		order = cycleOrder(tree, side, head, body[1]);
	}

	/**
	 * Parameters:
	 * int action: STRAIGHT, TURN_RIGHT or TURN_LEFT
	 *
	 * Returns Cell: The cell the action leads to (may lie off the board)
	 */
	Cell target(int action) const {
		int d = SnakeEnv::turn(env.dir(), action);
		return {head.first + SnakeEnv::DX[d], head.second + SnakeEnv::DY[d]};
	}

	/**
	 * Parameters:
	 * Cell c: A cell on the board
	 *
	 * Returns int: Steps from the head along the cycle
	 */
	int cyclePos(Cell c) const { return order[c.second * side + c.first]; }

	/**
	 * Returns int: The action that goes to the next cycle cell (STRAIGHT if none is found)
	 */
	int cycleStep() const {
		for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
			Cell c = target(a);
			if (c.first >= 0 && c.first < side && c.second >= 0 && c.second < side && cyclePos(c) == 1) return a;
		}
		return STRAIGHT;
	}

	/**
	 * Parameters:
	 * Cell c: A free cell the head would move to
	 *
	 * Returns bool: true if the body after that move still lies on a Hamiltonian cycle; the tail stays only when the move eats
	 */
	bool safe(Cell c) const {
		const Food& food = env.state().getFood();
		std::vector<Cell> next{c};
		next.insert(next.end(), body.begin(), body.end() - (c.first == food.getX() && c.second == food.getY() ? 0 : 1));
		Tree tree;
		return treeFor(next, side, tree);
	}

	/**
	 * Returns std::vector<int>: Shortest path length to the food over free cells, per cell (-1 if unreachable)
	 */
	std::vector<int> distToFood() const {
		const Food& food = env.state().getFood();
		std::vector<int> dist(cells, -1);
		std::queue<Cell> q;
		dist[food.getY() * side + food.getX()] = 0;
		q.push({food.getX(), food.getY()});
		while (!q.empty()) {
			Cell c = q.front();
			q.pop();
			for (int d = 0; d < 4; ++d) {
				int nx = c.first + SnakeEnv::DX[d], ny = c.second + SnakeEnv::DY[d];
				if (env.blocked(nx, ny) || dist[ny * side + nx] >= 0) continue;
				dist[ny * side + nx] = dist[c.second * side + c.first] + 1;
				q.push({nx, ny});
			}
		}
		return dist;
	}
};

}  // namespace

std::vector<Cell> bodyCells(const SnakeEnv& env) {
	std::vector<Cell> body;
	for (const SnakeSegment& s : env.state().getSnake().getSegments()) {
		body.push_back({s.getX(), s.getY()});
	}
	return body;
}

int SafeAgent::act(const SnakeEnv& env, Mode mode) {
	const View v(env, tree_, haveTree_);
	if (mode == CYCLE) {
		return v.cycleStep();
	}

	// ASTAR and SHORTCUT: rank the moves that keep the body on a Hamiltonian cycle
	const std::vector<int> dist = v.distToFood();
	const Food& food = env.state().getFood();
	const int foodPos = v.cyclePos({food.getX(), food.getY()});
	int best = -1, bestCost = 1 << 30;
	for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
		Cell c = v.target(a);
		if (env.blocked(c.first, c.second)) continue;
		// A* ranks by steps over free cells; SHORTCUT by steps along the current cycle, the classic jump-ahead rule
		int cost = mode == ASTAR ? dist[c.second * v.side + c.first]
			: ((foodPos - v.cyclePos(c)) % v.cells + v.cells) % v.cells;
		if (cost < 0 || cost >= bestCost || !v.safe(c)) continue;
		best = a;
		bestCost = cost;
	}
	return best >= 0 ? best : v.cycleStep();
}

std::vector<float> SafeAgent::features(const SnakeEnv& env) {
	const View v(env, tree_, haveTree_);
	const Cell tail = v.body.back();
	const Food& food = env.state().getFood();
	int safeMoves = 0;
	for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
		Cell c = v.target(a);
		if (!env.blocked(c.first, c.second)) safeMoves += v.safe(c);
	}
	const int foodPos = v.cyclePos({food.getX(), food.getY()}), tailPos = v.cyclePos(tail);
	const int path = v.distToFood()[v.head.second * v.side + v.head.first];
	const float cells = static_cast<float>(v.cells);
	return {static_cast<float>(v.body.size()) / cells,
		static_cast<float>(std::abs(v.head.first - food.getX()) + std::abs(v.head.second - food.getY())) / static_cast<float>(2 * (v.side - 1)),
		path < 0 ? 1.0f : static_cast<float>(path) / cells,
		static_cast<float>(foodPos) / cells,
		static_cast<float>(tailPos) / cells,
		std::min(1.0f, static_cast<float>(env.sinceFood()) / static_cast<float>(env.loopGuard())),
		foodPos < tailPos ? 1.0f : 0.0f,
		static_cast<float>(safeMoves) / 3.0f};
}

std::vector<float> SafeAgent::actionFeatures(const SnakeEnv& env) {
	const View v(env, tree_, haveTree_);
	const std::vector<int> dist = v.distToFood();
	struct Option { bool blocked, safe, cycleStep; int path; };
	std::vector<Option> options;
	int bestSafe = 1 << 30;
	for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
		Cell c = v.target(a);
		if (env.blocked(c.first, c.second)) {
			options.push_back({true, false, false, -1});
			continue;
		}
		Option o{false, v.safe(c), v.cyclePos(c) == 1, dist[c.second * v.side + c.first]};
		if (o.safe && o.path >= 0) bestSafe = std::min(bestSafe, o.path);
		options.push_back(o);
	}
	std::vector<float> f;
	for (const Option& o : options) {
		// Steps worse than the best safe move, capped at 4 and scaled to 0..1, so the choice is visible to the network
		float regret = o.blocked || o.path < 0 ? 1.0f : static_cast<float>(std::min(o.path - bestSafe, 4)) / 4.0f;
		f.insert(f.end(), {o.blocked ? 1.0f : 0.0f, o.safe ? 1.0f : 0.0f, std::max(regret, 0.0f), o.cycleStep ? 1.0f : 0.0f});
	}
	return f;
}
