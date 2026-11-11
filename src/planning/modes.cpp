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
	const Snapshot& snap;
	std::vector<char> occ;
	int side;
	int cells;
	std::vector<Cell> body;
	Cell head;
	std::vector<int> order;

	/**
	 * Parameters:
	 * const Snapshot& s: Current game, which must outlive the View
	 * Tree& tree: The agent's spanning tree; recomputed here if it no longer fits the body
	 * bool& haveTree: Whether tree holds a valid tree
	 */
	View(const Snapshot& s, Tree& tree, bool& haveTree) : snap(s), occ(s.size * s.size, 0), side(s.size), cells(side * side), body(s.body), head(body[0]) {
		// Occupancy grid so blocked() is constant time inside the BFS; the tail is left out as the game does
		for (size_t i = 0; i + 1 < body.size(); ++i) occ[body[i].second * side + body[i].first] = 1;
		if (!haveTree || !consistent(tree, side, body)) {
			haveTree = treeFor(body, side, tree);
		}
		order = cycleOrder(tree, side, head, body[1]);
	}

	/**
	 * Parameters:
	 * int x: Column
	 * int y: Row
	 *
	 * Returns bool: true if moving there is fatal (same rule as Snapshot::blocked)
	 */
	bool blocked(int x, int y) const { return x < 0 || x >= side || y < 0 || y >= side || occ[y * side + x]; }

	/**
	 * Parameters:
	 * int action: STRAIGHT, TURN_RIGHT or TURN_LEFT
	 *
	 * Returns Cell: The cell the action leads to (may lie off the board)
	 */
	Cell target(int action) const {
		int d = SnakeEnv::turn(snap.dir, action);
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
		const Cell food = snap.food;
		std::vector<Cell> next{c};
		next.insert(next.end(), body.begin(), body.end() - (c.first == food.first && c.second == food.second ? 0 : 1));
		Tree tree;
		return treeFor(next, side, tree);
	}

	/**
	 * Returns std::vector<int>: Shortest path length to the food over free cells, per cell (-1 if unreachable)
	 */
	std::vector<int> distToFood() const {
		const Cell food = snap.food;
		std::vector<int> dist(cells, -1);
		std::queue<Cell> q;
		dist[food.second * side + food.first] = 0;
		q.push({food.first, food.second});
		while (!q.empty()) {
			Cell c = q.front();
			q.pop();
			for (int d = 0; d < 4; ++d) {
				int nx = c.first + SnakeEnv::DX[d], ny = c.second + SnakeEnv::DY[d];
				if (blocked(nx, ny) || dist[ny * side + nx] >= 0) continue;
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

int SafeAgent::act(const Snapshot& s, Mode mode) {
	const View v(s, tree_, haveTree_);
	if (mode == CYCLE) {
		return v.cycleStep();
	}

	// ASTAR and SHORTCUT: rank the moves that keep the body on a Hamiltonian cycle
	const std::vector<int> dist = v.distToFood();
	const Cell food = s.food;
	const int foodPos = v.cyclePos({food.first, food.second});
	int best = -1, bestCost = 1 << 30;
	for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
		Cell c = v.target(a);
		if (v.blocked(c.first, c.second)) continue;
		// A* ranks by steps over free cells; SHORTCUT by steps along the current cycle, the classic jump-ahead rule
		int cost = mode == ASTAR ? dist[c.second * v.side + c.first]
			: ((foodPos - v.cyclePos(c)) % v.cells + v.cells) % v.cells;
		if (cost < 0 || cost >= bestCost || !v.safe(c)) continue;
		best = a;
		bestCost = cost;
	}
	return best >= 0 ? best : v.cycleStep();
}

std::vector<float> SafeAgent::features(const Snapshot& s) {
	const View v(s, tree_, haveTree_);
	const Cell tail = v.body.back();
	const Cell food = s.food;
	int safeMoves = 0;
	for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
		Cell c = v.target(a);
		if (!v.blocked(c.first, c.second)) safeMoves += v.safe(c);
	}
	const int foodPos = v.cyclePos({food.first, food.second}), tailPos = v.cyclePos(tail);
	const int path = v.distToFood()[v.head.second * v.side + v.head.first];
	const float cells = static_cast<float>(v.cells);
	return {static_cast<float>(v.body.size()) / cells,
		static_cast<float>(std::abs(v.head.first - food.first) + std::abs(v.head.second - food.second)) / static_cast<float>(2 * (v.side - 1)),
		path < 0 ? 1.0f : static_cast<float>(path) / cells,
		static_cast<float>(foodPos) / cells,
		static_cast<float>(tailPos) / cells,
		std::min(1.0f, static_cast<float>(s.sinceFood) / static_cast<float>(s.loopGuard())),
		foodPos < tailPos ? 1.0f : 0.0f,
		static_cast<float>(safeMoves) / 3.0f};
}

std::vector<float> SafeAgent::actionFeatures(const Snapshot& s) {
	const View v(s, tree_, haveTree_);
	const std::vector<int> dist = v.distToFood();
	struct Option { bool blocked, safe, cycleStep; int path; };
	std::vector<Option> options;
	int bestSafe = 1 << 30;
	for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
		Cell c = v.target(a);
		if (v.blocked(c.first, c.second)) {
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

int SafeAgent::act(const SnakeEnv& env, Mode mode) {
	return act(snapshotOf(env), mode);
}

std::vector<float> SafeAgent::features(const SnakeEnv& env) {
	return features(snapshotOf(env));
}

std::vector<float> SafeAgent::actionFeatures(const SnakeEnv& env) {
	return actionFeatures(snapshotOf(env));
}

int teacherAction(const Snapshot& s, SafeAgent& agent) {
	return agent.act(s, s.sinceFood > s.loopGuard() ? CYCLE : ASTAR);
}
