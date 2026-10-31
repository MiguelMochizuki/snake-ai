/**
 * expert.cpp
 * Author: Miguel Mochizuki Silva
 * Description: BFS teacher implementation
 */
#include "expert.hpp"
#include <queue>

namespace {

/**
 * Breadth-first search from a cell
 *
 * Parameters:
 * const SnakeEnv& env: Current game
 * int sx: Start column
 * int sy: Start row
 * int gx: Goal column, or -1 to only count reachable cells
 * int gy: Goal row
 * int* reached: Receives the number of cells reached
 *
 * Returns int: Heading of the first step toward the goal, or -1 if unreachable
 */
int bfs(const SnakeEnv& env, int sx, int sy, int gx, int gy, int* reached) {
	const int W = env.state().getCols(), H = env.state().getRows();
	std::vector<int> first(W * H, -2);  // -2 unseen, else first heading
	std::queue<std::pair<int, int>> q;
	first[sy * W + sx] = -1;
	q.push({sx, sy});
	*reached = 0;
	while (!q.empty()) {
		auto [x, y] = q.front();
		q.pop();
		++*reached;
		if (x == gx && y == gy) {
			return first[y * W + x];
		}
		for (int d = 0; d < 4; ++d) {
			int nx = x + SnakeEnv::DX[d], ny = y + SnakeEnv::DY[d];
			if (env.blocked(nx, ny) || first[ny * W + nx] != -2) {
				continue;
			}
			first[ny * W + nx] = (x == sx && y == sy) ? d : first[y * W + x];
			q.push({nx, ny});
		}
	}
	return -1;
}

}  // namespace

int expertAction(const SnakeEnv& env) {
	const SnakeSegment* head = env.state().getSnake().getHead();
	const int hx = head->getX(), hy = head->getY();
	const Food& food = env.state().getFood();
	int unused;
	int heading = bfs(env, hx, hy, food.getX(), food.getY(), &unused);
	for (int a : {STRAIGHT, TURN_RIGHT, TURN_LEFT}) {
		if (heading >= 0 && SnakeEnv::turn(env.dir(), a) == heading) {
			return a;
		}
	}
	// No path: survive by heading where there is most room
	int best = STRAIGHT, bestRoom = -1;
	for (int a : {STRAIGHT, TURN_RIGHT, TURN_LEFT}) {
		int d = SnakeEnv::turn(env.dir(), a);
		int nx = hx + SnakeEnv::DX[d], ny = hy + SnakeEnv::DY[d];
		if (env.blocked(nx, ny)) {
			continue;
		}
		int room;
		bfs(env, nx, ny, -1, -1, &room);
		if (room > bestRoom) {
			bestRoom = room;
			best = a;
		}
	}
	return best;
}
