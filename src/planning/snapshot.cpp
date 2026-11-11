/**
 * snapshot.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Snapshot construction and its blocked test
 */
#include "snake_ai/planning/snapshot.hpp"
#include "snake_ai/planning/modes.hpp"

bool Snapshot::blocked(int x, int y) const {
	if (x < 0 || x >= size || y < 0 || y >= size) return true;
	// The tail leaves its cell as the head arrives, so it is not an obstacle (the game's own rule)
	for (size_t i = 0; i + 1 < body.size(); ++i) {
		if (body[i].first == x && body[i].second == y) return true;
	}
	return false;
}

Snapshot snapshotOf(const SnakeEnv& env) {
	const Food& food = env.state().getFood();
	return {env.size(), env.dir(), bodyCells(env), {food.getX(), food.getY()}, env.sinceFood()};
}
