/**
 * snapshot.hpp
 * Author: Miguel Mochizuki Silva
 * Description: A plain copy of the facts a safe player decides from, so rollouts can use it without a SnakeEnv
 */
#ifndef SNAPSHOT_HPP
#define SNAPSHOT_HPP

#include "snake_ai/env/snake_env.hpp"
#include "snake_ai/planning/tiles.hpp"

#include <vector>

/**
 * Everything SafeAgent needs to choose a move. It carries no random generator, so a decision made from it cannot depend on the real game's future food
 *
 * Attributes:
 * int size: Board side in cells
 * int dir: Heading (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, as SnakeEnv)
 * std::vector<Cell> body: Body cells from head to tail
 * Cell food: The food cell
 * int sinceFood: Steps since the last food
 */
struct Snapshot {
	int size;
	int dir;
	std::vector<Cell> body;
	Cell food;
	int sinceFood;

	/** Steps without food after which a player falls back to the cycle: 5/8 cells, as SnakeEnv::loopGuard */
	int loopGuard() const { return size * size * 5 / 8; }
	/** Steps without food after which a game counts as lost: 1.5 cells per cell, as SnakeEnv::starveLimit */
	int starveLimit() const { return size * size * 3 / 2; }

	/**
	 * Tells whether a cell would kill the snake, like SnakeEnv::blocked: off the board or on the body, with the tail and the food not counting
	 * (linear in the body length; SafeAgent builds an occupancy grid instead of calling this in loops)
	 *
	 * Parameters:
	 * int x: Column
	 * int y: Row
	 *
	 * Returns bool: true if moving there is fatal
	 */
	bool blocked(int x, int y) const;
};

/**
 * Parameters:
 * const SnakeEnv& env: Current game
 *
 * Returns Snapshot: The game's facts as plain data
 */
Snapshot snapshotOf(const SnakeEnv& env);

#endif
