/**
 * sim.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Rollout simulator with the rules of SnakeEnv but its own food generator, so a lookahead never sees the real game's future
 */
#ifndef SIM_HPP
#define SIM_HPP

#include "snake_ai/planning/snapshot.hpp"

#include <deque>
#include <random>
#include <vector>

/**
 * One simulated game from a Snapshot. Same rules as SnakeEnv::step; every food after the one in the snapshot is drawn
 * uniformly from the free cells by the Sim's own generator
 *
 * Attributes:
 * int size_: Board side
 * int dir_: Heading
 * std::deque<Cell> body_: Body from head to tail
 * Cell food_: Current food
 * int sinceFood_: Steps since the last food
 * std::vector<char> occ_: Occupancy by cell (y * size_ + x), body only
 * bool dead_: true after a collision, starvation or a full board
 * bool won_: true if the last food filled the board
 * int eaten_: Foods eaten since the snapshot
 * int steps_: Steps taken since the snapshot
 * std::mt19937 rng_: Food generator, seeded by the caller
 */
class Sim {
	int size_;
	int dir_;
	std::deque<Cell> body_;
	Cell food_;
	int sinceFood_;
	std::vector<char> occ_;
	bool dead_ = false;
	bool won_ = false;
	int eaten_ = 0;
	int steps_ = 0;
	std::mt19937 rng_;

	/** Draws a new food on a uniformly random free cell; the caller guarantees one exists */
	void spawnFood();

public:
	/**
	 * Parameters:
	 * const Snapshot& s: Starting state; its food is kept as the first food
	 * unsigned seed: Seed of the generator that places every later food
	 */
	Sim(const Snapshot& s, unsigned seed);

	/**
	 * Moves one step
	 *
	 * Parameters:
	 * int action: STRAIGHT, TURN_RIGHT or TURN_LEFT
	 *
	 * Returns bool: true if the move ate food (also when it filled the board)
	 */
	bool step(int action);

	/**
	 * Returns Snapshot: The current state as plain data
	 */
	Snapshot snapshot() const;

	bool dead() const { return dead_; }
	bool won() const { return won_; }
	int eaten() const { return eaten_; }
	int steps() const { return steps_; }
};

#endif
