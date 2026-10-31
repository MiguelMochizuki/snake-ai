/**
 * modes.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Three ways to play (A*, Hamiltonian shortcuts, Hamiltonian cycle) that all keep the body on a tile-tree Hamiltonian cycle
 */
#ifndef MODES_HPP
#define MODES_HPP

#include "snake_env.hpp"
#include "tiles.hpp"

/**
 * Playing modes. Every move any mode makes leaves the body on some Hamiltonian cycle, so the snake can always follow that cycle
 * back to its tail, and the modes can be switched at any step
 */
enum Mode { ASTAR = 0, SHORTCUT = 1, CYCLE = 2, MODES = 3 };

/**
 * Body cells from head to tail
 *
 * Parameters:
 * const SnakeEnv& env: Current game
 *
 * Returns std::vector<Cell>: The cells
 */
std::vector<Cell> bodyCells(const SnakeEnv& env);

/**
 * Chooses a move with a mode; keeps the spanning tree between calls so Hamiltonian modes follow one stable cycle
 */
class SafeAgent {
	Tree tree_{};
	bool haveTree_ = false;

public:
	/**
	 * Parameters:
	 * const SnakeEnv& env: Current game
	 * Mode mode: ASTAR takes the shortest path step among moves that keep the body on a Hamiltonian cycle;
	 * SHORTCUT takes, among those moves, the one that gets closest to the food along the current cycle; CYCLE follows the cycle
	 *
	 * Returns int: STRAIGHT, TURN_RIGHT or TURN_LEFT
	 */
	int act(const SnakeEnv& env, Mode mode);

	/**
	 * State summary for choosing a mode, all fractions of the board or of a bound so it does not depend on scale
	 *
	 * Parameters:
	 * const SnakeEnv& env: Current game
	 *
	 * Returns std::vector<float>: FEATURES values: snake length, Manhattan distance to the food, shortest free path to the food (1 if none),
	 * cycle distance to the food, cycle distance to the tail, steps since food, food before the tail along the cycle, safe moves out of three
	 */
	std::vector<float> features(const SnakeEnv& env);

	static const int FEATURES = 8;
	static const int ACTION_FEATURES = 4 * SnakeEnv::ACTIONS;

	/**
	 * Per-action features, relative to the heading so they do not depend on rotation: for each of straight, right and left
	 * blocked, safe (the body stays on a Hamiltonian cycle after the move), how many steps longer its free path to the food is than the best
	 * safe move's, capped at 4 and over 4 (1 when blocked or no path), and whether the move is the next cell of the cycle
	 *
	 * Parameters:
	 * const SnakeEnv& env: Current game
	 *
	 * Returns std::vector<float>: ACTION_FEATURES values
	 */
	std::vector<float> actionFeatures(const SnakeEnv& env);
};

#endif
