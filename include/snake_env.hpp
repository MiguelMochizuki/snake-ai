/**
 * snake_env.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Headless Snake environment for training agents, built on GameState
 */
#ifndef SNAKE_ENV_HPP
#define SNAKE_ENV_HPP

#include "cycle.hpp"
#include "game_state.hpp"
#include <array>
#include <deque>
#include <vector>

/**
 * Relative actions: the agent never reverses, it only goes straight or turns
 */
enum Action { STRAIGHT = 0, TURN_RIGHT = 1, TURN_LEFT = 2 };

/**
 * What one move would do, in terms of the board and the Hamiltonian cycle
 *
 * Attributes:
 * bool blocked: Wall or body
 * bool allowed: Not blocked and either the next cycle cell or a shortcut that stays before the tail, which keeps the body in cycle order
 * int ahead: Cycle steps from the head to the target cell
 * int remaining: Cycle steps from the target cell to the food
 */
struct MoveInfo {
	bool blocked;
	bool allowed;
	int ahead;
	int remaining;
};

/**
 * One game on a 20x20 board
 *
 * Attributes:
 * GameState state_: The game logic
 * int dir_: Current heading (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, as GameState::moveSnake)
 * bool dead_: true after a collision, starvation or a full board
 * bool won_: true if the last food filled the board
 * int sinceFood_: Steps since the last food
 */
class SnakeEnv {
	GameState state_;
	int dir_ = 3;
	bool dead_ = false;
	bool won_ = false;
	int sinceFood_ = 0;
	std::deque<int> recent_;
	std::array<int, 400> visits_{};

public:
	static const int EGO_PER_ACTION = 7;
	static const int EGO_FEATURES = 3 * EGO_PER_ACTION + 1;
	static const int LOOP_WINDOW = 60;  // steps of head history used to spot loops
	static const int FEATURES = 13;
	static const int ACTIONS = 3;
	static const int DX[4];
	static const int DY[4];

	/**
	 * Parameters:
	 * unsigned seed: Seed for food placement
	 */
	explicit SnakeEnv(unsigned seed);

	/**
	 * Returns FEATURES floats: per action (straight, right, left) blocked, allowed and remaining cycle distance to the food over CYCLE_LEN; then the heading one-hot (up, down, left, right)
	 */
	std::vector<float> features() const;

	/**
	 * Features that do not change under rotation or board scale: everything is relative to the head's heading or a fraction of the board.
	 * Per action: blocked, closer to the food (+1/-1), reachable free cells over free cells, tail reachable, recent visits of the target cell over 4, free neighbours of the target over 3, coil (see coil()). Then the snake length over the board
	 *
	 * Parameters:
	 * int action: STRAIGHT, TURN_RIGHT or TURN_LEFT
	 *
	 * Returns std::array<float, EGO_PER_ACTION>: The action's features
	 */
	std::array<float, EGO_PER_ACTION> ego(int action) const;

	/**
	 * Estimates how much of the board the snake encircles: closes the body into a curve (target cell, head ... tail, back to the target)
	 * and counts the points (cell corners) around which the curve winds exactly once
	 *
	 * Parameters:
	 * int nx: Column of the cell the head would move to
	 * int ny: Row of that cell
	 *
	 * Returns float: Points with winding number of absolute value 1, over all points (0..1)
	 */
	float coil(int nx, int ny) const;

	/**
	 * Returns EGO_FEATURES floats: ego() of the three actions, then the length fraction
	 */
	std::vector<float> egoFeatures() const;

	/**
	 * Tells whether a cell would kill the snake (wall or body, tail excluded as the game does)
	 *
	 * Parameters:
	 * int x: Column
	 * int y: Row
	 *
	 * Returns bool: true if moving there is fatal
	 */
	bool blocked(int x, int y) const;

	/**
	 * Describes a move
	 *
	 * Parameters:
	 * int action: STRAIGHT, TURN_RIGHT or TURN_LEFT
	 *
	 * Returns MoveInfo: The move's board and cycle facts
	 */
	MoveInfo info(int action) const;

	/**
	 * Moves one step
	 *
	 * Parameters:
	 * int action: STRAIGHT, TURN_RIGHT or TURN_LEFT
	 *
	 * Returns bool: true if the move ate food
	 */
	bool step(int action);

	/**
	 * Turns a heading by an action
	 *
	 * Parameters:
	 * int dir: Heading
	 * int action: Relative action
	 *
	 * Returns int: The new absolute heading
	 */
	static int turn(int dir, int action);

	bool dead() const { return dead_; }
	int sinceFood() const { return sinceFood_; }
	int score() const { return state_.getScore() + (won_ ? 1 : 0); }
	int dir() const { return dir_; }
	int headVisits() const { return visits_[state_.getSnake().getHead()->getY() * state_.getCols() + state_.getSnake().getHead()->getX()]; }
	float lengthFraction() const { return static_cast<float>(state_.getSnake().getLength()) / CYCLE_LEN; }
	const GameState& state() const { return state_; }
};

#endif
