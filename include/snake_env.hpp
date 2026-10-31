/**
 * snake_env.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Headless Snake environment for training agents, built on GameState
 */
#ifndef SNAKE_ENV_HPP
#define SNAKE_ENV_HPP

#include "game_state.hpp"
#include <vector>

/**
 * Relative actions: the net never reverses, it only goes straight or turns
 */
enum Action { STRAIGHT = 0, TURN_RIGHT = 1, TURN_LEFT = 2 };

/**
 * One game on a 20x20 board
 *
 * Attributes:
 * GameState state_: The game logic
 * int dir_: Current heading (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, as GameState::moveSnake)
 * bool dead_: true after a collision or starvation
 * int sinceFood_: Steps since the last food
 */
class SnakeEnv {
	GameState state_;
	int dir_ = 3;
	bool dead_ = false;
	int sinceFood_ = 0;

public:
	static const int FEATURES = 11;
	static const int ACTIONS = 3;

	/**
	 * Parameters:
	 * unsigned seed: Seed for food placement
	 */
	explicit SnakeEnv(unsigned seed);

	/**
	 * Returns FEATURES floats: danger straight/right/left, heading one-hot (up, down, left, right), food up/down/left/right of the head
	 */
	std::vector<float> features() const;

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
	 * Moves one step
	 *
	 * Parameters:
	 * int action: STRAIGHT, TURN_RIGHT or TURN_LEFT
	 */
	void step(int action);

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

	static const int DX[4];
	static const int DY[4];

	bool dead() const { return dead_; }
	int score() const { return state_.getScore(); }
	int dir() const { return dir_; }
	const GameState& state() const { return state_; }
};

#endif
