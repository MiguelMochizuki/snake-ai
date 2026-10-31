/**
 * expert.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Scripted Snake player (Hamiltonian cycle with safe shortcuts) used as the imitation teacher
 */
#ifndef EXPERT_HPP
#define EXPERT_HPP

#include "snake_ai/env/snake_env.hpp"

/**
 * Picks the allowed move that lands closest, along the cycle, to the food; with none allowed, the next cycle cell.
 * Moves only ever jump forward and stay before the tail, so the body stays in cycle order and the snake never traps itself
 *
 * Parameters:
 * const SnakeEnv& env: Current game
 *
 * Returns int: STRAIGHT, TURN_RIGHT or TURN_LEFT
 */
int expertAction(const SnakeEnv& env);

#endif
