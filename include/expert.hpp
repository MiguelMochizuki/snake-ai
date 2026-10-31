/**
 * expert.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Scripted Snake player (BFS to the food) used as the imitation teacher
 */
#ifndef EXPERT_HPP
#define EXPERT_HPP

#include "snake_env.hpp"

/**
 * Picks the next move: the first step of a shortest safe path to the food; with no path, the safe move with the most reachable cells
 *
 * Parameters:
 * const SnakeEnv& env: Current game
 *
 * Returns int: STRAIGHT, TURN_RIGHT or TURN_LEFT
 */
int expertAction(const SnakeEnv& env);

#endif
