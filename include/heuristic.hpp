/**
 * heuristic.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Rotation- and scale-invariant move heuristic with weights that depend on snake length
 */
#ifndef HEURISTIC_HPP
#define HEURISTIC_HPP

#include "snake_env.hpp"
#include <array>
#include <string>

/**
 * Weights for ego features 1..6 (closer, room, tail reachable, loops, open, coil): the weight is base + slope * length fraction,
 * so a long snake can value survival over food while a short one chases food
 *
 * Attributes:
 * std::array<float, 12> w: base[6] then slope[6]
 */
struct Genome {
	std::array<float, 12> w{};
};

/**
 * Scores a move
 *
 * Parameters:
 * const Genome& g: Weights
 * const std::array<float, SnakeEnv::EGO_PER_ACTION>& f: SnakeEnv::ego() of the move
 * float length: Snake length over the board
 *
 * Returns float: Higher is better; blocked moves get a very low score
 */
float heuristicScore(const Genome& g, const std::array<float, SnakeEnv::EGO_PER_ACTION>& f, float length);

/**
 * Picks the best-scoring move
 *
 * Parameters:
 * const Genome& g: Weights
 * const SnakeEnv& env: Current game
 *
 * Returns int: STRAIGHT, TURN_RIGHT or TURN_LEFT
 */
int heuristicAction(const Genome& g, const SnakeEnv& env);

/**
 * Parameters:
 * const std::string& path: File to write
 * const Genome& g: Weights, as text
 */
void saveGenome(const std::string& path, const Genome& g);

/**
 * Parameters:
 * const std::string& path: File written by saveGenome
 *
 * Returns Genome: The weights; throws std::runtime_error if the file is missing or short
 */
Genome loadGenome(const std::string& path);

#endif
