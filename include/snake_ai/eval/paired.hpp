/**
 * paired.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Plays games on fixed seeds and compares two players on the same seeds with a bootstrap interval
 */
#ifndef PAIRED_HPP
#define PAIRED_HPP

#include "snake_ai/env/snake_env.hpp"
#include "snake_ai/planning/modes.hpp"

#include <functional>
#include <vector>

/**
 * Outcome of one game
 *
 * Attributes:
 * int steps: Steps played (the game stops at 200 per cell)
 * int score: Final score
 * bool full: true if the board was filled
 * bool died: true if the game ended dead without filling the board
 */
struct GameResult {
	int steps;
	int score;
	bool full;
	bool died;
};

/**
 * A mean and its 95% bootstrap interval
 *
 * Attributes:
 * double mean: The statistic on the full data
 * double lo: 2.5th percentile of the resampled statistic
 * double hi: 97.5th percentile of the resampled statistic
 */
struct Interval {
	double mean;
	double lo;
	double hi;
};

/**
 * Plays one game per seed, in seed order
 *
 * Parameters:
 * int size: Board side
 * const std::vector<unsigned>& seeds: Food seeds
 * const std::function<int(SnakeEnv&, SafeAgent&)>& act: Chooses the action; called from several threads when threads > 1, so it must not share mutable state
 * int threads: Worker threads; the results do not depend on it
 *
 * Returns std::vector<GameResult>: One result per seed
 */
std::vector<GameResult> playGames(int size, const std::vector<unsigned>& seeds, const std::function<int(SnakeEnv&, SafeAgent&)>& act, int threads);

/**
 * Relative difference of means on paired data, (mean(test) - mean(base)) / mean(base), with a bootstrap interval that resamples pair indices
 *
 * Parameters:
 * const std::vector<double>& base: Values of the baseline, one per seed
 * const std::vector<double>& test: Values of the player under test, same seeds, same length
 * int iters: Number of resamples
 * unsigned seed: Seed of the resampling generator
 *
 * Returns Interval: The relative difference and its 95% interval (negative is fewer steps)
 */
Interval bootstrapRelDiff(const std::vector<double>& base, const std::vector<double>& test, int iters, unsigned seed);

/**
 * The gate of the design: at least 3% fewer steps with the whole interval below zero
 *
 * Parameters:
 * const Interval& iv: Result of bootstrapRelDiff on steps
 *
 * Returns bool: true if the gate is passed
 */
bool passesGate(const Interval& iv);

#endif
