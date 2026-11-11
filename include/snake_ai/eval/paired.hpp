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
#include <string>
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

/**
 * Strict integer parsing for command lines: the whole text must be a decimal integer (no spaces, no trailing characters, no overflow)
 *
 * Parameters:
 * const char* text: The argument (may be null)
 * int minValue: Smallest accepted value
 * int& out: Receives the value; left untouched when the text is rejected
 *
 * Returns bool: true if text is a valid integer >= minValue
 */
bool parseInt(const char* text, int minValue, int& out);

/**
 * Parameters:
 * const char* text: A command-line mode (may be null)
 *
 * Returns bool: true for "validate", "test" or "smoke"
 */
bool isRunMode(const char* text);

/**
 * Food seeds of a mode: validate 20000..20019, test 100000..100029 (disjoint, the test set is touched once per arm), smoke 1000..1001 (a quick run that exercises the same code)
 *
 * Parameters:
 * const std::string& mode: "validate", "test" or "smoke"
 *
 * Returns std::vector<unsigned>: The seeds; throws std::invalid_argument for another mode
 */
std::vector<unsigned> seedsFor(const std::string& mode);

/**
 * Parameters:
 * const std::string& mode: "validate", "test" or "smoke"
 *
 * Returns int: Board side: 20, or 10 for smoke; throws std::invalid_argument for another mode
 */
int boardFor(const std::string& mode);

#endif
