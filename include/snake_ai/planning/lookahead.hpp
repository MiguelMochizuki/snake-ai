/**
 * lookahead.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Chooses among the safe moves by simulating the A* player for the next few foods (honest and leaked-future variants), and a time-aware A* baseline
 */
#ifndef LOOKAHEAD_HPP
#define LOOKAHEAD_HPP

#include "snake_ai/planning/modes.hpp"
#include "snake_ai/planning/sim.hpp"

/**
 * Parameters of the honest lookahead
 *
 * Attributes:
 * int foods: Foods each rollout must eat (K)
 * int samples: Sampled futures per move (M)
 * unsigned seed: Base seed of the sampled futures; the same samples are used for every move
 */
struct LookaheadConfig {
	int foods = 3;
	int samples = 8;
	unsigned seed = 1;
};

/**
 * Picks a move from the Snapshot only. Each safe move is scored by the mean, over sampled futures, of the steps the baseline player
 * needs to eat the next foods (a rollout that dies or starves counts the starvation limit for each food it did not eat);
 * the lowest score wins and ties go to the baseline's own move. Falls back to the baseline after the loop guard,
 * and to the next cycle cell when no move is safe
 *
 * Parameters:
 * const Snapshot& s: Current game
 * SafeAgent& agent: The player's agent (its spanning tree is reused; rollouts use copies)
 * const LookaheadConfig& cfg: Search parameters
 *
 * Returns int: STRAIGHT, TURN_RIGHT or TURN_LEFT
 */
int lookaheadHonest(const Snapshot& s, SafeAgent& agent, const LookaheadConfig& cfg);

/**
 * Same rule as lookaheadHonest but the rollouts run on copies of the real game, so they see the true future foods.
 * This is an upper bound for experiments and must never be used as a teacher or a player
 *
 * Parameters:
 * const SnakeEnv& env: Current game
 * SafeAgent& agent: The player's agent
 * int foods: Foods each rollout must eat (K)
 *
 * Returns int: STRAIGHT, TURN_RIGHT or TURN_LEFT
 */
int lookaheadLeaked(const SnakeEnv& env, SafeAgent& agent, int foods);

/**
 * Steps until the head reaches the food when it enters a cell at a given step, counting that the body moves: the body cell with index i
 * (0 = head) is free for the head from step len - i on, because the tail leaves its cell each step. The baseline A* treats the body as a fixed wall
 * and so misjudges long snakes
 *
 * Parameters:
 * const Snapshot& s: Current game
 * Cell from: Cell the head enters
 * int t0: Step at which it enters (1 for a move made now)
 *
 * Returns int: The step at which the head can reach the food, or -1 if it cannot
 */
int timedDistance(const Snapshot& s, Cell from, int t0);

/**
 * A* with the time-aware distance: among the moves flagged safe, takes the one whose timedDistance to the food is lowest, ties to the baseline's
 * move. Same fallbacks as lookaheadHonest. A cheap player with no simulation, used as a stronger baseline than the plain A*
 *
 * Parameters:
 * const Snapshot& s: Current game
 * SafeAgent& agent: The player's agent
 *
 * Returns int: STRAIGHT, TURN_RIGHT or TURN_LEFT
 */
int timedAstarAction(const Snapshot& s, SafeAgent& agent);

#endif
