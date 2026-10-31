/**
 * expert.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Hamiltonian-cycle teacher implementation
 */
#include "expert.hpp"

int expertAction(const SnakeEnv& env) {
	int best = -1, bestRemaining = 0, cycleStep = STRAIGHT;
	for (int a : {STRAIGHT, TURN_RIGHT, TURN_LEFT}) {
		MoveInfo m = env.info(a);
		if (m.ahead == 1 && !m.blocked) {
			cycleStep = a;
		}
		if (m.allowed && (best < 0 || m.remaining < bestRemaining)) {
			best = a;
			bestRemaining = m.remaining;
		}
	}
	return best >= 0 ? best : cycleStep;
}
