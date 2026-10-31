/**
 * heuristic.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Move heuristic implementation
 */
#include "snake_ai/teachers/heuristic.hpp"
#include <fstream>
#include <stdexcept>

float heuristicScore(const Genome& g, const std::array<float, SnakeEnv::EGO_PER_ACTION>& f, float length) {
	if (f[0] > 0.5f) {
		return -1e9f;
	}
	float s = 0;
	for (int i = 0; i < 6; ++i) {
		s += (g.w[i] + g.w[6 + i] * length) * f[i + 1];
	}
	return s;
}

int heuristicAction(const Genome& g, const SnakeEnv& env) {
	int best = STRAIGHT;
	float bestScore = -2e9f;
	for (int a : {STRAIGHT, TURN_RIGHT, TURN_LEFT}) {
		float s = heuristicScore(g, env.ego(a), env.lengthFraction());
		if (s > bestScore) {
			bestScore = s;
			best = a;
		}
	}
	return best;
}

void saveGenome(const std::string& path, const Genome& g) {
	std::ofstream out(path);
	for (float x : g.w) {
		out << x << "\n";
	}
}

Genome loadGenome(const std::string& path) {
	std::ifstream in(path);
	Genome g;
	for (float& x : g.w) {
		if (!(in >> x)) {
			throw std::runtime_error("cannot read genome from " + path);
		}
	}
	return g;
}
