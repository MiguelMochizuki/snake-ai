/**
 * evolve.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Genetic algorithm that evolves the heuristic's length-dependent weights
 */
#include "snake_ai/teachers/heuristic.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <random>
#include <thread>

namespace {

const int POP = 40;
const int GENERATIONS = 40;
const int ELITE = 4;
const int GAMES = 4;
int maxSteps = 4000;  // set from the command line
bool noCoil = false;  // ablation: keep the coil weights at zero
const float MUTATION = 0.25f;

/**
 * Mean score over a few games; loops end in starvation, so they cost points by themselves
 *
 * Parameters:
 * const Genome& g: Weights
 * unsigned seedBase: First food seed
 *
 * Returns double: Mean final score
 */
double fitness(const Genome& g, unsigned seedBase) {
	double total = 0;
	for (int i = 0; i < GAMES; ++i) {
		SnakeEnv env(seedBase + i);
		for (int s = 0; s < maxSteps && !env.dead(); ++s) {
			env.step(heuristicAction(g, env));
		}
		total += env.score();
	}
	return total / GAMES;
}

}  // namespace

int main(int argc, char** argv) {
	const char* out = argc > 1 ? argv[1] : "genome.txt";
	if (argc > 2) maxSteps = std::atoi(argv[2]);
	noCoil = argc > 3 && std::string(argv[3]) == "nocoil";
	std::mt19937 rng(2024);
	std::normal_distribution<float> gauss(0.0f, 1.0f);
	std::uniform_real_distribution<float> unit(0.0f, 1.0f);

	std::vector<Genome> pop(POP);
	for (Genome& g : pop) {
		for (float& x : g.w) x = gauss(rng);
		if (noCoil) g.w[5] = g.w[11] = 0.0f;
	}
	std::vector<double> fit(POP);
	Genome best;
	unsigned nthreads = std::max(1u, std::thread::hardware_concurrency());

	for (int gen = 0; gen < GENERATIONS; ++gen) {
		// Fresh seeds every generation so weights cannot memorise one food sequence
		unsigned seedBase = 1000 + static_cast<unsigned>(gen) * 100;
		std::vector<std::thread> pool;
		for (unsigned t = 0; t < nthreads; ++t) {
			pool.emplace_back([&, t] {
				for (size_t i = t; i < pop.size(); i += nthreads) fit[i] = fitness(pop[i], seedBase);
			});
		}
		for (std::thread& th : pool) th.join();

		std::vector<int> order(POP);
		for (int i = 0; i < POP; ++i) order[i] = i;
		std::sort(order.begin(), order.end(), [&](int a, int b) { return fit[a] > fit[b]; });
		best = pop[order[0]];
		double mean = 0;
		for (double f : fit) mean += f;
		std::printf("gen %2d best %6.1f mean %6.1f\n", gen, fit[order[0]], mean / POP);
		std::fflush(stdout);
		saveGenome(out, best);

		// Next generation: keep the elite, breed the rest from tournament winners
		auto pick = [&] {
			int w = static_cast<int>(rng() % POP);
			for (int k = 0; k < 2; ++k) {
				int c = static_cast<int>(rng() % POP);
				if (fit[c] > fit[w]) w = c;
			}
			return pop[w];
		};
		float sigma = MUTATION * (1.0f - 0.7f * gen / GENERATIONS);
		std::vector<Genome> next;
		for (int i = 0; i < ELITE; ++i) next.push_back(pop[order[i]]);
		while (static_cast<int>(next.size()) < POP) {
			Genome a = pick(), b = pick(), c;
			for (size_t i = 0; i < c.w.size(); ++i) {
				c.w[i] = unit(rng) < 0.5f ? a.w[i] : b.w[i];
				if (unit(rng) < 0.3f) c.w[i] += sigma * gauss(rng);
			}
			next.push_back(c);
		}
		if (noCoil) {
			for (Genome& g : next) g.w[5] = g.w[11] = 0.0f;
		}
		pop = next;
	}
	std::printf("saved best genome of the last generation to %s\n", out);
	return 0;
}
