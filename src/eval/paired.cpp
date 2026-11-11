/**
 * paired.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Paired evaluation harness
 */
#include "snake_ai/eval/paired.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <thread>

std::vector<GameResult> playGames(int size, const std::vector<unsigned>& seeds, const std::function<int(SnakeEnv&, SafeAgent&)>& act, int threads) {
	std::vector<GameResult> results(seeds.size());
	auto one = [&](size_t g) {
		SnakeEnv env(seeds[g], size);
		SafeAgent agent;
		int i = 0;
		for (; i < 200 * size * size && !env.dead(); ++i) {
			env.step(act(env, agent));
		}
		bool full = env.score() == env.maxScore();
		results[g] = {i, env.score(), full, env.dead() && !full};
	};
	// Seeds are dealt round-robin; each result lands in its own slot, so the thread count cannot change the output
	const int n = std::max(1, threads);
	std::vector<std::thread> pool;
	for (int t = 0; t < n; ++t) {
		pool.emplace_back([&, t] {
			for (size_t g = static_cast<size_t>(t); g < seeds.size(); g += static_cast<size_t>(n)) one(g);
		});
	}
	for (std::thread& th : pool) th.join();
	return results;
}

Interval bootstrapRelDiff(const std::vector<double>& base, const std::vector<double>& test, int iters, unsigned seed) {
	if (base.size() != test.size() || base.empty()) throw std::invalid_argument("bootstrapRelDiff: need paired, non-empty data");
	const size_t n = base.size();
	auto stat = [&](const std::vector<size_t>& idx) {
		double b = 0, t = 0;
		for (size_t i : idx) {
			b += base[i];
			t += test[i];
		}
		return (t - b) / b;
	};
	std::vector<size_t> all(n);
	for (size_t i = 0; i < n; ++i) all[i] = i;
	std::mt19937 rng(seed);
	std::uniform_int_distribution<size_t> pick(0, n - 1);
	std::vector<double> samples;
	std::vector<size_t> idx(n);
	for (int k = 0; k < iters; ++k) {
		for (size_t& i : idx) i = pick(rng);
		samples.push_back(stat(idx));
	}
	std::sort(samples.begin(), samples.end());
	const size_t m = samples.size();
	return {stat(all), samples[static_cast<size_t>(0.025 * static_cast<double>(m))], samples[std::min(m - 1, static_cast<size_t>(0.975 * static_cast<double>(m)))]};
}

bool passesGate(const Interval& iv) {
	return iv.mean <= -0.03 && iv.hi < 0.0;
}
