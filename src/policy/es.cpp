/**
 * es.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Evolution strategies over flat network weights
 */
#include "snake_ai/policy/es.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

std::vector<float> getWeights(const nn::Sequential& net) {
	std::vector<float> w;
	for (nn::Value& p : net.parameters()) {
		const nn::Storage& d = p.data().data();
		w.insert(w.end(), d.begin(), d.end());
	}
	return w;
}

void setWeights(nn::Sequential& net, const std::vector<float>& w) {
	std::vector<nn::Value> params = net.parameters();
	size_t total = 0;
	for (nn::Value& p : params) total += p.data().data().size();
	if (total != w.size()) throw std::invalid_argument("setWeights: expected " + std::to_string(total) + " weights, got " + std::to_string(w.size()));
	size_t at = 0;
	for (nn::Value& p : params) {
		nn::Storage& d = p.data().data();
		std::copy(w.begin() + static_cast<long>(at), w.begin() + static_cast<long>(at + d.size()), d.begin());
		at += d.size();
	}
}

std::vector<float> esStep(const std::vector<float>& theta, const EsConfig& cfg, int generation, const std::function<double(const std::vector<float>&)>& cost) {
	if (cfg.population < 2 || cfg.population % 2 != 0) throw std::invalid_argument("esStep: population must be even and at least 2");
	if (!(cfg.sigma > 0.0f)) throw std::invalid_argument("esStep: sigma must be positive");
	const size_t n = theta.size(), pop = static_cast<size_t>(cfg.population);

	std::seed_seq seq{cfg.seed, static_cast<unsigned>(generation)};
	std::mt19937 rng(seq);
	std::normal_distribution<float> normal;
	std::vector<std::vector<float>> eps(pop / 2, std::vector<float>(n));
	for (std::vector<float>& e : eps) {
		for (float& x : e) x = normal(rng);
	}

	// Candidate 2k is theta + sigma * eps_k and candidate 2k+1 is theta - sigma * eps_k; each lands in its own slot, so threads cannot change the result
	std::vector<double> costs(pop);
	auto evaluate = [&](size_t i) {
		const float sign = i % 2 == 0 ? 1.0f : -1.0f;
		std::vector<float> cand(n);
		for (size_t j = 0; j < n; ++j) cand[j] = theta[j] + sign * cfg.sigma * eps[i / 2][j];
		costs[i] = cost(cand);
	};
	const size_t workers = static_cast<size_t>(std::max(1, cfg.threads));
	std::vector<std::thread> pool;
	for (size_t t = 0; t < workers; ++t) {
		pool.emplace_back([&, t] {
			for (size_t i = t; i < pop; i += workers) evaluate(i);
		});
	}
	for (std::thread& th : pool) th.join();

	// Centred ranks: the costliest candidate gets -0.5, the cheapest +0.5. Tied candidates share the mean of their ranks, so a flat
	// landscape (every move of the network still picks the same action) gives every candidate 0 and theta stays put
	std::vector<size_t> order(pop);
	std::iota(order.begin(), order.end(), size_t{0});
	std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return costs[a] > costs[b]; });
	std::vector<float> utility(pop);
	for (size_t r = 0; r < pop;) {
		size_t end = r;
		while (end + 1 < pop && costs[order[end + 1]] == costs[order[r]]) ++end;
		const float mid = 0.5f * static_cast<float>(r + end) / static_cast<float>(pop - 1) - 0.5f;
		for (size_t k = r; k <= end; ++k) utility[order[k]] = mid;
		r = end + 1;
	}
	std::vector<float> next = theta;
	const float step = cfg.lr / (static_cast<float>(pop) * cfg.sigma);
	for (size_t i = 0; i < pop; ++i) {
		const float sign = i % 2 == 0 ? 1.0f : -1.0f;
		for (size_t j = 0; j < n; ++j) next[j] += step * utility[i] * sign * eps[i / 2][j];
	}
	return next;
}
