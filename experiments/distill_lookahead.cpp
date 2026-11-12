/**
 * distill_lookahead.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Distils the honest K=1 lookahead into the extended network by supervised learning (DAgger): the lookahead labels the states,
 * first the ones it visits itself, then the ones the network visits, and the result is compared with the safe A* on paired seeds
 */
#include "snake_ai/eval/paired.hpp"
#include "snake_ai/planning/lookahead.hpp"
#include "snake_ai/policy/astar_policy.hpp"
#include "snake_ai/policy/es.hpp"
#include "nn/checkpoint.hpp"
#include "nn/nn/loss.hpp"
#include "nn/optim/adam.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

namespace {

const int BOARD = 20;
const int STRIDE = 4;                 // one state in four: neighbouring states are near copies
const unsigned TRAIN_BASE = 200000;   // training seeds; validation is 20000+g and the test 100000+g, so they never meet
const int EPOCHS = 6;                 // passes over the whole aggregated set per round
const size_t BATCH = 64;
const float LR = 1e-3f;
const size_t F = SafeAgent::EXT_FEATURES;
const char* const START = "models/imitation_astar.ckpt";
const char* const OUT = "models/distilled.ckpt";        // written only by a run with --final
const char* const TRIAL = "models/distill_trial.ckpt";  // where every other run saves

/**
 * Labelled states
 *
 * Attributes:
 * std::vector<float> xs: Extended features, F per state
 * std::vector<float> ys: The lookahead's action for each state
 */
struct Data {
	std::vector<float> xs;
	std::vector<float> ys;
};

/**
 * Runs fn(worker, job) for job in 0..n-1 on worker threads dealt round-robin
 *
 * Parameters:
 * int n: Number of jobs
 * int threads: Worker threads
 * const std::function<void(int, int)>& fn: The job; each index is handled by exactly one worker
 */
void parallelFor(int n, int threads, const std::function<void(int, int)>& fn) {
	const int workers = std::max(1, std::min(threads, n));
	std::vector<std::thread> pool;
	for (int w = 0; w < workers; ++w) {
		pool.emplace_back([&, w] {
			for (int j = w; j < n; j += workers) fn(w, j);
		});
	}
	for (std::thread& t : pool) t.join();
}

/**
 * Loads the imitation checkpoint into the widened network, the same start as es_finetune (exactly the safe A* at the start)
 *
 * Returns nn::Sequential: The starting network
 */
nn::Sequential startingNet() {
	nn::Sequential old = makeAstarNet(), wide = makeExtendedNet();
	nn::load_checkpoint(START, old);
	widenFromAstar(wide, old);
	return wide;
}

/**
 * Plays games and labels their states with the honest K=1 lookahead. With no weights the lookahead itself plays; with weights the
 * network plays (safe-move mask and loop guard on) and the lookahead only labels. Only states with at least two safe moves count, since
 * with one there is nothing to learn, and states after the loop guard are skipped because every player takes the cycle there
 *
 * Parameters:
 * unsigned baseSeed: Seed of the first game
 * int games: Number of games
 * const std::vector<float>* w: Weights of the playing network, or null for the lookahead to play
 * int threads: Worker threads; the data do not depend on it
 *
 * Returns Data: The labelled states, in game order
 */
Data collect(unsigned baseSeed, int games, const std::vector<float>* w, int threads) {
	std::vector<Data> per(games);
	std::vector<nn::Sequential> nets;
	for (int i = 0; i < std::max(1, std::min(threads, games)); ++i) {
		nets.push_back(makeExtendedNet());
		if (w) setWeights(nets.back(), *w);
	}
	LookaheadConfig cfg;
	cfg.foods = 1;
	cfg.samples = 1;
	parallelFor(games, threads, [&](int worker, int g) {
		SnakeEnv env(baseSeed + g, BOARD);
		SafeAgent agent;
		for (int i = 0; i < 200 * BOARD * BOARD && !env.dead(); ++i) {
			const Snapshot snap = snapshotOf(env);
			const std::vector<float> f = agent.extendedFeatures(snap);
			const bool starving = env.sinceFood() > env.loopGuard();
			int safe = 0;
			for (int a = 0; a < SnakeEnv::ACTIONS; ++a) safe += f[4 * a + 1] > 0.5f;
			const bool record = !starving && safe >= 2 && i % STRIDE == 0;
			int act;
			if (w) {
				act = astarNetAction(nets[worker], f, starving, true);
				if (record) {
					per[g].xs.insert(per[g].xs.end(), f.begin(), f.end());
					per[g].ys.push_back(static_cast<float>(lookaheadHonest(snap, agent, cfg)));
				}
			} else {
				act = lookaheadHonest(snap, agent, cfg);
				if (record) {
					per[g].xs.insert(per[g].xs.end(), f.begin(), f.end());
					per[g].ys.push_back(static_cast<float>(act));
				}
			}
			env.step(act);
		}
	});
	Data all;
	for (const Data& d : per) {
		all.xs.insert(all.xs.end(), d.xs.begin(), d.xs.end());
		all.ys.insert(all.ys.end(), d.ys.begin(), d.ys.end());
	}
	return all;
}

/**
 * Trains the network on the aggregated set with cross entropy
 *
 * Parameters:
 * nn::Sequential& net: The network, updated in place
 * const Data& d: The labelled states
 * unsigned seed: Seed of the shuffling
 *
 * Returns double: Mean loss of the last epoch
 */
double train(nn::Sequential& net, const Data& d, unsigned seed) {
	using namespace nn;
	const size_t n = d.ys.size();
	Adam optimizer(net.parameters(), LR);
	std::vector<size_t> order(n);
	std::iota(order.begin(), order.end(), 0);
	std::mt19937 rng(seed);
	double last = 0;
	for (int epoch = 0; epoch < EPOCHS; ++epoch) {
		std::shuffle(order.begin(), order.end(), rng);
		double total = 0;
		size_t batches = 0;
		for (size_t i = 0; i + BATCH <= n; i += BATCH) {
			std::vector<float> bx, by;
			for (size_t j = i; j < i + BATCH; ++j) {
				const size_t k = order[j];
				bx.insert(bx.end(), d.xs.begin() + k * F, d.xs.begin() + (k + 1) * F);
				by.push_back(d.ys[k]);
			}
			Value loss = cross_entropy_loss(net.forward(Value::leaf(Tensor({BATCH, F}, bx), false)), Tensor({BATCH}, by));
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
			total += loss.data().data()[0];
			++batches;
		}
		last = total / static_cast<double>(batches);
	}
	return last;
}

/**
 * Plays complete games with the network, one per seed, in seed order
 *
 * Parameters:
 * int size: Board side
 * const std::vector<unsigned>& seeds: Food seeds
 * const std::vector<float>& w: Weights of the extended network
 * int threads: Worker threads; the results do not depend on it
 *
 * Returns std::vector<GameResult>: One result per seed
 */
std::vector<GameResult> playNet(int size, const std::vector<unsigned>& seeds, const std::vector<float>& w, int threads) {
	std::vector<GameResult> results(seeds.size());
	std::vector<nn::Sequential> nets;
	for (int i = 0; i < std::max(1, std::min(threads, static_cast<int>(seeds.size()))); ++i) {
		nets.push_back(makeExtendedNet());
		setWeights(nets.back(), w);
	}
	parallelFor(static_cast<int>(seeds.size()), threads, [&](int worker, int g) {
		SnakeEnv env(seeds[g], size);
		SafeAgent agent;
		int i = 0;
		for (; i < 200 * size * size && !env.dead(); ++i) {
			env.step(astarNetAction(nets[worker], agent.extendedFeatures(snapshotOf(env)), env.sinceFood() > env.loopGuard(), true));
		}
		const bool full = env.score() == env.maxScore();
		results[g] = {i, env.score(), full, env.dead() && !full};
	});
	return results;
}

/**
 * Plays the safe A* and the network on the same seeds and prints the paired comparison with its verdict
 *
 * Parameters:
 * const char* label: Name of the evaluation
 * int size: Board side
 * const std::vector<unsigned>& seeds: Food seeds
 * const std::vector<float>& w: Weights of the extended network
 * int threads: Worker threads
 *
 * Returns double: Mean steps of the network, a game that did not fill the board counting as the step limit
 */
double compare(const char* label, int size, const std::vector<unsigned>& seeds, const std::vector<float>& w, int threads) {
	std::vector<GameResult> base = playGames(size, seeds, [](SnakeEnv& e, SafeAgent& a) { return teacherAction(snapshotOf(e), a); }, threads);
	std::vector<GameResult> net = playNet(size, seeds, w, threads);
	std::vector<double> b, t;
	double bs = 0, ns = 0, score = 0;
	int deaths = 0, notFull = 0;
	const double n = static_cast<double>(seeds.size());
	for (size_t i = 0; i < seeds.size(); ++i) {
		b.push_back(base[i].steps);
		t.push_back(net[i].steps);
		bs += base[i].steps / n;
		ns += net[i].steps / n;
		score += (net[i].full ? net[i].steps : 200 * size * size) / n;
		deaths += net[i].died;
		notFull += !net[i].full;
	}
	Interval iv = bootstrapRelDiff(b, t, 10000, 1);
	std::printf("%s %dx%d, %zu seeds: A* steps %.1f, network steps %.1f, deaths %d/%zu, not full %d/%zu, change %+.2f%% 95%% CI [%+.2f%%, %+.2f%%], verdict %s\n",
		label, size, size, seeds.size(), bs, ns, deaths, seeds.size(), notFull, seeds.size(), 100 * iv.mean, 100 * iv.lo, 100 * iv.hi, passesGate(iv) ? "PASS" : "FAIL");
	std::fflush(stdout);
	return score;
}

}  // namespace

/**
 * Usage: distill_lookahead [rounds=4] [games=48] [threads=12] [--final]
 * Round 0 labels the states the K=1 lookahead visits; each later round labels the states the current network visits (DAgger) and the
 * network is retrained on everything collected so far. The weights with the fewest validation steps are kept. Only --final plays the
 * test seeds and writes models/distilled.ckpt; any other run writes models/distill_trial.ckpt. Run from the repository root
 */
int main(int argc, char** argv) {
	std::setvbuf(stdout, nullptr, _IOLBF, 0);
	int rounds = 4, games = 48, threads = 12;
	bool final = false;
	int pos = 0;
	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "--final") == 0) {
			final = true;
		} else if (!(pos == 0 && parseInt(argv[i], 1, rounds)) && !(pos == 1 && parseInt(argv[i], 1, games)) && !(pos == 2 && parseInt(argv[i], 1, threads))) {
			std::fprintf(stderr, "usage: %s [rounds=4] [games=48] [threads=12] [--final]  (integers >= 1)\n", argv[0]);
			return 2;
		} else {
			++pos;
		}
	}
	std::printf("distill_lookahead: %d rounds, %d games per round, threads %d%s\n", rounds, games, threads, final ? ", --final" : "");

	nn::Sequential net = startingNet();
	Data all;
	std::vector<float> best = getWeights(net);
	double bestSteps = 1e18;
	const std::vector<unsigned> val = seedsFor("validate");
	for (int r = 0; r < rounds; ++r) {
		const std::vector<float> w = getWeights(net);
		Data d = collect(TRAIN_BASE + static_cast<unsigned>(r * games), games, r == 0 ? nullptr : &w, threads);
		all.xs.insert(all.xs.end(), d.xs.begin(), d.xs.end());
		all.ys.insert(all.ys.end(), d.ys.begin(), d.ys.end());
		const double loss = train(net, all, 7 + r);
		std::printf("round %d (%s): %zu new states, %zu total, loss %.4f\n", r, r == 0 ? "lookahead plays" : "network plays", d.ys.size(), all.ys.size(), loss);
		const double steps = compare("VALIDATION", boardFor("validate"), val, getWeights(net), threads);
		if (steps < bestSteps) {
			bestSteps = steps;
			best = getWeights(net);
		}
	}
	setWeights(net, best);
	nn::save_checkpoint(final ? OUT : TRIAL, net, nullptr, rounds);
	std::printf("best validation steps %.1f, saved %s\n", bestSteps, final ? OUT : TRIAL);
	if (final) {
		compare("TEST", boardFor("test"), seedsFor("test"), best, threads);
	}
	return 0;
}
