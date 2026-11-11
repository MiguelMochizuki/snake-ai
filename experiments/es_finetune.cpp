/**
 * es_finetune.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Phase 1 of the beat-the-teacher design: fine-tunes the imitation network with evolution strategies on the steps needed
 * to eat the next foods, then compares it with the safe A* on complete games of three board sizes
 */
#include "snake_ai/eval/paired.hpp"
#include "snake_ai/policy/astar_policy.hpp"
#include "snake_ai/policy/es.hpp"
#include "nn/checkpoint.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace {

const int BOARD = 20;               // training board side
const int FOODS = 20;               // foods the network plays in one training episode
const int BATCH = 16;               // episodes per generation, the same for every candidate
const unsigned TRAIN_BASE = 50000;  // training seeds; validation uses 20000+g and the test 100000+g, so the three never meet
const int VALIDATE_EVERY = 10;      // generations between validations
const int PATIENCE = 3;             // validations without improvement before the run stops
const char* const START = "models/imitation_astar.ckpt";
const char* const OUT = "models/es_finetuned.ckpt";

/**
 * Runs fn(0) .. fn(n-1) on worker threads dealt round-robin; each index is handled by exactly one worker
 *
 * Parameters:
 * int n: Number of jobs
 * int threads: Worker threads
 * const std::function<void(int worker, int job)>& fn: The job; workers never share a job
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
 * The network's move: extended features, the safe-move mask and the loop guard, as astarNetAction applies them
 *
 * Parameters:
 * nn::Sequential& net: The extended network
 * SnakeEnv& e: Current game
 * SafeAgent& a: Agent that supplies the features
 *
 * Returns int: STRAIGHT, TURN_RIGHT or TURN_LEFT
 */
int netAction(nn::Sequential& net, SnakeEnv& e, SafeAgent& a) {
	return astarNetAction(net, a.extendedFeatures(snapshotOf(e)), e.sinceFood() > e.loopGuard(), true);
}

/**
 * Plays the safe A* from an empty board until the snake has the given length, so a training episode can start in any phase of the game
 *
 * Parameters:
 * unsigned seed: Food seed
 * int length: Snake length to reach
 *
 * Returns SnakeEnv: The game at that point
 */
SnakeEnv teacherPrefix(unsigned seed, int length) {
	SnakeEnv env(seed, BOARD);
	SafeAgent agent;
	while (!env.dead() && env.score() + 3 < length) env.step(teacherAction(snapshotOf(env), agent));
	return env;
}

/**
 * Steps a player needs to eat the next foods from a game in progress; a player that does not get there scores the worst cost
 *
 * Parameters:
 * SnakeEnv env: The game, copied so the caller's stays untouched
 * const std::function<int(SnakeEnv&, SafeAgent&)>& act: The player
 *
 * Returns double: Steps used, or FOODS * starve limit if the foods were not eaten
 */
double stepsForFoods(SnakeEnv env, const std::function<int(SnakeEnv&, SafeAgent&)>& act) {
	SafeAgent agent;
	const int target = env.score() + FOODS, cap = FOODS * (BOARD * BOARD * 3 / 2);
	int steps = 0;
	while (!env.dead() && env.score() < target && steps < cap) {
		env.step(act(env, agent));
		++steps;
	}
	return env.score() >= target ? steps : cap;
}

/**
 * The episodes of one generation: seeds roll by BATCH each generation, and the starting length cycles through 25%, 50%, 75% and 90% of the board
 *
 * Parameters:
 * int generation: Generation index
 * int threads: Threads that build the prefixes
 *
 * Returns std::vector<std::unique_ptr<SnakeEnv>>: BATCH games ready for the player under test
 */
std::vector<std::unique_ptr<SnakeEnv>> batchFor(int generation, int threads) {
	const int lengths[4] = {BOARD * BOARD / 4, BOARD * BOARD / 2, BOARD * BOARD * 3 / 4, BOARD * BOARD * 9 / 10};
	std::vector<std::unique_ptr<SnakeEnv>> batch(BATCH);
	parallelFor(BATCH, threads, [&](int, int j) {
		batch[j] = std::make_unique<SnakeEnv>(teacherPrefix(TRAIN_BASE + static_cast<unsigned>(generation * BATCH + j), lengths[j % 4]));
	});
	return batch;
}

/**
 * Mean cost of a set of weights over a batch
 *
 * Parameters:
 * const std::vector<float>& w: Weights of the extended network
 * const std::vector<std::unique_ptr<SnakeEnv>>& batch: The episodes
 *
 * Returns double: Mean steps to eat FOODS foods
 */
double meanCost(const std::vector<float>& w, const std::vector<std::unique_ptr<SnakeEnv>>& batch) {
	nn::Sequential net = makeExtendedNet();
	setWeights(net, w);
	double sum = 0;
	for (const auto& e : batch) sum += stepsForFoods(*e, [&](SnakeEnv& env, SafeAgent& a) { return netAction(net, env, a); });
	return sum / static_cast<double>(batch.size());
}

/**
 * Mean cost of the safe A* over a batch, the yardstick for the network's cost
 *
 * Parameters:
 * const std::vector<std::unique_ptr<SnakeEnv>>& batch: The episodes
 *
 * Returns double: Mean steps to eat FOODS foods
 */
double teacherCost(const std::vector<std::unique_ptr<SnakeEnv>>& batch) {
	double sum = 0;
	for (const auto& e : batch) sum += stepsForFoods(*e, [](SnakeEnv& env, SafeAgent& a) { return teacherAction(snapshotOf(env), a); });
	return sum / static_cast<double>(batch.size());
}

/**
 * Plays complete games with the network, one per seed, in seed order; each worker thread owns its network
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
	const int workers = std::max(1, std::min(threads, static_cast<int>(seeds.size())));
	std::vector<nn::Sequential> nets;
	for (int i = 0; i < workers; ++i) {
		nets.push_back(makeExtendedNet());
		setWeights(nets.back(), w);
	}
	parallelFor(static_cast<int>(seeds.size()), workers, [&](int worker, int g) {
		SnakeEnv env(seeds[g], size);
		SafeAgent agent;
		int i = 0;
		for (; i < 200 * size * size && !env.dead(); ++i) env.step(netAction(nets[worker], env, agent));
		const bool full = env.score() == env.maxScore();
		results[g] = {i, env.score(), full, env.dead() && !full};
	});
	return results;
}

/**
 * Validation score of a set of weights: mean steps over the validation games, a game that did not fill the board counting as the step limit
 *
 * Parameters:
 * const std::vector<float>& w: Weights of the extended network
 * int threads: Worker threads
 *
 * Returns double: Mean steps, lower is better
 */
double validationSteps(const std::vector<float>& w, int threads) {
	double sum = 0;
	const std::vector<unsigned> seeds = seedsFor("validate");
	for (const GameResult& g : playNet(boardFor("validate"), seeds, w, threads)) sum += g.full ? g.steps : 200 * BOARD * BOARD;
	return sum / static_cast<double>(seeds.size());
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
 */
void compare(const char* label, int size, const std::vector<unsigned>& seeds, const std::vector<float>& w, int threads) {
	std::vector<GameResult> base = playGames(size, seeds, [](SnakeEnv& e, SafeAgent& a) { return teacherAction(snapshotOf(e), a); }, threads);
	std::vector<GameResult> net = playNet(size, seeds, w, threads);
	std::vector<double> b, t;
	double bs = 0, ns = 0;
	int deaths = 0, notFull = 0;
	for (size_t i = 0; i < seeds.size(); ++i) {
		b.push_back(base[i].steps);
		t.push_back(net[i].steps);
		bs += base[i].steps / static_cast<double>(seeds.size());
		ns += net[i].steps / static_cast<double>(seeds.size());
		deaths += net[i].died;
		notFull += !net[i].full;
	}
	Interval iv = bootstrapRelDiff(b, t, 10000, 1);
	std::printf("%s %dx%d, %zu seeds: A* steps %.1f, network steps %.1f, deaths %d/%zu, not full %d/%zu, change %+.2f%% 95%% CI [%+.2f%%, %+.2f%%], verdict %s\n",
		label, size, size, seeds.size(), bs, ns, deaths, seeds.size(), notFull, seeds.size(), 100 * iv.mean, 100 * iv.lo, 100 * iv.hi, passesGate(iv) ? "PASS" : "FAIL");
}

/**
 * The final test: complete games of three board sizes, never used for training or selection
 *
 * Parameters:
 * const std::vector<float>& w: Weights of the extended network
 * int threads: Worker threads
 */
void finalTest(const std::vector<float>& w, int threads) {
	compare("TEST", boardFor("test"), seedsFor("test"), w, threads);
	std::vector<unsigned> s10, s40;
	for (unsigned g = 0; g < 12; ++g) s10.push_back(100000 + g);
	for (unsigned g = 0; g < 4; ++g) s40.push_back(100000 + g);
	compare("SCALE", 10, s10, w, threads);
	compare("SCALE", 40, s40, w, threads);
}

/**
 * Loads the imitation checkpoint into the widened network
 *
 * Returns std::vector<float>: The starting weights
 */
std::vector<float> startingWeights() {
	nn::Sequential old = makeAstarNet(), wide = makeExtendedNet();
	nn::load_checkpoint(START, old);
	widenFromAstar(wide, old);
	return getWeights(wide);
}

}  // namespace

/**
 * Parses a whole positive decimal number
 *
 * Parameters:
 * const char* text: The text
 * float& out: Receives the value
 *
 * Returns bool: true if the entire text is one number above zero
 */
bool parsePositiveFloat(const char* text, float& out) {
	char* end = nullptr;
	const float v = std::strtof(text, &end);
	if (end == text || *end != '\0' || !(v > 0.0f)) return false;
	out = v;
	return true;
}

/**
 * Usage: es_finetune [generations=200] [threads=12] [--sigma=X] [--lr=X] [--final]   trains and saves models/es_finetuned.ckpt;
 *                    the test games run only with --final, so tuning on validation never touches them
 *        es_finetune --eval [threads=12]                                              reloads the checkpoint and runs the final test only
 */
int main(int argc, char** argv) {
	std::setvbuf(stdout, nullptr, _IOLBF, 0);
	EsConfig cfg;
	int generations = 200, threads = 12;
	bool eval = false, final = false, ok = true;
	std::vector<int*> ints;
	if (argc >= 2 && std::strcmp(argv[1], "--eval") == 0) {
		eval = true;
		ints = {&threads};
	} else {
		ints = {&generations, &threads};
	}
	size_t nextInt = 0;
	for (int i = eval ? 2 : 1; i < argc && ok; ++i) {
		if (std::strncmp(argv[i], "--sigma=", 8) == 0 && !eval) {
			ok = parsePositiveFloat(argv[i] + 8, cfg.sigma);
		} else if (std::strncmp(argv[i], "--lr=", 5) == 0 && !eval) {
			ok = parsePositiveFloat(argv[i] + 5, cfg.lr);
		} else if (std::strcmp(argv[i], "--final") == 0 && !eval) {
			final = true;
		} else if (argv[i][0] != '-' && nextInt < ints.size()) {
			ok = parseInt(argv[i], 1, *ints[nextInt++]);
		} else {
			ok = false;
		}
	}
	if (!ok) {
		std::fprintf(stderr, "usage: %s [generations=200] [threads=12] [--sigma=X] [--lr=X] [--final] | --eval [threads=12]  (integers >= 1, numbers > 0, nothing else)\n", argv[0]);
		return 2;
	}

	if (eval) {
		nn::Sequential net = makeExtendedNet();
		nn::load_checkpoint(OUT, net);
		std::printf("es_finetune --eval: %s, threads %d\n", OUT, threads);
		finalTest(getWeights(net), threads);
		return 0;
	}

	cfg.threads = threads;
	std::vector<float> theta = startingWeights();
	std::vector<float> best = theta;
	double bestVal = validationSteps(theta, threads);
	std::printf("es_finetune: generations %d, threads %d, population %d, sigma %g, lr %g, batch %d episodes of %d foods, starting lengths 25/50/75/90%% of %dx%d\n",
		generations, threads, cfg.population, cfg.sigma, cfg.lr, BATCH, FOODS, BOARD, BOARD);
	std::printf("validation start: %.1f steps\n", bestVal);
	int stale = 0;
	for (int g = 0; g < generations; ++g) {
		auto batch = batchFor(g, threads);
		const double net = meanCost(theta, batch), teacher = teacherCost(batch);
		std::printf("gen %3d  network cost %7.1f  A* cost %7.1f  (%+.1f%%)\n", g, net, teacher, 100 * (net - teacher) / teacher);
		theta = esStep(theta, cfg, g, [&](const std::vector<float>& w) { return meanCost(w, batch); });
		const bool last = g + 1 == generations;
		if ((g + 1) % VALIDATE_EVERY != 0 && !last) continue;
		const double val = validationSteps(theta, threads);
		const bool improved = val < bestVal;
		std::printf("validation after gen %3d: %.1f steps%s\n", g, val, improved ? "  (best)" : "");
		if (improved) {
			bestVal = val;
			best = theta;
			stale = 0;
		} else if (++stale >= PATIENCE) {
			std::printf("no improvement in %d validations: stopping\n", PATIENCE);
			break;
		}
	}
	nn::Sequential out = makeExtendedNet();
	setWeights(out, best);
	nn::save_checkpoint(OUT, out, nullptr, generations);
	std::printf("best validation %.1f steps, saved %s\n", bestVal, OUT);
	if (final) finalTest(best, threads);
	return 0;
}
