/**
 * compare.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Compares the Hamiltonian expert, a network trained to imitate it and a DQN agent on fresh games
 */
#include "snake_ai/teachers/expert.hpp"
#include "snake_ai/teachers/heuristic.hpp"
#include "snake_ai/policy/policy_net.hpp"
#include "snake_ai/env/snake_env.hpp"
#include "nn/checkpoint.hpp"
#include "nn/nn/loss.hpp"
#include "nn/optim/adam.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>

using namespace nn;

namespace {

const size_t F = SnakeEnv::FEATURES;
const size_t A = SnakeEnv::ACTIONS;

const int EVAL_GAMES = 30;
const int EVAL_MAX_STEPS = 60000;
const unsigned EVAL_SEED = 100000;

const int IMITATION_GAMES = 60;
const int IMITATION_MAX_STEPS = 50000;
const int IMITATION_STRIDE = 7;  // keeps one state in seven: neighbouring states are near copies
const int IMITATION_EPOCHS = 8;
const char* const IMITATION_CHECKPOINT = "imitation.ckpt";

const int DQN_STEPS = 300000;
const size_t DQN_BATCH = 64;
const size_t DQN_BUFFER = 50000;
const int DQN_WARMUP = 2000;
const int DQN_SYNC = 1000;
const float DQN_GAMMA = 0.95f;
const float DQN_LR = 1e-3f;

/**
 * Plays fresh games with a policy and prints the result
 *
 * Parameters:
 * const char* name: Label
 * Policy policy: Callable taking const SnakeEnv& and returning an action
 */
template <class Policy>
void evaluate(const char* name, Policy policy) {
	std::vector<int> scores;
	int deaths = 0;
	double steps = 0;
	for (int g = 0; g < EVAL_GAMES; ++g) {
		SnakeEnv env(EVAL_SEED + g);
		int i = 0;
		for (; i < EVAL_MAX_STEPS && !env.dead(); ++i) {
			env.step(policy(env));
		}
		steps += i;
		deaths += env.dead() && env.score() < env.maxScore();
		scores.push_back(env.score());
	}
	double mean = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
	std::printf("%-10s mean %6.1f  min %3d  max %3d  deaths %d/%d  steps/game %.0f\n", name, mean,
		*std::min_element(scores.begin(), scores.end()), *std::max_element(scores.begin(), scores.end()),
		deaths, EVAL_GAMES, steps / EVAL_GAMES);
	std::fflush(stdout);
}

/**
 * Trains a network to copy the expert, then evaluates it
 */
void imitation() {
	std::vector<float> xs, ys;
	for (int g = 0; g < IMITATION_GAMES; ++g) {
		SnakeEnv env(static_cast<unsigned>(g + 1));
		for (int i = 0; i < IMITATION_MAX_STEPS && !env.dead(); ++i) {
			int a = expertAction(env);
			if (i % IMITATION_STRIDE == 0) {
				std::vector<float> f = env.features();
				xs.insert(xs.end(), f.begin(), f.end());
				ys.push_back(static_cast<float>(a));
			}
			env.step(a);
		}
	}
	const size_t n = ys.size();
	std::printf("imitation: %zu samples from %d expert games\n", n, IMITATION_GAMES);

	Sequential net = makeNet(32);
	Adam optimizer(net.parameters(), 3e-3f);
	std::vector<size_t> order(n);
	std::iota(order.begin(), order.end(), 0);
	std::mt19937 rng(7);
	for (int epoch = 0; epoch < IMITATION_EPOCHS; ++epoch) {
		std::shuffle(order.begin(), order.end(), rng);
		double total = 0;
		size_t batches = 0;
		for (size_t i = 0; i + DQN_BATCH <= n; i += DQN_BATCH) {
			std::vector<float> bx, by;
			for (size_t j = i; j < i + DQN_BATCH; ++j) {
				size_t k = order[j];
				bx.insert(bx.end(), xs.begin() + k * F, xs.begin() + (k + 1) * F);
				by.push_back(ys[k]);
			}
			Value loss = cross_entropy_loss(net.forward(Value::leaf(Tensor({DQN_BATCH, F}, bx), false)),
				Tensor({DQN_BATCH}, by));
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
			total += loss.data().data()[0];
			++batches;
		}
		std::printf("  epoch %d loss %.4f\n", epoch, total / batches);
		std::fflush(stdout);
	}
	// Save, then evaluate the reloaded copy so the file is proven to hold the trained network
	save_checkpoint(IMITATION_CHECKPOINT, net, nullptr, IMITATION_EPOCHS);
	std::printf("  saved %s\n", IMITATION_CHECKPOINT);
	Sequential loaded = makeNet(32);
	load_checkpoint(IMITATION_CHECKPOINT, loaded);
	evaluate("imitation", [&](const SnakeEnv& env) { return greedy(loaded, env.features()); });
}

/**
 * One remembered step
 */
struct Transition {
	std::vector<float> s, s2;
	int a;
	float r;
	bool done;
};

/**
 * Heuristic value of a state, used as the potential for reward shaping
 *
 * Parameters:
 * const Genome& g: Evolved weights
 * const std::vector<float>& f: SnakeEnv::egoFeatures()
 *
 * Returns float: Best move score, clipped to [-3, 3]
 */
float potential(const Genome& g, const std::vector<float>& f) {
	float best = -3.0f;
	for (int a = 0; a < 3; ++a) {
		std::array<float, SnakeEnv::EGO_PER_ACTION> e;
		std::copy(f.begin() + a * SnakeEnv::EGO_PER_ACTION, f.begin() + (a + 1) * SnakeEnv::EGO_PER_ACTION, e.begin());
		best = std::max(best, std::min(3.0f, heuristicScore(g, e, f.back())));
	}
	return best;
}

/**
 * Trains a DQN, then evaluates it. Plain: +1 food, -1 death, -0.01 per step on the cycle features.
 * Ego: rotation- and scale-invariant features; death costs more the longer the snake (-1 - 4 * length fraction);
 * revisiting recent cells costs 0.05 per visit; and the evolved heuristic shapes the reward as a potential
 *
 * Parameters:
 * bool ego: Use the invariant features and the shaped reward
 * const char* genomePath: Genome file for the shaping (ego only)
 */
void dqn(bool ego, const char* genomePath) {
	const size_t in = ego ? SnakeEnv::EGO_FEATURES : F;
	const Genome genome = ego ? loadGenome(genomePath) : Genome();
	auto features = [&](const SnakeEnv& e) { return ego ? e.egoFeatures() : e.features(); };
	Sequential net = makeNet(64, in);
	Sequential target = makeNet(64, in);
	auto sync = [&] {
		std::vector<Value> from = net.parameters(), to = target.parameters();
		for (size_t i = 0; i < from.size(); ++i) {
			to[i].data() = from[i].data();
		}
	};
	sync();
	Adam optimizer(net.parameters(), DQN_LR);
	std::vector<Transition> buffer;
	std::mt19937 rng(11);
	std::uniform_real_distribution<float> unit(0.0f, 1.0f);

	unsigned seed = 1;
	SnakeEnv env(seed);
	int episodeScore = 0, episodes = 0;
	double recent = 0;
	for (int t = 0; t < DQN_STEPS; ++t) {
		float eps = std::max(0.05f, 1.0f - t / (0.6f * DQN_STEPS));
		std::vector<float> s = features(env);
		int a = unit(rng) < eps ? static_cast<int>(rng() % A) : greedy(net, s);
		float lengthBefore = env.lengthFraction();
		bool ate = env.step(a);
		float r = env.dead() ? -1.0f - (ego ? 4.0f * lengthBefore : 0.0f) : ate ? 1.0f : -0.01f;
		std::vector<float> s2 = features(env);
		if (ego) {
			if (!env.dead()) r -= 0.05f * std::max(0, env.headVisits() - 1);
			r += 0.3f * ((env.dead() ? 0.0f : DQN_GAMMA * potential(genome, s2)) - potential(genome, s));
		}
		Transition tr{s, s2, a, r, env.dead()};
		if (buffer.size() < DQN_BUFFER) {
			buffer.push_back(std::move(tr));
		} else {
			buffer[static_cast<size_t>(t) % DQN_BUFFER] = std::move(tr);
		}
		if (env.dead()) {
			recent = 0.95 * recent + 0.05 * env.score();
			episodeScore = env.score();
			++episodes;
			env = SnakeEnv(++seed);
		}

		if (t >= DQN_WARMUP && t % 2 == 0) {
			std::vector<float> bs, bs2, mask(DQN_BATCH * A, 0.0f), tgt(DQN_BATCH * A, 0.0f);
			std::vector<float> rewards, notDone;
			std::vector<int> actions;
			for (size_t b = 0; b < DQN_BATCH; ++b) {
				const Transition& x = buffer[rng() % buffer.size()];
				bs.insert(bs.end(), x.s.begin(), x.s.end());
				bs2.insert(bs2.end(), x.s2.begin(), x.s2.end());
				actions.push_back(x.a);
				rewards.push_back(x.r);
				notDone.push_back(x.done ? 0.0f : 1.0f);
			}
			{
				NoGradGuard noGrad;
				Value q2 = target.forward(Value::leaf(Tensor({DQN_BATCH, in}, bs2), false));
				const std::vector<float>& q = q2.data().data();
				for (size_t b = 0; b < DQN_BATCH; ++b) {
					float best = *std::max_element(q.begin() + b * A, q.begin() + (b + 1) * A);
					mask[b * A + actions[b]] = 1.0f;
					tgt[b * A + actions[b]] = rewards[b] + DQN_GAMMA * notDone[b] * best;
				}
			}
			// Only the taken action carries error: mask zeroes the other outputs and targets
			Value pred = net.forward(Value::leaf(Tensor({DQN_BATCH, in}, bs), false));
			Value loss = mse_loss(pred * Value::leaf(Tensor({DQN_BATCH, A}, mask), false),
				Value::leaf(Tensor({DQN_BATCH, A}, tgt), false));
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
		}
		if (t % DQN_SYNC == 0) {
			sync();
		}
		if (t % 25000 == 0) {
			std::printf("  dqn step %6d eps %.2f episodes %5d recent score %.1f\n", t, eps, episodes, recent);
			std::fflush(stdout);
		}
	}
	(void)episodeScore;
	evaluate(ego ? "dqn-ego" : "dqn", [&](const SnakeEnv& e) { return greedy(net, features(e)); });
}

}  // namespace

int main(int argc, char** argv) {
	const char* what = argc > 1 ? argv[1] : "all";
	bool all = !std::strcmp(what, "all");
	if (all || !std::strcmp(what, "expert")) evaluate("expert", expertAction);
	if (all || !std::strcmp(what, "imitation")) imitation();
	const char* genomePath = argc > 2 ? argv[2] : "genome.txt";
	if (all || !std::strcmp(what, "dqn")) dqn(false, genomePath);
	if (!std::strcmp(what, "ga")) {
		Genome g = loadGenome(genomePath);
		evaluate("ga", [&](const SnakeEnv& env) { return heuristicAction(g, env); });
	}
	if (!std::strcmp(what, "dqn-ego")) dqn(true, genomePath);
	return 0;
}
