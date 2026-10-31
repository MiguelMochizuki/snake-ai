/**
 * compare.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Compares the Hamiltonian expert, a network trained to imitate it and a DQN agent on fresh games
 */
#include "expert.hpp"
#include "snake_env.hpp"
#include "nn/autograd/value.hpp"
#include "nn/nn/activation.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/loss.hpp"
#include "nn/nn/sequential.hpp"
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

const int DQN_STEPS = 300000;
const size_t DQN_BATCH = 64;
const size_t DQN_BUFFER = 50000;
const int DQN_WARMUP = 2000;
const int DQN_SYNC = 1000;
const float DQN_GAMMA = 0.95f;
const float DQN_LR = 1e-3f;

/**
 * Builds the shared network shape
 *
 * Parameters:
 * size_t hidden: Width of both hidden layers
 *
 * Returns Sequential: FEATURES -> hidden -> hidden -> ACTIONS
 */
Sequential makeNet(size_t hidden) {
	Sequential net;
	net.add(std::make_unique<Linear>(F, hidden));
	net.add(std::make_unique<ReLU>());
	net.add(std::make_unique<Linear>(hidden, hidden));
	net.add(std::make_unique<ReLU>());
	net.add(std::make_unique<Linear>(hidden, A));
	return net;
}

/**
 * Index of the largest output for a batch of one state
 *
 * Parameters:
 * Sequential& net: The network
 * const std::vector<float>& features: One state
 *
 * Returns int: Best action
 */
int greedy(Sequential& net, const std::vector<float>& features) {
	NoGradGuard noGrad;
	Value out = net.forward(Value::leaf(Tensor({1, F}, features), false));
	const std::vector<float>& q = out.data().data();
	return static_cast<int>(std::max_element(q.begin(), q.end()) - q.begin());
}

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
	for (int g = 0; g < EVAL_GAMES; ++g) {
		SnakeEnv env(EVAL_SEED + g);
		for (int i = 0; i < EVAL_MAX_STEPS && !env.dead(); ++i) {
			env.step(policy(env));
		}
		deaths += env.dead() && env.score() < 397;
		scores.push_back(env.score());
	}
	double mean = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
	std::printf("%-10s mean %6.1f  min %3d  max %3d  deaths %d/%d\n", name, mean,
		*std::min_element(scores.begin(), scores.end()), *std::max_element(scores.begin(), scores.end()),
		deaths, EVAL_GAMES);
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
	evaluate("imitation", [&](const SnakeEnv& env) { return greedy(net, env.features()); });
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
 * Trains a DQN from reward alone (+1 food, -1 death, -0.01 per step), then evaluates it
 */
void dqn() {
	Sequential net = makeNet(64);
	Sequential target = makeNet(64);
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
		std::vector<float> s = env.features();
		int a = unit(rng) < eps ? static_cast<int>(rng() % A) : greedy(net, s);
		bool ate = env.step(a);
		float r = env.dead() ? -1.0f : ate ? 1.0f : -0.01f;
		Transition tr{s, env.features(), a, r, env.dead()};
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
				Value q2 = target.forward(Value::leaf(Tensor({DQN_BATCH, F}, bs2), false));
				const std::vector<float>& q = q2.data().data();
				for (size_t b = 0; b < DQN_BATCH; ++b) {
					float best = *std::max_element(q.begin() + b * A, q.begin() + (b + 1) * A);
					mask[b * A + actions[b]] = 1.0f;
					tgt[b * A + actions[b]] = rewards[b] + DQN_GAMMA * notDone[b] * best;
				}
			}
			// Only the taken action carries error: mask zeroes the other outputs and targets
			Value pred = net.forward(Value::leaf(Tensor({DQN_BATCH, F}, bs), false));
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
	evaluate("dqn", [&](const SnakeEnv& e) { return greedy(net, e.features()); });
}

}  // namespace

int main(int argc, char** argv) {
	const char* what = argc > 1 ? argv[1] : "all";
	bool all = !std::strcmp(what, "all");
	if (all || !std::strcmp(what, "expert")) evaluate("expert", expertAction);
	if (all || !std::strcmp(what, "imitation")) imitation();
	if (all || !std::strcmp(what, "dqn")) dqn();
	return 0;
}
