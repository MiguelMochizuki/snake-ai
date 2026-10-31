/**
 * train_imitation.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Trains an MLP to imitate the BFS expert, then compares both on fresh games
 */
#include "expert.hpp"
#include "snake_env.hpp"
#include "nn/nn/activation.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/loss.hpp"
#include "nn/nn/sequential.hpp"
#include "nn/optim/sgd.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <numeric>
#include <random>

using namespace nn;

namespace {

const int TRAIN_GAMES = 600;
const int SAMPLES_PER_GAME = 300;  // keeps long expert games from dominating the data
const int EVAL_GAMES = 100;
const int EVAL_MAX_STEPS = 3000;
const int EPOCHS = 8;
const size_t BATCH = 64;
const float LR = 0.05f;
const unsigned EVAL_SEED = 100000;

/**
 * Plays a game with a policy
 *
 * Parameters:
 * unsigned seed: Food seed
 * const std::function<int(const SnakeEnv&)>& policy: Chooses the action
 *
 * Returns int: Final score
 */
template <class Policy>
int play(unsigned seed, Policy policy) {
	SnakeEnv env(seed);
	for (int i = 0; i < EVAL_MAX_STEPS && !env.dead(); ++i) {
		env.step(policy(env));
	}
	return env.score();
}

/**
 * Prints mean, max and zero-score share of a set of scores
 */
void report(const char* name, const std::vector<int>& scores) {
	double mean = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
	long zeros = std::count(scores.begin(), scores.end(), 0);
	std::printf("%-8s mean %.2f  max %d  zero-score games %ld/%zu\n", name, mean,
		*std::max_element(scores.begin(), scores.end()), zeros, scores.size());
}

}  // namespace

int main() {
	// 1. Record the expert
	std::vector<float> xs;
	std::vector<float> ys;
	for (int g = 0; g < TRAIN_GAMES; ++g) {
		SnakeEnv env(static_cast<unsigned>(g + 1));
		for (int i = 0; i < SAMPLES_PER_GAME && !env.dead(); ++i) {
			std::vector<float> f = env.features();
			int a = expertAction(env);
			xs.insert(xs.end(), f.begin(), f.end());
			ys.push_back(static_cast<float>(a));
			env.step(a);
		}
	}
	const size_t n = ys.size();
	std::printf("recorded %zu samples from %d expert games\n", n, TRAIN_GAMES);

	// 2. Train
	Sequential model;
	model.add(std::make_unique<Linear>(SnakeEnv::FEATURES, 32));
	model.add(std::make_unique<ReLU>());
	model.add(std::make_unique<Linear>(32, 32));
	model.add(std::make_unique<ReLU>());
	model.add(std::make_unique<Linear>(32, SnakeEnv::ACTIONS));
	SGD optimizer(model.parameters(), LR);

	std::vector<size_t> order(n);
	std::iota(order.begin(), order.end(), 0);
	std::mt19937 rng(7);
	for (int epoch = 0; epoch < EPOCHS; ++epoch) {
		std::shuffle(order.begin(), order.end(), rng);
		double total = 0;
		size_t batches = 0;
		for (size_t i = 0; i + BATCH <= n; i += BATCH) {
			std::vector<float> bx, by;
			for (size_t j = i; j < i + BATCH; ++j) {
				size_t k = order[j];
				bx.insert(bx.end(), xs.begin() + k * SnakeEnv::FEATURES, xs.begin() + (k + 1) * SnakeEnv::FEATURES);
				by.push_back(ys[k]);
			}
			Value x = Value::leaf(Tensor({BATCH, static_cast<size_t>(SnakeEnv::FEATURES)}, bx), false);
			Value loss = cross_entropy_loss(model.forward(x), Tensor({BATCH}, by));
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
			total += loss.data().data()[0];
			++batches;
		}
		std::printf("epoch %d loss %.4f\n", epoch, total / batches);
	}

	// 3. Compare on games neither has seen
	auto netPolicy = [&](const SnakeEnv& env) {
		Value x = Value::leaf(Tensor({1, static_cast<size_t>(SnakeEnv::FEATURES)}, env.features()), false);
		Value out = model.forward(x);
		const std::vector<float>& logits = out.data().data();
		return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
	};
	std::vector<int> net, expert;
	for (int g = 0; g < EVAL_GAMES; ++g) {
		net.push_back(play(EVAL_SEED + g, netPolicy));
		expert.push_back(play(EVAL_SEED + g, expertAction));
	}
	report("expert", expert);
	report("network", net);
	return 0;
}
