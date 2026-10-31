/**
 * imitate_astar.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Trains a network to imitate the safe A* player, then plays it raw and with the safety filter
 */
#include "snake_ai/policy/astar_policy.hpp"
#include "nn/checkpoint.hpp"
#include "nn/nn/loss.hpp"
#include "nn/optim/adam.hpp"

#include <cstdio>
#include <numeric>
#include <random>
#include <thread>

using namespace nn;

namespace {

const int TEACHER_GAMES = 48;
const int STRIDE = 4;  // one state in four: neighbouring states are near copies
const int EPOCHS = 15;
const size_t BATCH = 64;
const int EVAL_GAMES = 12;
const unsigned EVAL_SEED = 100000;
const char* const CHECKPOINT = "imitation_astar.ckpt";
const size_t F = SafeAgent::ACTION_FEATURES;

/**
 * Plays the teacher and records states with its action
 *
 * Parameters:
 * unsigned seed: Food seed
 * std::vector<float>& xs: Receives features
 * std::vector<float>& ys: Receives actions
 */
void record(unsigned seed, std::vector<float>& xs, std::vector<float>& ys) {
	SnakeEnv env(seed);
	SafeAgent agent;
	for (int i = 0; i < 200 * env.size() * env.size() && !env.dead(); ++i) {
		int a = agent.act(env, env.sinceFood() > env.loopGuard() ? CYCLE : ASTAR);
		if (i % STRIDE == 0) {
			std::vector<float> f = agent.actionFeatures(env);
			xs.insert(xs.end(), f.begin(), f.end());
			ys.push_back(static_cast<float>(a));
		}
		env.step(a);
	}
}

/**
 * Plays fresh games with the network
 *
 * Parameters:
 * const char* name: Label
 * Sequential& net: The network
 * bool onlySafe: Apply the safety filter
 */
void evaluate(const char* name, Sequential& net, bool onlySafe) {
	double score = 0, steps = 0;
	int died = 0;
	for (int g = 0; g < EVAL_GAMES; ++g) {
		SnakeEnv env(EVAL_SEED + g);
		SafeAgent agent;
		int i = 0;
		for (; i < 200 * env.size() * env.size() && !env.dead(); ++i) {
			env.step(astarNetAction(net, agent.actionFeatures(env), env.sinceFood() > env.loopGuard(), onlySafe));
		}
		score += env.score();
		steps += i;
		died += env.dead() && env.score() < env.maxScore();
	}
	std::printf("%-22s score %6.1f  steps %7.0f  deaths %d/%d\n", name, score / EVAL_GAMES, steps / EVAL_GAMES, died, EVAL_GAMES);
	std::fflush(stdout);
}

}  // namespace

int main() {
	std::vector<std::vector<float>> xp(TEACHER_GAMES), yp(TEACHER_GAMES);
	std::vector<std::thread> pool;
	unsigned workers = std::max(1u, std::thread::hardware_concurrency());
	for (unsigned w = 0; w < workers; ++w) {
		pool.emplace_back([&, w] {
			for (unsigned g = w; g < static_cast<unsigned>(TEACHER_GAMES); g += workers) record(1 + g, xp[g], yp[g]);
		});
	}
	for (auto& t : pool) t.join();
	std::vector<float> xs, ys;
	for (int g = 0; g < TEACHER_GAMES; ++g) {
		xs.insert(xs.end(), xp[g].begin(), xp[g].end());
		ys.insert(ys.end(), yp[g].begin(), yp[g].end());
	}
	const size_t n = ys.size();
	std::printf("recorded %zu states from %d teacher games\n", n, TEACHER_GAMES);
	std::fflush(stdout);

	Sequential net = makeAstarNet();
	Adam optimizer(net.parameters(), 3e-3f);
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
				bx.insert(bx.end(), xs.begin() + k * F, xs.begin() + (k + 1) * F);
				by.push_back(ys[k]);
			}
			Value loss = cross_entropy_loss(net.forward(Value::leaf(Tensor({BATCH, F}, bx), false)), Tensor({BATCH}, by));
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
			total += loss.data().data()[0];
			++batches;
		}
		std::printf("  epoch %d loss %.4f\n", epoch, total / batches);
		std::fflush(stdout);
	}
	save_checkpoint(CHECKPOINT, net, nullptr, EPOCHS);
	std::printf("saved %s\n", CHECKPOINT);

	// Evaluate the reloaded copy so the file is proven to hold the trained network
	Sequential loaded = makeAstarNet();
	load_checkpoint(CHECKPOINT, loaded);
	evaluate("network, raw", loaded, false);
	evaluate("network, safety filter", loaded, true);
	return 0;
}
