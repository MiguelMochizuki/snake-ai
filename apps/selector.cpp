/**
 * selector.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Trains a network to pick the playing mode (A*, shortcuts, cycle) from rollouts, then compares it with fixed schedules
 */
#include "snake_ai/planning/modes.hpp"
#include "nn/autograd/value.hpp"
#include "nn/checkpoint.hpp"
#include "nn/nn/activation.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/loss.hpp"
#include "nn/nn/sequential.hpp"
#include "nn/optim/adam.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <thread>

using namespace nn;

namespace {

const int DATA_GAMES = 48;          // games played with random mode switches to reach varied states
const int SAMPLE_EVERY = 90;        // steps between labelled states
const int COMMIT_STEPS = 100;       // steps a candidate mode is held
const int FOLLOW_STEPS = 150;       // steps of A* afterwards, so a mode that sets up a bad future is charged for it
const int MARGIN = 1;               // another mode must eat this many more foods to beat A*
const int EPOCHS = 12;
const size_t BATCH = 64;
const int EVAL_GAMES = 12;
const unsigned EVAL_SEED = 100000;
const char* const CHECKPOINT = "selector.ckpt";

const size_t F = SafeAgent::FEATURES;

/**
 * Foods eaten when a mode is held for a while and A* follows
 *
 * Parameters:
 * SnakeEnv env: Copy of the state (taken by value)
 * SafeAgent agent: Copy of the agent
 * Mode mode: Mode to hold
 *
 * Returns int: Score gained
 */
int rollout(SnakeEnv env, SafeAgent agent, Mode mode) {
	int start = env.score();
	for (int i = 0; i < COMMIT_STEPS + FOLLOW_STEPS && !env.dead(); ++i) {
		env.step(agent.act(env, i < COMMIT_STEPS ? mode : ASTAR));
	}
	return env.score() - start;
}

struct Sample {
	std::vector<float> x;
	int label;
};

/**
 * Plays one game with random mode switches and labels states along it
 *
 * Parameters:
 * unsigned seed: Food seed and randomness
 *
 * Returns std::vector<Sample>: Labelled states
 */
std::vector<Sample> collect(unsigned seed) {
	std::mt19937 rng(seed);
	SnakeEnv env(seed);
	SafeAgent agent;
	std::vector<Sample> out;
	Mode mode = ASTAR;
	for (int i = 0; i < 40000 && !env.dead(); ++i) {
		if (i % 40 == 0) mode = static_cast<Mode>(rng() % MODES);
		if (i % SAMPLE_EVERY == 0 && i > 0) {
			int gain[MODES];
			for (int m = 0; m < MODES; ++m) gain[m] = rollout(env, agent, static_cast<Mode>(m));
			int best = ASTAR;
			for (int m = 0; m < MODES; ++m) {
				if (gain[m] >= gain[best] + MARGIN) best = m;
			}
			out.push_back({agent.features(env), best});
		}
		env.step(agent.act(env, env.sinceFood() > env.loopGuard() ? CYCLE : mode));
	}
	return out;
}

/**
 * Plays games with a mode picker
 *
 * Parameters:
 * const char* name: Label
 * const std::function<Mode(const SnakeEnv&, SafeAgent&)>& pick: Chooses the mode for a state
 */
void evaluate(const char* name, const std::function<Mode(const SnakeEnv&, SafeAgent&)>& pick) {
	double score = 0, steps = 0;
	int died = 0;
	int used[MODES] = {0, 0, 0};
	for (int g = 0; g < EVAL_GAMES; ++g) {
		SnakeEnv env(EVAL_SEED + g);
		SafeAgent agent;
		int i = 0;
		for (; i < 200 * env.size() * env.size() && !env.dead(); ++i) {
			Mode m = env.sinceFood() > env.loopGuard() ? CYCLE : pick(env, agent);
			++used[m];
			env.step(agent.act(env, m));
		}
		score += env.score();
		steps += i;
		died += env.dead() && env.score() < env.maxScore();
	}
	double total = used[0] + used[1] + used[2];
	std::printf("%-28s score %6.1f  steps %7.0f  deaths %d/%d  modes A* %2.0f%% shortcuts %2.0f%% cycle %2.0f%%\n", name,
		score / EVAL_GAMES, steps / EVAL_GAMES, died, EVAL_GAMES, 100 * used[0] / total, 100 * used[1] / total, 100 * used[2] / total);
	std::fflush(stdout);
}

}  // namespace

int main() {
	// 1. Label states by rollouts, games in parallel
	std::vector<std::vector<Sample>> parts(DATA_GAMES);
	std::vector<std::thread> pool;
	unsigned workers = std::max(1u, std::thread::hardware_concurrency());
	for (unsigned w = 0; w < workers; ++w) {
		pool.emplace_back([&, w] {
			for (unsigned g = w; g < static_cast<unsigned>(DATA_GAMES); g += workers) parts[g] = collect(500 + g);
		});
	}
	for (auto& t : pool) t.join();
	std::vector<Sample> data;
	for (auto& p : parts) data.insert(data.end(), p.begin(), p.end());
	int counts[MODES] = {0, 0, 0};
	for (const Sample& s : data) ++counts[s.label];
	std::printf("labelled %zu states: A* %d, shortcuts %d, cycle %d\n", data.size(), counts[0], counts[1], counts[2]);
	std::fflush(stdout);

	// 2. Train the selector
	Sequential net;
	net.add(std::make_unique<Linear>(F, 32));
	net.add(std::make_unique<ReLU>());
	net.add(std::make_unique<Linear>(32, 32));
	net.add(std::make_unique<ReLU>());
	net.add(std::make_unique<Linear>(32, MODES));
	Adam optimizer(net.parameters(), 3e-3f);
	std::vector<size_t> order(data.size());
	std::iota(order.begin(), order.end(), 0);
	std::mt19937 rng(5);
	for (int epoch = 0; epoch < EPOCHS; ++epoch) {
		std::shuffle(order.begin(), order.end(), rng);
		double total = 0;
		size_t batches = 0;
		for (size_t i = 0; i + BATCH <= order.size(); i += BATCH) {
			std::vector<float> bx, by;
			for (size_t j = i; j < i + BATCH; ++j) {
				const Sample& s = data[order[j]];
				bx.insert(bx.end(), s.x.begin(), s.x.end());
				by.push_back(static_cast<float>(s.label));
			}
			Value loss = cross_entropy_loss(net.forward(Value::leaf(Tensor({BATCH, F}, bx), false)), Tensor({BATCH}, by));
			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
			total += loss.data().data()[0];
			++batches;
		}
		if (epoch % 3 == 2) std::printf("  epoch %d loss %.4f\n", epoch, total / batches);
	}
	save_checkpoint(CHECKPOINT, net, nullptr, EPOCHS);
	std::printf("saved %s\n", CHECKPOINT);

	// 3. Compare on fresh games
	evaluate("cycle", [](const SnakeEnv&, SafeAgent&) { return CYCLE; });
	evaluate("shortcuts", [](const SnakeEnv&, SafeAgent&) { return SHORTCUT; });
	evaluate("A*", [](const SnakeEnv&, SafeAgent&) { return ASTAR; });
	evaluate("A* until 100, then shortcuts", [](const SnakeEnv& e, SafeAgent&) {
		return e.state().getSnake().getLength() < 100 ? ASTAR : SHORTCUT;
	});
	evaluate("learned selector", [&](const SnakeEnv& env, SafeAgent& agent) {
		NoGradGuard noGrad;
		Value out = net.forward(Value::leaf(Tensor({1, F}, agent.features(env)), false));
		const std::vector<float>& q = out.data().data();
		return static_cast<Mode>(std::max_element(q.begin(), q.end()) - q.begin());
	});
	return 0;
}
