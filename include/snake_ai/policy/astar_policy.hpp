/**
 * astar_policy.hpp
 * Author: Miguel Mochizuki Silva
 * Description: The network that imitates the safe A* player, shared by training and playback
 */
#ifndef ASTAR_POLICY_HPP
#define ASTAR_POLICY_HPP

#include "snake_ai/planning/modes.hpp"
#include "nn/autograd/value.hpp"
#include "nn/nn/activation.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/sequential.hpp"

#include <algorithm>
#include <memory>
#include <vector>

/**
 * Builds the network: ACTION_FEATURES -> 32 -> 32 -> ACTIONS
 *
 * Returns nn::Sequential: A freshly initialised network
 */
inline nn::Sequential makeAstarNet() {
	nn::Sequential net;
	net.add(std::make_unique<nn::Linear>(SafeAgent::ACTION_FEATURES, 32));
	net.add(std::make_unique<nn::ReLU>());
	net.add(std::make_unique<nn::Linear>(32, 32));
	net.add(std::make_unique<nn::ReLU>());
	net.add(std::make_unique<nn::Linear>(32, SnakeEnv::ACTIONS));
	return net;
}

/**
 * Builds the network with the extra lookahead features: EXT_FEATURES -> 32 -> 32 -> ACTIONS
 *
 * Returns nn::Sequential: A freshly initialised network
 */
inline nn::Sequential makeExtendedNet() {
	nn::Sequential net;
	net.add(std::make_unique<nn::Linear>(SafeAgent::EXT_FEATURES, 32));
	net.add(std::make_unique<nn::ReLU>());
	net.add(std::make_unique<nn::Linear>(32, 32));
	net.add(std::make_unique<nn::ReLU>());
	net.add(std::make_unique<nn::Linear>(32, SnakeEnv::ACTIONS));
	return net;
}

/**
 * Copies a trained makeAstarNet() into a makeExtendedNet(): every old weight, the old first-layer columns into the matching input columns,
 * and zero for the new input columns, so the output equals the old network's whatever the extra features are
 *
 * Parameters:
 * nn::Sequential& wide: The extended network, overwritten
 * const nn::Sequential& old: The network with ACTION_FEATURES inputs
 */
inline void widenFromAstar(nn::Sequential& wide, const nn::Sequential& old) {
	std::vector<nn::Value> w = wide.parameters(), o = old.parameters();
	const size_t oldIn = SafeAgent::ACTION_FEATURES, newIn = SafeAgent::EXT_FEATURES;
	for (size_t i = 0; i < w.size(); ++i) {
		std::vector<float>& dst = w[i].data().data();
		const std::vector<float>& src = o[i].data().data();
		if (i == 0) {
			// First-layer weight, row-major {out, in}: widen each row
			std::fill(dst.begin(), dst.end(), 0.0f);
			for (size_t r = 0; r < src.size() / oldIn; ++r) std::copy(src.begin() + r * oldIn, src.begin() + (r + 1) * oldIn, dst.begin() + r * newIn);
		} else {
			dst = src;
		}
	}
}

/**
 * Picks the network's best action
 *
 * Parameters:
 * nn::Sequential& net: The network
 * const std::vector<float>& features: SafeAgent::actionFeatures() or extendedFeatures() (the safe and next-cycle flags sit at the same positions in both)
 * bool starving: Take the next cycle cell, as the teacher does after too many steps without food, which ends loops among safe moves
 * bool onlySafe: Restrict the choice to moves that keep the body on a Hamiltonian cycle, which keeps the safety guarantee;
 * if none is safe, take the next cycle cell
 *
 * Returns int: STRAIGHT, TURN_RIGHT or TURN_LEFT
 */
inline int astarNetAction(nn::Sequential& net, const std::vector<float>& features, bool starving, bool onlySafe) {
	if (starving) {
		for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
			if (features[4 * a + 3] > 0.5f) return a;
		}
	}
	nn::NoGradGuard noGrad;
	nn::Value out = net.forward(nn::Value::leaf(nn::Tensor({1, features.size()}, features), false));
	const std::vector<float>& q = out.data().data();
	int best = -1;
	for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
		bool allowed = !onlySafe || features[4 * a + 1] > 0.5f;
		if (allowed && (best < 0 || q[a] > q[best])) best = a;
	}
	if (best >= 0) return best;
	for (int a = 0; a < SnakeEnv::ACTIONS; ++a) {
		if (features[4 * a + 3] > 0.5f) return a;
	}
	return STRAIGHT;
}

#endif
