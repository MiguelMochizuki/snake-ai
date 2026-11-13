/**
 * policy_net.hpp
 * Author: Miguel Mochizuki Silva
 * Description: The network shape shared by training and playback, and greedy action choice
 */
#ifndef POLICY_NET_HPP
#define POLICY_NET_HPP

#include "snake_ai/env/snake_env.hpp"
#include "nn/autograd/value.hpp"
#include "nn/nn/activation.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/sequential.hpp"

#include <algorithm>
#include <memory>
#include <vector>

/**
 * Builds the network: inputs -> hidden -> hidden -> ACTIONS with ReLU between layers
 *
 * Parameters:
 * size_t hidden: Width of both hidden layers
 * size_t inputs: Number of features (default: the cycle features)
 *
 * Returns nn::Sequential: A freshly initialised network
 */
inline nn::Sequential makeNet(size_t hidden, size_t inputs = SnakeEnv::FEATURES) {
	nn::Sequential net;
	net.add(std::make_unique<nn::Linear>(inputs, hidden));
	net.add(std::make_unique<nn::ReLU>());
	net.add(std::make_unique<nn::Linear>(hidden, hidden));
	net.add(std::make_unique<nn::ReLU>());
	net.add(std::make_unique<nn::Linear>(hidden, SnakeEnv::ACTIONS));
	return net;
}

/**
 * Index of the largest output for one state
 *
 * Parameters:
 * nn::Sequential& net: The network
 * const std::vector<float>& features: One state
 *
 * Returns int: Best action
 */
inline int greedy(nn::Sequential& net, const std::vector<float>& features) {
	nn::NoGradGuard noGrad;
	nn::Value out = net.forward(nn::Value::leaf(nn::Tensor({1, features.size()}, features), false));
	const nn::Storage& q = out.data().data();
	return static_cast<int>(std::max_element(q.begin(), q.end()) - q.begin());
}

#endif
