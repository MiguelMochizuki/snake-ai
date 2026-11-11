/**
 * es.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Evolution strategies over the flat weight vector of a network: antithetic sampling and rank-shaped fitness
 */
#ifndef ES_HPP
#define ES_HPP

#include "nn/nn/sequential.hpp"

#include <functional>
#include <vector>

/**
 * Settings of one evolution strategies update
 *
 * Attributes:
 * int population: Candidates per generation; even, because every noise vector is used with both signs
 * float sigma: Standard deviation of the noise added to the weights
 * float lr: Step size of the update
 * unsigned seed: Noise depends only on (seed, generation)
 * int threads: Worker threads that evaluate candidates; the cost function must be safe to call from several threads when above 1,
 * and the result does not depend on it
 */
struct EsConfig {
	int population = 32;
	float sigma = 0.02f;
	float lr = 0.05f;
	unsigned seed = 1;
	int threads = 1;
};

/**
 * Parameters:
 * const nn::Sequential& net: A network
 *
 * Returns std::vector<float>: Every trainable weight, flat, in parameters() order
 */
std::vector<float> getWeights(const nn::Sequential& net);

/**
 * Overwrites every trainable weight of a network
 *
 * Parameters:
 * nn::Sequential& net: The network
 * const std::vector<float>& w: Weights in parameters() order
 *
 * Throws std::invalid_argument: If w does not have exactly as many values as the network has weights
 */
void setWeights(nn::Sequential& net, const std::vector<float>& w);

/**
 * One generation: samples antithetic candidates around theta, scores each with cost (lower is better), shapes the scores by rank
 * into [-0.5, 0.5] and moves theta towards the noise of the cheap candidates
 *
 * Parameters:
 * const std::vector<float>& theta: Current weights
 * const EsConfig& cfg: Population, noise, step and seed
 * int generation: Index of this generation, so each one draws different noise
 * const std::function<double(const std::vector<float>&)>& cost: Cost of a candidate's weights
 *
 * Returns std::vector<float>: The new weights
 *
 * Throws std::invalid_argument: If the population is not even and at least 2, or sigma is not positive
 */
std::vector<float> esStep(const std::vector<float>& theta, const EsConfig& cfg, int generation, const std::function<double(const std::vector<float>&)>& cost);

#endif
