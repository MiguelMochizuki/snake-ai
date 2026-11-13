/**
 * test_es.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the evolution strategies module
 */

#include "doctest/doctest.h"
#include "snake_ai/policy/astar_policy.hpp"
#include "snake_ai/policy/es.hpp"

#include <stdexcept>

namespace {

/** Squared distance of w from the all-ones vector, the quadratic the ES tests minimise */
double quadratic(const std::vector<float>& w) {
	double c = 0;
	for (float x : w) c += (x - 1.0) * (x - 1.0);
	return c;
}

/** Runs generations of esStep from zeros and returns the final weights */
std::vector<float> runQuadratic(const EsConfig& cfg, int generations) {
	std::vector<float> theta(8, 0.0f);
	for (int g = 0; g < generations; ++g) theta = esStep(theta, cfg, g, quadratic);
	return theta;
}

}  // namespace

TEST_CASE("get/set weights round trip leaves the outputs unchanged") {
	nn::Sequential net = makeAstarNet();
	std::vector<float> w = getWeights(net);
	CHECK(w.size() == 1571);  // 12*32+32 + 32*32+32 + 32*3+3
	std::vector<float> in(SafeAgent::ACTION_FEATURES, 0.3f);
	nn::NoGradGuard noGrad;
	auto forward = [&] { return net.forward(nn::Value::leaf(nn::Tensor({1, in.size()}, in), false)).data().data(); };
	const nn::Storage before = forward();
	setWeights(net, w);
	CHECK(forward() == before);
	// A copy of the weights in a second network makes it behave the same
	nn::Sequential other = makeAstarNet();
	setWeights(other, w);
	CHECK(getWeights(other) == w);
	w.pop_back();
	CHECK_THROWS_AS(setWeights(net, w), std::invalid_argument);
}

TEST_CASE("esStep minimises a quadratic") {
	EsConfig cfg;
	cfg.sigma = 0.1f;
	cfg.lr = 0.1f;
	cfg.population = 32;
	const double initial = quadratic(std::vector<float>(8, 0.0f));
	CHECK(quadratic(runQuadratic(cfg, 200)) < 0.1 * initial);
}

TEST_CASE("esStep is deterministic and does not depend on the thread count") {
	EsConfig cfg;
	cfg.sigma = 0.1f;
	cfg.lr = 0.1f;
	const std::vector<float> a = runQuadratic(cfg, 5);
	CHECK(runQuadratic(cfg, 5) == a);
	cfg.threads = 4;
	CHECK(runQuadratic(cfg, 5) == a);
}

TEST_CASE("esStep rejects a population that cannot be paired and a non-positive sigma") {
	EsConfig cfg;
	cfg.population = 3;
	CHECK_THROWS_AS(esStep({0.0f}, cfg, 0, quadratic), std::invalid_argument);
	cfg.population = 32;
	cfg.sigma = 0.0f;
	CHECK_THROWS_AS(esStep({0.0f}, cfg, 0, quadratic), std::invalid_argument);
}

TEST_CASE("esStep leaves the weights alone when every candidate costs the same") {
	// A flat landscape carries no information: tied candidates must not push theta anywhere
	EsConfig cfg;
	const std::vector<float> theta(8, 0.5f);
	const std::vector<float> next = esStep(theta, cfg, 3, [](const std::vector<float>&) { return 7.0; });
	for (size_t j = 0; j < theta.size(); ++j) CHECK(next[j] == doctest::Approx(theta[j]).epsilon(1e-6));
}
