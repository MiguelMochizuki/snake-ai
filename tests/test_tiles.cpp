/**
 * test_tiles.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for tile-tree Hamiltonian cycles
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "snake_ai/planning/tiles.hpp"

#include <set>

TEST_CASE("a two-cell body fits a tree on several board sizes") {
	for (int n : {10, 20, 40}) {
		Tree tree;
		std::vector<Cell> body{{1, 0}, {0, 0}};
		REQUIRE(treeFor(body, n, tree));
		CHECK(consistent(tree, n, body));
	}
}

TEST_CASE("cycleOrder visits every cell once, stepping to a neighbour each time") {
	const int n = 10;
	Tree tree;
	std::vector<Cell> body{{1, 0}, {0, 0}};
	REQUIRE(treeFor(body, n, tree));
	std::vector<int> order = cycleOrder(tree, n, body[0], body[1]);
	REQUIRE(order.size() == static_cast<size_t>(n * n));
	CHECK(std::set<int>(order.begin(), order.end()).size() == static_cast<size_t>(n * n));
	CHECK(order[0 * n + 1] == 0);
	CHECK(order[0 * n + 0] == n * n - 1);
}

TEST_CASE("a body that closes a loop around a tile has no cycle") {
	// A 2x2 block of the snake's own body cannot be threaded by a Hamiltonian cycle through the neck
	Tree tree;
	std::vector<Cell> body{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 2}, {1, 2}, {2, 2}, {2, 1}, {2, 0}};
	CHECK_FALSE(treeFor(body, 10, tree));
}
