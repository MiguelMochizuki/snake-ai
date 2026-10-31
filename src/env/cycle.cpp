/**
 * cycle.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Hamiltonian cycle table
 */
#include "snake_ai/env/cycle.hpp"
#include <array>

namespace {

/**
 * Builds the Hamiltonian cycle of the 20x20 board as a table of cell positions
 *
 * Returns std::array<int, CYCLE_LEN>: Position along the cycle for each cell, indexed by y * 20 + x
 */
std::array<int, CYCLE_LEN> buildTable() {
	std::array<int, CYCLE_LEN> t{};
	int n = 0;
	for (int x = 0; x < 20; ++x) {
		t[0 * 20 + x] = n++;
	}
	for (int y = 1; y < 20; ++y) {
		if (y % 2) {
			for (int x = 19; x >= 1; --x) t[y * 20 + x] = n++;
		} else {
			for (int x = 1; x <= 19; ++x) t[y * 20 + x] = n++;
		}
	}
	for (int y = 19; y >= 1; --y) {
		t[y * 20] = n++;
	}
	return t;
}

}  // namespace

int cycleIndex(int x, int y) {
	static const std::array<int, CYCLE_LEN> table = buildTable();
	return table[y * 20 + x];
}
