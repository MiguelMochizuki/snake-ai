/**
 * tiles.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Hamiltonian cycles built from spanning trees of the 2x2 tiles, and the test of whether a snake body still lies on one
 */
#ifndef TILES_HPP
#define TILES_HPP

#include <array>
#include <utility>
#include <vector>

/**
 * The 20x20 board is cut into 10x10 tiles of 2x2 cells. Any spanning tree of the tile grid gives a Hamiltonian cycle of the cells:
 * across a tree edge the cycle crosses the tile border on both lanes, and inside a tile it uses the cell pair along every side
 * that has no tree edge. So a body step between tiles forces that tile edge into the tree, and a step inside a tile forces
 * the edge on that side out of it. A body lies on some such cycle exactly when a spanning tree satisfies all these constraints
 */
const int TILES = 10;
const int TILE_EDGES = 2 * TILES * (TILES - 1);

using Cell = std::pair<int, int>;
using Tree = std::array<char, TILE_EDGES>;

/**
 * Finds a spanning tree whose cycle contains the body as an arc
 *
 * Parameters:
 * const std::vector<Cell>& body: Body cells from head to tail
 * Tree& tree: Receives the tree (1 = edge in the tree)
 *
 * Returns bool: false if no such tree exists
 */
bool treeFor(const std::vector<Cell>& body, Tree& tree);

/**
 * Tells whether two adjacent cells are joined by the tree's cycle
 *
 * Parameters:
 * const Tree& tree: A spanning tree
 * Cell a: A cell
 * Cell b: A cell next to a
 *
 * Returns bool: true if the cycle uses the step a-b
 */
bool onCycle(const Tree& tree, Cell a, Cell b);

/**
 * Tells whether a tree's cycle still contains the body as an arc
 *
 * Parameters:
 * const Tree& tree: A spanning tree
 * const std::vector<Cell>& body: Body cells from head to tail
 *
 * Returns bool: true if every step of the body is on the cycle
 */
bool consistent(const Tree& tree, const std::vector<Cell>& body);

/**
 * Walks the cycle away from the neck, numbering cells by steps from the head
 *
 * Parameters:
 * const Tree& tree: A spanning tree consistent with the body
 * Cell head: Head cell
 * Cell neck: Cell behind the head
 *
 * Returns std::vector<int>: Index y * 20 + x -> steps from the head along the cycle (0 for the head, 399 for the cell behind it)
 */
std::vector<int> cycleOrder(const Tree& tree, Cell head, Cell neck);

#endif
