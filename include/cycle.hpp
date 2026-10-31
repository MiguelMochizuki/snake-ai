/**
 * cycle.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Hamiltonian cycle over the 20x20 board
 */
#ifndef CYCLE_HPP
#define CYCLE_HPP

/**
 * Number of cells on the board and in the cycle
 */
const int CYCLE_LEN = 400;

/**
 * Position of a cell along the cycle. The cycle runs along the top row, zigzags through
 * columns 1..19 of rows 1..19, and returns up column 0, so every cell is visited once
 *
 * Parameters:
 * int x: Column, 0..19
 * int y: Row, 0..19
 *
 * Returns int: Index in 0..CYCLE_LEN-1
 */
int cycleIndex(int x, int y);

/**
 * Steps forward along the cycle from one index to another
 *
 * Parameters:
 * int from: Start index
 * int to: End index
 *
 * Returns int: 0..CYCLE_LEN-1
 */
inline int cycleDist(int from, int to) {
	return ((to - from) % CYCLE_LEN + CYCLE_LEN) % CYCLE_LEN;
}

#endif
