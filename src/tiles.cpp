/**
 * tiles.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Tile-tree Hamiltonian cycles
 */
#include "tiles.hpp"
#include <numeric>

namespace {

const int W = 20;
const int H_EDGES = TILES * (TILES - 1);

// Edge between tile (tx, ty) and (tx + 1, ty)
int hEdge(int tx, int ty) { return ty * (TILES - 1) + tx; }
// Edge between tile (tx, ty) and (tx, ty + 1)
int vEdge(int tx, int ty) { return H_EDGES + ty * TILES + tx; }

/**
 * Tile edge that a step between two adjacent cells crosses, or the edge on the side of the tile the step runs along
 *
 * Parameters:
 * Cell a: A cell
 * Cell b: A cell next to a
 * bool* crosses: Receives true if the step goes between tiles
 *
 * Returns int: Edge id, or -1 if the side lies on the board border
 */
int edgeOf(Cell a, Cell b, bool* crosses) {
	const int ax = a.first, ay = a.second, bx = b.first, by = b.second;
	if (ay == by) {  // horizontal step
		*crosses = ax / 2 != bx / 2;
		if (*crosses) return hEdge(std::min(ax, bx) / 2, ay / 2);
		int ty = ay / 2;
		return ay % 2 == 0 ? (ty > 0 ? vEdge(ax / 2, ty - 1) : -1) : (ty < TILES - 1 ? vEdge(ax / 2, ty) : -1);
	}
	*crosses = ay / 2 != by / 2;
	if (*crosses) return vEdge(ax / 2, std::min(ay, by) / 2);
	int tx = ax / 2;
	return ax % 2 == 0 ? (tx > 0 ? hEdge(tx - 1, ay / 2) : -1) : (tx < TILES - 1 ? hEdge(tx, ay / 2) : -1);
}

struct UnionFind {
	std::vector<int> p;
	UnionFind() : p(TILES * TILES) { std::iota(p.begin(), p.end(), 0); }
	int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
	bool unite(int a, int b) {
		a = find(a);
		b = find(b);
		if (a == b) return false;
		p[a] = b;
		return true;
	}
};

void endpoints(int e, int* a, int* b) {
	if (e < H_EDGES) {
		*a = (e / (TILES - 1)) * TILES + e % (TILES - 1);
		*b = *a + 1;
	} else {
		*a = (e - H_EDGES);
		*b = *a + TILES;
	}
}

}  // namespace

bool onCycle(const Tree& tree, Cell a, Cell b) {
	bool crosses;
	int e = edgeOf(a, b, &crosses);
	if (crosses) return tree[e];
	return e < 0 || !tree[e];
}

bool consistent(const Tree& tree, const std::vector<Cell>& body) {
	for (size_t i = 0; i + 1 < body.size(); ++i) {
		if (!onCycle(tree, body[i], body[i + 1])) return false;
	}
	return true;
}

bool treeFor(const std::vector<Cell>& body, Tree& tree) {
	std::array<char, TILE_EDGES> forced{};  // 0 free, 1 in the tree, 2 out of it
	for (size_t i = 0; i + 1 < body.size(); ++i) {
		bool crosses;
		int e = edgeOf(body[i], body[i + 1], &crosses);
		if (e < 0) continue;
		char want = crosses ? 1 : 2;
		if (forced[e] && forced[e] != want) return false;
		forced[e] = want;
	}
	UnionFind uf;
	tree.fill(0);
	int joined = 0;
	for (int e = 0; e < TILE_EDGES; ++e) {
		if (forced[e] != 1) continue;
		int a, b;
		endpoints(e, &a, &b);
		if (!uf.unite(a, b)) return false;  // required edges close a loop
		tree[e] = 1;
		++joined;
	}
	for (int e = 0; e < TILE_EDGES; ++e) {
		if (forced[e] != 0) continue;
		int a, b;
		endpoints(e, &a, &b);
		if (uf.unite(a, b)) {
			tree[e] = 1;
			++joined;
		}
	}
	return joined == TILES * TILES - 1;
}

std::vector<int> cycleOrder(const Tree& tree, Cell head, Cell neck) {
	static const int DX[4] = {0, 0, -1, 1}, DY[4] = {-1, 1, 0, 0};
	std::vector<int> order(W * W, -1);
	Cell prev = neck, cur = head;
	for (int k = 0; k < W * W; ++k) {
		order[cur.second * W + cur.first] = k;
		Cell next = prev;
		for (int d = 0; d < 4; ++d) {
			Cell c{cur.first + DX[d], cur.second + DY[d]};
			if (c.first < 0 || c.first >= W || c.second < 0 || c.second >= W || c == prev) continue;
			if (onCycle(tree, cur, c)) {
				next = c;
				break;
			}
		}
		prev = cur;
		cur = next;
	}
	return order;
}
