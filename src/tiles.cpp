/**
 * tiles.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Tile-tree Hamiltonian cycles
 */
#include "tiles.hpp"
#include <algorithm>
#include <numeric>

namespace {

// Edge between tile (tx, ty) and (tx + 1, ty); t tiles per side
int hEdge(int t, int tx, int ty) { return ty * (t - 1) + tx; }
// Edge between tile (tx, ty) and (tx, ty + 1)
int vEdge(int t, int tx, int ty) { return t * (t - 1) + ty * t + tx; }

/**
 * Tile edge that a step between two adjacent cells crosses, or the edge on the side of the tile the step runs along
 *
 * Parameters:
 * int t: Tiles per side
 * Cell a: A cell
 * Cell b: A cell next to a
 * bool* crosses: Receives true if the step goes between tiles
 *
 * Returns int: Edge id, or -1 if the side lies on the board border
 */
int edgeOf(int t, Cell a, Cell b, bool* crosses) {
	const int ax = a.first, ay = a.second, bx = b.first, by = b.second;
	if (ay == by) {  // horizontal step
		*crosses = ax / 2 != bx / 2;
		if (*crosses) return hEdge(t, std::min(ax, bx) / 2, ay / 2);
		int ty = ay / 2;
		return ay % 2 == 0 ? (ty > 0 ? vEdge(t, ax / 2, ty - 1) : -1) : (ty < t - 1 ? vEdge(t, ax / 2, ty) : -1);
	}
	*crosses = ay / 2 != by / 2;
	if (*crosses) return vEdge(t, ax / 2, std::min(ay, by) / 2);
	int tx = ax / 2;
	return ax % 2 == 0 ? (tx > 0 ? hEdge(t, tx - 1, ay / 2) : -1) : (tx < t - 1 ? hEdge(t, tx, ay / 2) : -1);
}

struct UnionFind {
	std::vector<int> p;
	explicit UnionFind(int size) : p(size) { std::iota(p.begin(), p.end(), 0); }
	int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
	bool unite(int a, int b) {
		a = find(a);
		b = find(b);
		if (a == b) return false;
		p[a] = b;
		return true;
	}
};

}  // namespace

void tileEdgeEnds(int n, int e, int* a, int* b) {
	const int t = n / 2, hEdges = t * (t - 1);
	if (e < hEdges) {
		*a = (e / (t - 1)) * t + e % (t - 1);
		*b = *a + 1;
	} else {
		*a = e - hEdges;
		*b = *a + t;
	}
}

bool onCycle(const Tree& tree, int n, Cell a, Cell b) {
	bool crosses;
	int e = edgeOf(n / 2, a, b, &crosses);
	if (crosses) return tree[e];
	return e < 0 || !tree[e];
}

bool consistent(const Tree& tree, int n, const std::vector<Cell>& body) {
	for (size_t i = 0; i + 1 < body.size(); ++i) {
		if (!onCycle(tree, n, body[i], body[i + 1])) return false;
	}
	return true;
}

bool treeFor(const std::vector<Cell>& body, int n, Tree& tree) {
	const int t = n / 2, edges = tileEdgeCount(n);
	std::vector<char> forced(edges, 0);  // 0 free, 1 in the tree, 2 out of it
	for (size_t i = 0; i + 1 < body.size(); ++i) {
		bool crosses;
		int e = edgeOf(t, body[i], body[i + 1], &crosses);
		if (e < 0) continue;
		char want = crosses ? 1 : 2;
		if (forced[e] && forced[e] != want) return false;
		forced[e] = want;
	}
	UnionFind uf(t * t);
	tree.assign(edges, 0);
	int joined = 0;
	for (int e = 0; e < edges; ++e) {
		if (forced[e] != 1) continue;
		int a, b;
		tileEdgeEnds(n, e, &a, &b);
		if (!uf.unite(a, b)) return false;  // required edges close a loop
		tree[e] = 1;
		++joined;
	}
	for (int e = 0; e < edges; ++e) {
		if (forced[e] != 0) continue;
		int a, b;
		tileEdgeEnds(n, e, &a, &b);
		if (uf.unite(a, b)) {
			tree[e] = 1;
			++joined;
		}
	}
	return joined == t * t - 1;
}

std::vector<int> cycleOrder(const Tree& tree, int n, Cell head, Cell neck) {
	static const int DX[4] = {0, 0, -1, 1}, DY[4] = {-1, 1, 0, 0};
	std::vector<int> order(n * n, -1);
	Cell prev = neck, cur = head;
	for (int k = 0; k < n * n; ++k) {
		order[cur.second * n + cur.first] = k;
		Cell next = prev;
		for (int d = 0; d < 4; ++d) {
			Cell c{cur.first + DX[d], cur.second + DY[d]};
			if (c.first < 0 || c.first >= n || c.second < 0 || c.second >= n || c == prev) continue;
			if (onCycle(tree, n, cur, c)) {
				next = c;
				break;
			}
		}
		prev = cur;
		cur = next;
	}
	return order;
}
