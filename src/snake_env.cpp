/**
 * snake_env.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Headless Snake environment implementation
 */
#include "snake_env.hpp"
#include "cycle.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <queue>
#include <stdexcept>
#include <string>

const int SnakeEnv::DX[4] = {0, 0, -1, 1};
const int SnakeEnv::DY[4] = {-1, 1, 0, 0};

SnakeEnv::SnakeEnv(unsigned seed, int size) : state_(size, size, 0, seed), n_(size), visits_(size * size, 0) {
}

int SnakeEnv::turn(int dir, int action) {
	static const int right[4] = {3, 2, 0, 1};  // UP->RIGHT, DOWN->LEFT, LEFT->UP, RIGHT->DOWN
	static const int left[4] = {2, 3, 1, 0};
	return action == TURN_RIGHT ? right[dir] : action == TURN_LEFT ? left[dir] : dir;
}

bool SnakeEnv::blocked(int x, int y) const {
	if (x < 0 || x >= state_.getCols() || y < 0 || y >= state_.getRows()) {
		return true;
	}
	const SnakeSegment& tail = state_.getSnake().getSegments().back();
	const Food& food = state_.getFood();
	// The occupancy map also marks the food, which is not an obstacle
	return state_.isOccupied(x, y) && !(x == tail.getX() && y == tail.getY())
		&& !(x == food.getX() && y == food.getY());
}

MoveInfo SnakeEnv::info(int action) const {
	const SnakeSegment* head = state_.getSnake().getHead();
	const SnakeSegment& tail = state_.getSnake().getSegments().back();
	const Food& food = state_.getFood();
	int d = turn(dir_, action);
	int nx = head->getX() + DX[d], ny = head->getY() + DY[d];
	MoveInfo m{blocked(nx, ny), false, 0, 0};
	if (m.blocked) {
		return m;
	}
	int h = cycleIndex(head->getX(), head->getY());
	int n = cycleIndex(nx, ny);
	m.ahead = cycleDist(h, n);
	m.remaining = cycleDist(n, cycleIndex(food.getX(), food.getY()));
	m.allowed = m.ahead == 1 || m.ahead < cycleDist(h, cycleIndex(tail.getX(), tail.getY()));
	return m;
}

std::vector<float> SnakeEnv::features() const {
	std::vector<float> f;
	for (int a : {STRAIGHT, TURN_RIGHT, TURN_LEFT}) {
		MoveInfo m = info(a);
		f.push_back(m.blocked ? 1.0f : 0.0f);
		f.push_back(m.allowed ? 1.0f : 0.0f);
		f.push_back(m.blocked ? 1.0f : static_cast<float>(m.remaining) / CYCLE_LEN);
	}
	for (int d = 0; d < 4; ++d) {
		f.push_back(dir_ == d ? 1.0f : 0.0f);
	}
	return f;
}

float SnakeEnv::coil(int nx, int ny) const {
	const int W = state_.getCols(), H = state_.getRows();
	const auto& seg = state_.getSnake().getSegments();
	// Point (x, y) of the grid of faces sits at (x + 0.5 + EPS_X, y + 0.5 + EPS_Y); the unequal
	// nudge keeps a diagonal closing edge from passing exactly through a point, where the winding number is undefined. Cast a ray to +x from each
	// point; every edge of the curve that crosses the ray's height to its right adds +1 (down) or -1 (up)
	const double EPS_X = 1.4142e-4, EPS_Y = 1e-4;
	std::vector<int> diff(H * (W + 1), 0);
	auto edge = [&](int x1, int y1, int x2, int y2) {
		if (y1 == y2) {
			return;
		}
		int lo = std::min(y1, y2), hi = std::max(y1, y2);
		int sign = y2 > y1 ? 1 : -1;
		for (int y = std::max(lo, 0); y < hi && y < H; ++y) {
			double xc = x1 + static_cast<double>(x2 - x1) * (y + 0.5 + EPS_Y - y1) / (y2 - y1);
			// Faces x < xc - 0.5 - EPS_X lie left of the crossing
			int c = static_cast<int>(std::ceil(xc - 0.5 - EPS_X));
			if (c > 0) {
				diff[y * (W + 1)] += sign;
				diff[y * (W + 1) + std::min(c, W)] -= sign;
			}
		}
	};
	int px = nx, py = ny;
	for (const SnakeSegment& s : seg) {
		edge(px, py, s.getX(), s.getY());
		px = s.getX();
		py = s.getY();
	}
	edge(px, py, nx, ny);
	int once = 0;
	for (int y = 0; y < H - 1; ++y) {
		int w = 0;
		for (int x = 0; x < W - 1; ++x) {
			w += diff[y * (W + 1) + x];
			once += std::abs(w) == 1;
		}
	}
	return static_cast<float>(once) / ((W - 1) * (H - 1));
}

std::array<float, SnakeEnv::EGO_PER_ACTION> SnakeEnv::ego(int action) const {
	const SnakeSegment* head = state_.getSnake().getHead();
	const SnakeSegment& tail = state_.getSnake().getSegments().back();
	const Food& food = state_.getFood();
	const int W = state_.getCols(), H = state_.getRows();
	int d = turn(dir_, action);
	int nx = head->getX() + DX[d], ny = head->getY() + DY[d];
	if (blocked(nx, ny)) {
		return {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	}
	int closer = std::abs(head->getX() - food.getX()) + std::abs(head->getY() - food.getY())
		- std::abs(nx - food.getX()) - std::abs(ny - food.getY());

	// Flood fill from the target: how much room, and whether the tail is in it
	std::vector<char> seen(W * H, 0);
	std::queue<int> q;
	q.push(ny * W + nx);
	seen[ny * W + nx] = 1;
	int room = 0;
	bool tailReach = false;
	while (!q.empty()) {
		int c = q.front();
		q.pop();
		++room;
		int cx = c % W, cy = c / W;
		tailReach = tailReach || (cx == tail.getX() && cy == tail.getY());
		for (int k = 0; k < 4; ++k) {
			int ax = cx + DX[k], ay = cy + DY[k];
			if (!blocked(ax, ay) && !seen[ay * W + ax]) {
				seen[ay * W + ax] = 1;
				q.push(ay * W + ax);
			}
		}
	}
	int freeCells = W * H - static_cast<int>(state_.getSnake().getLength());
	int open = 0;
	for (int k = 0; k < 4; ++k) {
		open += !blocked(nx + DX[k], ny + DY[k]);
	}
	return {0.0f, closer > 0 ? 1.0f : -1.0f, static_cast<float>(room) / std::max(freeCells, 1),
		tailReach ? 1.0f : 0.0f, std::min(visits_[ny * W + nx], 4) / 4.0f, open / 3.0f,
		coil(nx, ny)};
}

std::vector<float> SnakeEnv::egoFeatures() const {
	std::vector<float> f;
	for (int a : {STRAIGHT, TURN_RIGHT, TURN_LEFT}) {
		std::array<float, EGO_PER_ACTION> e = ego(a);
		f.insert(f.end(), e.begin(), e.end());
	}
	f.push_back(lengthFraction());
	return f;
}

bool SnakeEnv::step(int action) {
	dir_ = turn(dir_, action);
	int before = state_.getScore();
	try {
		state_.moveSnake(dir_);
	} catch (const std::runtime_error& e) {
		dead_ = true;
		// Eating the last free cell fills the board: that is a win, not a crash
		won_ = std::string(e.what()).find("full") != std::string::npos;
		return won_;
	}
	const SnakeSegment* head = state_.getSnake().getHead();
	int cell = head->getY() * state_.getCols() + head->getX();
	++visits_[cell];
	recent_.push_back(cell);
	if (static_cast<int>(recent_.size()) > LOOP_WINDOW) {
		--visits_[recent_.front()];
		recent_.pop_front();
	}
	bool ate = state_.getScore() > before;
	sinceFood_ = ate ? 0 : sinceFood_ + 1;
	if (sinceFood_ >= starveLimit()) {
		dead_ = true;
	}
	return ate;
}
