/**
 * snake_env.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Headless Snake environment implementation
 */
#include "snake_env.hpp"
#include "cycle.hpp"
#include <stdexcept>
#include <string>

namespace {
// Gives up on a game that wanders without eating; above the longest cycle walk to a food
const int STARVE_STEPS = 600;
}  // namespace

const int SnakeEnv::DX[4] = {0, 0, -1, 1};
const int SnakeEnv::DY[4] = {-1, 1, 0, 0};

SnakeEnv::SnakeEnv(unsigned seed) : state_(20, 20, 0, seed) {
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
	const auto& seg = state_.getSnake().getSegments();
	for (size_t i = 0; i + 1 < seg.size(); ++i) {
		if (seg[i].getX() == x && seg[i].getY() == y) {
			return true;
		}
	}
	return false;
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
	bool ate = state_.getScore() > before;
	sinceFood_ = ate ? 0 : sinceFood_ + 1;
	if (sinceFood_ >= STARVE_STEPS) {
		dead_ = true;
	}
	return ate;
}
