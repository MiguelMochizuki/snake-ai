/**
 * snake_env.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Headless Snake environment implementation
 */
#include "snake_env.hpp"
#include <stdexcept>

namespace {
// Gives up on a game that wanders without eating, so loops cannot run forever
const int STARVE_STEPS = 400;
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

std::vector<float> SnakeEnv::features() const {
	const SnakeSegment* head = state_.getSnake().getHead();
	const int hx = head->getX(), hy = head->getY();
	std::vector<float> f;
	for (int a : {STRAIGHT, TURN_RIGHT, TURN_LEFT}) {
		int d = turn(dir_, a);
		f.push_back(blocked(hx + DX[d], hy + DY[d]) ? 1.0f : 0.0f);
	}
	for (int d = 0; d < 4; ++d) {
		f.push_back(dir_ == d ? 1.0f : 0.0f);
	}
	const Food& food = state_.getFood();
	f.push_back(food.getY() < hy ? 1.0f : 0.0f);
	f.push_back(food.getY() > hy ? 1.0f : 0.0f);
	f.push_back(food.getX() < hx ? 1.0f : 0.0f);
	f.push_back(food.getX() > hx ? 1.0f : 0.0f);
	return f;
}

void SnakeEnv::step(int action) {
	dir_ = turn(dir_, action);
	int before = state_.getScore();
	try {
		state_.moveSnake(dir_);
	} catch (const std::runtime_error&) {
		dead_ = true;
		return;
	}
	sinceFood_ = state_.getScore() > before ? 0 : sinceFood_ + 1;
	if (sinceFood_ >= STARVE_STEPS) {
		dead_ = true;
	}
}
