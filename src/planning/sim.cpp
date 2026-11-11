/**
 * sim.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Rollout simulator
 */
#include "snake_ai/planning/sim.hpp"

Sim::Sim(const Snapshot& s, unsigned seed)
	: size_(s.size), dir_(s.dir), body_(s.body.begin(), s.body.end()), food_(s.food), sinceFood_(s.sinceFood),
	  occ_(s.size * s.size, 0), rng_(seed) {
	for (const Cell& c : body_) occ_[c.second * size_ + c.first] = 1;
}

void Sim::spawnFood() {
	// ponytail: collects the free cells on every eat, O(cells); an incremental free list if rollouts dominate the profile
	std::vector<int> free;
	for (int i = 0; i < size_ * size_; ++i) {
		if (!occ_[i]) free.push_back(i);
	}
	int pick = free[std::uniform_int_distribution<size_t>(0, free.size() - 1)(rng_)];
	food_ = {pick % size_, pick / size_};
}

bool Sim::step(int action) {
	dir_ = SnakeEnv::turn(dir_, action);
	++steps_;
	const Cell head = body_.front();
	const Cell next{head.first + SnakeEnv::DX[dir_], head.second + SnakeEnv::DY[dir_]};
	if (next.first < 0 || next.first >= size_ || next.second < 0 || next.second >= size_) {
		dead_ = true;
		return false;
	}
	const bool ate = next == food_;
	// The tail cell is free for the head only when the snake does not grow, as in GameState::moveSnake
	if (!ate && occ_[next.second * size_ + next.first] && next != body_.back()) {
		dead_ = true;
		return false;
	}
	if (!ate) {
		const Cell tail = body_.back();
		occ_[tail.second * size_ + tail.first] = 0;
		body_.pop_back();
	}
	body_.push_front(next);
	occ_[next.second * size_ + next.first] = 1;
	if (ate) {
		++eaten_;
		sinceFood_ = 0;
		if (static_cast<int>(body_.size()) == size_ * size_) {
			dead_ = true;
			won_ = true;
			return true;
		}
		spawnFood();
		return true;
	}
	++sinceFood_;
	if (sinceFood_ >= 3 * size_ * size_ / 2) dead_ = true;
	return false;
}

Snapshot Sim::snapshot() const {
	return {size_, dir_, std::vector<Cell>(body_.begin(), body_.end()), food_, sinceFood_};
}
