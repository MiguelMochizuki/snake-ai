/**
 * watch.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Plays one game in the terminal with a network loaded from a checkpoint
 */
#include "astar_policy.hpp"
#include "policy_net.hpp"
#include "nn/checkpoint.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace {

const int MAX_STEPS = 100000;
volatile std::sig_atomic_t interrupted = 0;

void onSignal(int) {
	interrupted = 1;
}

/**
 * Draws the board in the game's palette: cyan walls, green snake, red food
 *
 * Parameters:
 * const SnakeEnv& env: Game to draw
 * int step: Steps played
 */
void draw(const SnakeEnv& env, int step) {
	const GameState& s = env.state();
	std::string out = "\033[H";
	const std::string wall = "\033[36m#\033[0m";
	for (int i = 0; i < s.getCols() + 2; ++i) out += wall;
	out += "\n";
	const SnakeSegment* head = s.getSnake().getHead();
	for (int y = 0; y < s.getRows(); ++y) {
		out += wall;
		for (int x = 0; x < s.getCols(); ++x) {
			if (x == head->getX() && y == head->getY()) out += "\033[1;92mO\033[0m";
			else if (s.getSnake().hasPosition(x, y)) out += "\033[32mO\033[0m";
			else if (x == s.getFood().getX() && y == s.getFood().getY()) out += "\033[31m*\033[0m";
			else out += ' ';
		}
		out += wall + "\n";
	}
	for (int i = 0; i < s.getCols() + 2; ++i) out += wall;
	out += "\nScore: " + std::to_string(env.score()) + "   Step: " + std::to_string(step) + "      \n";
	std::fputs(out.c_str(), stdout);
	std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s CHECKPOINT [seed=1] [ms per frame=50] [draw every N steps=1] [astar]\n", argv[0]);
		return 1;
	}
	unsigned seed = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 1;
	int ms = argc > 3 ? std::atoi(argv[3]) : 50;
	int stride = argc > 4 ? std::max(1, std::atoi(argv[4])) : 1;

	const bool astar = argc > 5 && std::string(argv[5]) == "astar";  // checkpoint from imitate_astar
	nn::Sequential net = astar ? makeAstarNet() : makeNet(32);
	SafeAgent agent;
	try {
		nn::load_checkpoint(argv[1], net);
	} catch (const std::exception& e) {
		std::fprintf(stderr, "cannot load %s: %s\n", argv[1], e.what());
		return 1;
	}

	std::signal(SIGINT, onSignal);
	std::fputs("\033[2J\033[?25l", stdout);  // clear, hide cursor
	SnakeEnv env(seed);
	int step = 0;
	while (!env.dead() && step < MAX_STEPS && !interrupted) {
		if (step % stride == 0) {
			draw(env, step);
			std::this_thread::sleep_for(std::chrono::milliseconds(ms));
		}
		env.step(astar ? astarNetAction(net, agent.actionFeatures(env), env.sinceFood() > 250, true) : greedy(net, env.features()));
		++step;
	}
	draw(env, step);
	std::fputs("\033[?25h", stdout);  // show cursor
	std::printf("%s after %d steps with score %d\n", interrupted ? "Stopped" : env.score() >= 397 ? "Board full" : env.dead() ? "Game over" : "Step limit", step, env.score());
	return 0;
}
