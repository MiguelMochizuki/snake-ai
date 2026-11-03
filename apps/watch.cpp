/**
 * watch.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Plays one game with a network loaded from a checkpoint, in the terminal or (with the GUI build)
 * in a window or as off-screen frames drawn by snake-game's renderer
 */
#include "snake_ai/policy/astar_policy.hpp"
#include "snake_ai/policy/policy_net.hpp"
#include "nn/checkpoint.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#ifdef SNAKE_AI_HAVE_GUI
#include "snake_ai/gui/gui_player.hpp"
#endif

namespace {

const int MAX_STEPS = 100000;
volatile std::sig_atomic_t interrupted = 0;

/**
 * Signal handler that asks the game loop to stop so the terminal can be restored
 *
 * Parameters:
 * int: Signal number, unused
 */
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

/**
 * Plays one game in the terminal with a network loaded from a checkpoint
 *
 * Parameters:
 * int argc: Argument count
 * char** argv: argv[1] is the checkpoint, then optional seed, milliseconds per frame, draw stride and "astar";
 * --gui (window) and --frames DIR (off-screen PPM frames) may appear anywhere and need the GUI build
 *
 * Returns int: 0 on success, 1 on bad usage or if the checkpoint cannot be loaded
 */
int main(int argc, char** argv) {
	bool gui = false;
	std::string framesDir;
	std::vector<char*> args{argv[0]};
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a == "--gui") {
			gui = true;
		} else if (a == "--frames" && i + 1 < argc) {
			framesDir = argv[++i];
		} else {
			args.push_back(argv[i]);
		}
	}
	argc = static_cast<int>(args.size());
	argv = args.data();
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s CHECKPOINT [seed=1] [ms per frame=50] [draw every N steps=1] [astar] [--gui | --frames DIR]\n", argv[0]);
		return 1;
	}
#ifndef SNAKE_AI_HAVE_GUI
	if (gui || !framesDir.empty()) {
		std::fprintf(stderr, "this build has no GUI: configure with -DSNAKE_AI_BUILD_GUI=ON\n");
		return 1;
	}
#endif
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

#ifdef SNAKE_AI_HAVE_GUI
	if (gui || !framesDir.empty()) {
		SnakeEnv genv(seed);
		const ActionFn act = [&](SnakeEnv& e) { return astar ? astarNetAction(net, agent.actionFeatures(e), e.sinceFood() > e.loopGuard(), true) : greedy(net, e.features()); };
		const int result = framesDir.empty() ? playInWindow(genv, act, ms, stride, MAX_STEPS) : playToFrames(genv, act, framesDir, stride, MAX_STEPS);
		if (result < 0) {
			std::fprintf(stderr, "cannot open a %s\n", framesDir.empty() ? "window (is a display available?)" : "canvas");
			return 1;
		}
		if (!framesDir.empty()) {
			std::printf("wrote %d frames to %s, final score %d\n", result, framesDir.c_str(), genv.score());
		}
		return 0;
	}
#endif

	std::signal(SIGINT, onSignal);
	std::fputs("\033[2J\033[?25l", stdout);  // clear, hide cursor
	SnakeEnv env(seed);
	int step = 0;
	while (!env.dead() && step < MAX_STEPS && !interrupted) {
		if (step % stride == 0) {
			draw(env, step);
			std::this_thread::sleep_for(std::chrono::milliseconds(ms));
		}
		env.step(astar ? astarNetAction(net, agent.actionFeatures(env), env.sinceFood() > env.loopGuard(), true) : greedy(net, env.features()));
		++step;
	}
	draw(env, step);
	std::fputs("\033[?25h", stdout);  // show cursor
	std::printf("%s after %d steps with score %d\n", interrupted ? "Stopped" : env.score() >= env.maxScore() ? "Board full" : env.dead() ? "Game over" : "Step limit", step, env.score());
	return 0;
}
