/**
 * gui_player.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Plays a SnakeEnv game while drawing it through snake-game's SDL renderer, in a window or off-screen.
 * Header-only and available only when built with SNAKE_AI_BUILD_GUI. It does not use GameEngine: the game loop is ours.
 */
#ifndef SNAKE_AI_GUI_PLAYER_HPP
#define SNAKE_AI_GUI_PLAYER_HPP

#include "snake_ai/env/snake_env.hpp"
#include "snake_game/gui/canvas.hpp"
#include "snake_game/gui/frame_dumper.hpp"
#include "snake_game/gui/sdl_app.hpp"
#include "snake_game/gui/sdl_renderer.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>

// Picks the relative action (0 straight, 1 right, 2 left) for the current state of the game
using ActionFn = std::function<int(SnakeEnv&)>;

namespace gui_player {

const int WIDTH = 640;
const int HEIGHT = 720;

/**
 * The state to draw: a copy whose best score is the score reached so far in this run. When the last food fills the board the game
 * throws before it counts that food, so the score is topped up to SnakeEnv::score() and the final frame reads the full score
 *
 * Parameters:
 * const SnakeEnv& env: Game being played
 *
 * Returns GameState: The state to hand to the renderer
 */
inline GameState drawable(const SnakeEnv& env) {
	GameState state = env.state();
	while (state.getScore() < env.score()) state.incrementScore();
	state.setBestScore(state.getScore());
	return state;
}

}  // namespace gui_player

/**
 * Plays a game off-screen and writes frame-00000.ppm, frame-00001.ppm and so on to a directory
 *
 * A frame is written for the first state, then every stride steps, and always for the last state
 *
 * Parameters:
 * SnakeEnv& env: Game to play, drawn from its current state
 * const ActionFn& act: Chooses each move
 * const std::string& dir: Existing directory for the frames
 * int stride: Steps between frames
 * int maxSteps: Step limit
 *
 * Returns int: Number of frames written, or -1 if no canvas can be created
 */
inline int playToFrames(SnakeEnv& env, const ActionFn& act, const std::string& dir, int stride, int maxSteps) {
	std::unique_ptr<Canvas> canvas = Canvas::offscreen(gui_player::WIDTH, gui_player::HEIGHT);
	if (!canvas) {
		return -1;
	}
	stride = std::max(1, stride);
	SdlApp app(*canvas, false);
	SdlRenderer sdl(*canvas, app);
	FrameDumper dumper(sdl, *canvas, dir, 1);  // the stride is applied here, so every frame handed over is written
	int step = 0;
	sdl.clear();
	dumper.render(gui_player::drawable(env));  // frame 0
	int written = 1;
	bool lastSaved = true;
	while (!env.dead() && step < maxSteps) {
		env.step(act(env));
		++step;
		if (step % stride == 0) {
			sdl.clear();
			dumper.render(gui_player::drawable(env));
			++written;
			lastSaved = true;
		} else {
			lastSaved = false;
		}
	}
	if (!lastSaved) {
		sdl.clear();
		sdl.render(gui_player::drawable(env));  // draws the final state without counting it as a stride frame
		char name[32];
		std::snprintf(name, sizeof name, "/frame-%05d.ppm", written);
		canvas->writePpm(dir + name);
		++written;
	}
	return written;
}

/**
 * Plays a game in a window and keeps the final board on screen until the window is closed
 *
 * Parameters:
 * SnakeEnv& env: Game to play, drawn from its current state
 * const ActionFn& act: Chooses each move
 * int ms: Milliseconds to wait after each drawn frame
 * int stride: Steps between drawn frames
 * int maxSteps: Step limit
 *
 * Returns int: 0 on success, -1 if no window can be created
 */
inline int playInWindow(SnakeEnv& env, const ActionFn& act, int ms, int stride, int maxSteps) {
	std::unique_ptr<Canvas> canvas = Canvas::window("Snake AI", gui_player::WIDTH, gui_player::HEIGHT);
	if (!canvas) {
		return -1;
	}
	stride = std::max(1, stride);
	SdlApp app(*canvas, true);
	SdlRenderer sdl(*canvas, app);
	int step = 0;
	while (!env.dead() && step < maxSteps && !app.closeRequested()) {
		if (step % stride == 0) {
			sdl.clear();
			sdl.render(gui_player::drawable(env));
			sdl.pause(ms);
		}
		env.step(act(env));
		++step;
	}
	sdl.clear();
	sdl.render(gui_player::drawable(env));
	while (!app.closeRequested()) {
		app.pump(100);
	}
	return 0;
}

#endif
