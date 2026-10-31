# snake-ai

Agents that play [snake-game](https://github.com/MiguelMochizuki/snake-game), trained with the neural network library [nn-cpp](https://github.com/MiguelMochizuki/nn-cpp). The chosen approach is imitation: a small network learns to copy an expert player. The best network copies a safe A* player, one that only makes moves that keep the body on a Hamiltonian cycle of 2x2 tiles, so it can never trap itself. It scores 397 points (full board) in about 15,800 steps with no deaths in 12 games, and the same weights fill 10x10 and 40x40 boards without retraining.

## Overview

- Environment (`env/`): a headless environment over snake-game's `GameState`, with relative actions (straight, right, left), a runtime board size and two feature families.
- Planning (`planning/`): Hamiltonian cycles built from spanning trees of 2x2 tiles, a union-find test of whether the body still lies on some cycle, and three playing modes (A\*, shortcuts, cycle) that all preserve that property.
- Teachers (`teachers/`): the cycle player with safe shortcuts, and the evolved heuristic used as a baseline.
- Policy (`policy/`): the network architecture and the policy that imitates the safe A\* player.

Results and design decisions are in [docs/REPORT.md](docs/REPORT.md); the raw logs of the final runs are in [docs/results/](docs/results/).

## Project Organization

```
apps/           executables: compare, evolve, imitate_astar, selector, watch
cmake/          FetchDoctest.cmake (fetches doctest for the tests)
deps/           submodules: snake-game (game logic) and nn-cpp (neural network)
docs/           REPORT.md and results/ (logs of the final runs)
experiments/    bfs_safe, safe_modes, scale_test
include/snake_ai/
  env/          headless environment and the fixed 20x20 Hamiltonian cycle
  planning/     tile-tree cycles and the A*, shortcut and cycle modes
  teachers/     cycle player with shortcuts, and the evolved heuristic
  policy/       network architecture and the policy imitating safe A*
src/            implementations, mirroring include/snake_ai
tests/          unit tests (doctest)
models/         trained checkpoints
```

## Tools & Stack

C++17, CMake, [nn-cpp](https://github.com/MiguelMochizuki/nn-cpp) (tensors, autograd, `Sequential`, Adam), [snake-game](https://github.com/MiguelMochizuki/snake-game) (game logic only, no terminal code) and [doctest](https://github.com/doctest/doctest) for the tests. CI runs on GitHub Actions.

## Build

Requirements: a C++17 compiler and CMake 3.16+.

```bash
git clone --recursive git@github.com:MiguelMochizuki/snake-ai.git
cd snake-ai
make test      # builds everything and runs the tests; binaries land in build/bin
```

Without `make`: `cmake -S . -B build && cmake --build build -j`. The options `SNAKE_AI_BUILD_APPS`, `SNAKE_AI_BUILD_EXPERIMENTS` and `SNAKE_AI_BUILD_TESTS` switch each part on or off.

Both dependencies are git submodules in `deps/`: `snake-game` at v2.0.0 plus the module reorganization (headers under `snake_game/`) and `nn-cpp` at a pinned commit.

## Usage

```bash
./build/bin/compare expert           # Hamiltonian cycle player
./build/bin/compare imitation        # trains the cycle-imitation network, writes imitation.ckpt, evaluates
./build/bin/compare dqn              # DQN with the cycle features
./build/bin/compare dqn-ego GENOME   # DQN with invariant features and shaping
./build/bin/compare ga GENOME        # evaluates the evolved heuristic
./build/bin/evolve GENOME [steps] [nocoil]   # genetic algorithm
./build/bin/imitate_astar            # trains the safe-A* imitation network, writes imitation_astar.ckpt
./build/bin/selector                 # learned mode selector
./build/bin/watch models/imitation_astar.ckpt 1 120 1 astar   # watch a network play in the terminal
./build/bin/safe_modes [games]       # A*, shortcut and cycle modes with the safety guarantee
./build/bin/scale_test               # 10x10, 20x20 and 40x40 boards with the 20x20-trained network
./build/bin/bfs_safe [games]         # shortest-path experiment
```

`GENOME` is the weights file written by `evolve`; `steps` is the per-game step cap during evolution (default 4000); `nocoil` zeroes the winding-number term for comparison. `watch` takes `checkpoint [seed] [ms per frame] [draw every N steps] [astar]`. To see output while training, run under `stdbuf -oL`, since stdout is buffered when redirected.

## Key Results

12 games on 20x20, same seeds, 397 = full board (details in [docs/REPORT.md](docs/REPORT.md)):

| Player | Points | Steps per game | Deaths |
|---|---|---|---|
| Hamiltonian cycle | 397 | 39,574 | 0/12 |
| Safe A\* (teacher) | 397 | 15,792 | 0/12 |
| Network imitating safe A\* | 397 | 15,792 | 0/12 |

The same network reaches the full board on 10x10 and 40x40 with exactly the teacher's step counts.

## Documentation Conventions

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).
