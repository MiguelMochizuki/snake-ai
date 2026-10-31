# Report: AI for the snake game

## Summary

Update: the tile scheme (section below) gave a player that is faster than the cycle and just as safe: the guaranteed A\* scores 397 points in about 15,800 steps, against 39,600 for the cycle. A network trained to imitate that safe A\* (section "Imitating the safe A\*") reproduces the teacher: 397 points, 15,792 steps, 0 deaths in 12 games. The chosen imitation is now that one; the cycle network stays as the first version.

We compared four players on `snake-game` v2.0.0, 20x20 board. Imitation was chosen: a 13→32→32→3 network trained to copy the Hamiltonian cycle player reaches 393.8 points on average out of a maximum of 397 (full board). The two approaches that learn without a teacher fell well behind.

## How we measured

- 30 fresh games per player, with the same food seeds (100000 to 100029), never used in training.
- Cap of 60,000 steps per game. A game also ends by starvation: 600 steps without eating count as a death.
- Maximum score: 397, the 400-cell board with the initial snake of 3. Filling the board counts as a win.
- Each number comes from one training run and one set of 30 games. Training is not deterministic: retraining the imitation network to save the model gave a mean of 397.0 (0 deaths) instead of 393.8. We did not repeat with several seeds.

## Results

| Player | Mean | Min | Max | Deaths |
|---|---|---|---|---|
| Expert (Hamiltonian cycle with safe shortcuts) | 397.0 | 397 | 397 | 0/30 |
| Imitation (13 features, MLP 13→32→32→3, Adam) | 393.8 (other run: 397.0) | 301 | 397 | 1/30 (other run: 0/30) |
| Heuristic evolved by GA, with winding term, 20k steps during evolution | 263.5 | 173 | 369 | 30/30 |
| Heuristic evolved by GA, without the term, 20k steps during evolution | 251.7 | 122 | 385 | 30/30 |
| DQN with invariant features and heuristic shaping | 38.5 | 14 | 74 | 30/30 |
| DQN with the cycle features | 7.3 | 0 | 24 | 30/30 |

First version, with another expert and another test set (100 games): the BFS expert averaged 64.1 and the network, with 11 local features and SGD, 27.5.

## Decisions and rationale

### 1. Replace the BFS expert with the Hamiltonian cycle

BFS takes the snake to the food by the shortest path, but it ends up trapping itself (mean 64). The Hamiltonian cycle visits every cell in a fixed order. With shortcuts that only move forward and never pass the tail, the body stays in cycle order and the snake never traps itself. It won all 30 games.
Consequence: the teacher became perfect, so the ceiling for imitation rose from 64 to 397.

### 2. Give the network cycle-relative features

Per action (straight, right, left) the network receives: whether it is blocked, whether the cycle allows the move, and the remaining distance to the food along the cycle (divided by 400). Plus the current heading as one-hot, 13 values in total. The expert's decision (smallest distance among the allowed moves) becomes a simple function of these features, easy to learn: the loss reaches 0.0001.
Consequence: with 11 local features (no cycle), imitation stayed at 27.5.

### 3. Choose imitation

- It reaches 98% of the expert's score (393.8 of 397), against 66% for the best teacher-free path (GA heuristic, 263.5) and 10% for the best DQN (38.5).
- It trains in about 1 minute. DQN spends 300 thousand steps and does not pass 40 points.
- It is supervised, so it uses `nn-cpp` without needing reinforcement learning components that the library does not have.

### 4. What we did not adopt, and why

- DQN: it learns not to die immediately, but not to plan. With gamma 0.95 the horizon is about 20 steps, and the food is often more than 100 steps away along the cycle. Invariant features, heuristic shaping, a death penalty proportional to size and a loop penalty raised it from 7.3 to 38.5, still far off. A larger gamma, more steps and Double DQN were not tested.
- Genetic algorithm over the heuristic: it reached about 260 points, but every game ends in death. The heuristic is invariant to rotation and scale and has weights that depend on snake size, which was the goal, but it does not give the guarantee the cycle gives.
- Winding-number term (board points with winding number 1 with respect to the closed curve formed by the body): it improved the mean by 3% to 5%, in the same direction in both rounds, but with one evolution per configuration and scores ranging from 173 to 369, we cannot claim the effect is real. The computation was checked against a brute-force angle sum, with no divergence in 2,200 states.

### 5. Dependencies as git submodules

`snake-game` (tag v2.0.0) and `nn-cpp` (pinned commit) live in `deps/`. They pin the versions, and the project builds without the other two repositories next to it. We tried FetchContent first and dropped it at the user's request.

## Limitations

- The cycle features hold for the 20x20 board and the chosen cycle. The cycle imitation is not invariant to rotation or scale, unlike the heuristic. For another size a different cycle must be generated and the network retrained.
- The network learns the expert's rule, it does not discover a strategy. The best it can do is what the expert already does.
- The single death of the imitation (301 points) happened in a long game. We did not investigate the cause.
- One training run and 30 games per player. Small differences (for example, between the heuristics) are not conclusive.

## Attempt: shorter paths to the food

Question: after imitating the cycle, can the network zigzag and minimize the distance to the food? First we measured the teacher's pace (mean of 20 games):

| Stretch | Steps per food |
|---|---|
| up to 25 points | 26 |
| 25 to 100 | 36 |
| 100 to 200 | 74 |
| 200 to 300 | 120 |
| 300 to 397 | 146 |

A full game takes about 37 thousand steps. We tried two ways to shorten it:

1. Shortest safe route inside the cycle rule. Dynamic programming over cycle indices, accepting only routes that advance along the cycle and stay before the tail. Safe (0 deaths in 30 games), but no gain: 37,434 steps per game against 37,500. Only the start improves (26 to 23 steps per food up to 25 points). With a long snake the free stretch ahead is short, and the food often lands on a cell skipped by an earlier shortcut, which sits behind the tail on the cycle; the snake waits for the tail to move. The limit comes from the cycle's own guarantee. There was also a mistake of ours along the way: without handling food behind the tail, the route crossed the body and the snake died (mean 35.5). We reverted that code, since it brought no gain.
2. Shortest route by BFS with a safety test (`experiments/bfs_safe.cpp`). It follows BFS to the food if a simulation shows the tail stays reachable after eating; otherwise it chases the tail. In 10 games: 392.4 points on average with 23,204 steps (38% fewer than the cycle), but all 10 end without filling the board: they land between 387 and 395 and starve (600 steps without eating) when few cells remain, because the test always rejects the path to the food. Relaxing the test is much worse: accepting a path longer than 360 gave 362.7 points and collisions; accepting after 150 steps without eating gave 194.4 and collisions.

Conclusion. The cycle guarantees 397 points in 37 thousand steps; BFS with the tail test makes about 392 in 23 thousand, with no guarantee and a stuck ending. We did not train the network to imitate BFS: it would need features encoding the route and the simulation result (as the cycle features encode the current teacher), and the stuck ending would have to be solved first. The chosen approach remained cycle imitation.

## Tile scheme: dynamic Hamiltonian cycles

A\* alone is not safe: it breaks the body's cycle order and the ending gets stuck (see above). A scheme of 2x2 tiles (per-cell arrows) solves this. The board becomes a 10x10 grid of tiles. Any spanning tree of that grid defines a Hamiltonian cycle of the 400 cells: along a tree edge the cycle crosses the tile border in both lanes; inside a tile it uses the pair of cells on each side that has no tree edge.

Each step of the body imposes a condition on the tree: passing between tiles forces that edge to be in the tree; moving inside a tile forces the edge of the corresponding side to stay out. The body lies on some cycle exactly when a spanning tree exists that respects all conditions. This is decided with union-find: the mandatory edges cannot close a loop, and the non-forbidden edges must connect the grid (`src/planning/tiles.cpp`). If a move leaves the body on some cycle, the snake can always follow that cycle to its tail, and the modes can be switched at any step.

Validation: 50 random trees give a single cycle with degree 2 at every cell, and every prefix of the cycle is accepted as a body (`experiments/safe_modes.cpp`).

Modes (`src/planning/modes.cpp`), all restricted to moves that keep this property:

- A\*: the safe move with the shortest real path to the food.
- Shortcuts: the safe move closest to the food along the current cycle.
- Cycle: follows the cycle.

12 games, same seeds, 397 = full board:

| Strategy | Points | Steps per game | Deaths |
|---|---|---|---|
| Pure cycle | 397 | 39,574 | 0/12 |
| Shortcuts | 397 | 15,943 | 0/12 |
| A\* | 397 | 15,792 | 0/12 |
| A\* until 100, then shortcuts | 397 | 15,110 | 0/12 |
| Shortcuts until 100, then A\* | 397 | 17,022 | 0/12 |
| A\* until 100, then cycle | 397 | 26,023 | 0/12 |
| A\* until 300, then cycle | 397 | 16,504 | 0/12 |

With the guarantee, A\* fills the board in 60% fewer steps than the cycle and with no deaths, which BFS with the tail test could not do (392 points and a stuck ending). The earlier attempt at "classic" shortcuts (skipping cells of the fixed cycle) breaks the property, because the body stops being a contiguous stretch of the cycle; inside the scheme, shortcuts become a criterion for choosing among safe moves.

### Learned mode selector

`apps/selector.cpp`: 9,748 states sampled over 48 games with random mode switching. At each state, each mode is held for 100 steps and followed by 150 steps of A\*; the label is the mode that ate the most, switching away from A\* only if it wins by at least one food (A\* 6251, shortcuts 2597, cycle 900). An 8→32→32→3 network over 8 scale-free features (size, distances to food and tail, steps without eating, food before tail, safe moves).

The learned selector picks A\* in 98% of steps and finishes in 15,792 steps, the same as pure A\*. The loss stays at 0.83, so the labels are noisy: the difference between modes over 250 steps is a few foods. The best fixed switch (A\* until 100, then shortcuts) gains 4% over A\*, with 12 games and no repetition, so we cannot say it is real.

Conclusion. Choosing among the three modes yields little; the gain came from the safety guarantee, not from the choice. Safe A\* is the best player on speed and always scores 397.

## Imitating the safe A\*

`apps/imitate_astar.cpp` trains a 12→32→32→3 network to imitate the safe A\* (188,270 states from 48 teacher games, one state in four, Adam, 15 epochs). The 12 features are, per action (straight, right, left): blocked, safe (the body stays on some cycle), how many steps longer the free path to the food is than the best safe move's (capped at 4, divided by 4), and whether it is the next step of the cycle. They do not depend on rotation.

12 games, same seeds (100000 to 100011), evaluating the copy reloaded from the checkpoint:

| Player | Points | Steps per game | Deaths |
|---|---|---|---|
| Safe A\* (teacher) | 397 | 15,792 | 0/12 |
| Network, no filter | 397.0 | 15,792 | 0/12 |
| Network, safe moves only | 397.0 | 15,792 | 0/12 |

The network never chose an unsafe move, so the filter changes nothing here. It stays in the code as a guarantee. With 2.5 times fewer steps than the cycle network (39,574), it is the best trained player.

Two problems came up on the way:

1. Distance divided by 400. In the first version the distance feature was the distance over 400, so two moves differed by 0.0025 and the network could not tell them apart. It spun around at the start without eating (4 of 12 games ended in starvation with 0 points; mean 201 to 264, 5 to 6 deaths). Replacing it with "steps more than the best safe move" fixed it, with the loss at 0.014.
2. Loop guard. The teacher follows the cycle after 250 steps without eating; the network gets the same guard in the code that plays (`astarNetAction`). This is not learned by the network.

Limitations: the features already contain the teacher's answer (the rule is "the shortest among the safe ones"), so the network learns to read that information, not to discover it. The tree safety test (`treeFor`) runs at every step and costs more than the network. One training run and 12 games.

## Other board scales, without retraining

`experiments/scale_test.cpp` plays safe A\*, the cycle and the network trained on 20x20 (`models/imitation_astar.ckpt`, not retrained) on 10x10, 20x20 and 40x40 boards. For this the environment, the tiles and the modes accept the board side (even); cycle construction was validated on all three (random trees, single cycle, prefixes accepted). The starvation limit and the steps-without-food limit for returning to the cycle grow with the number of cells (1.5 and 5/8 times), as does the per-game step cap (200 per cell). The network's features do not depend on scale, so the same weights apply.

12 games on 10x10 and 20x20, 4 on 40x40 (the 40x40 games are long and the network plays on a single thread). Maximum score = cells minus 3.

| Board | Player | Points | Steps per game (per cell) | Not full |
|---|---|---|---|---|
| 10x10 | Safe A\* (teacher) | 97 / 97 | 1,330 (13.3) | 0/12 |
| 10x10 | Cycle | 97 / 97 | 2,431 (24.3) | 0/12 |
| 10x10 | Network (20x20 weights) | 97 / 97 | 1,330 (13.3) | 0/12 |
| 20x20 | Safe A\* | 397 / 397 | 15,792 (39.5) | 0/12 |
| 20x20 | Cycle | 397 / 397 | 39,574 (98.9) | 0/12 |
| 20x20 | Network | 397 / 397 | 15,792 (39.5) | 0/12 |
| 40x40 | Safe A\* | 1597 / 1597 | 199,338 (124.6) | 0/4 |
| 40x40 | Cycle | 481.5 / 1597 | 320,000 (200.0) | 4/4 (stopped at the step cap) |
| 40x40 | Network (20x20 weights) | 1597 / 1597 | 199,338 (124.6) | 0/4 |

The network trained only on 20x20 fills the board on 10x10 and 40x40 with exactly the teacher's steps, so it copies the whole trajectory. This confirms what the features already suggested: safe move, "steps more than the best safe move" and "next cycle step" do not change with scale.

Observations:

- Safe A\* spends 13.3, 39.5 and 124.6 steps per cell on 10, 20 and 40: about 3 times more per doubling of the side. The cycle is worse at every scale (24.3 and 98.9 on the first two).
- The cycle on 40x40 did not finish: it stopped at the 320,000-step cap with 481.5 points on average. We cannot say how many steps it would take, only that it would take far more than A\* (199,338).
- The network uses the code's safety filter and loop guard (the same as the teacher's); what is learned is the choice among safe moves.
- The 40x40 comparison has 4 games.

## Saved network and watch mode

`compare imitation` writes the checkpoint (`imitation.ckpt`) and evaluates the reloaded copy, to prove the file holds the trained network. The network in `models/imitation.ckpt` fills the board (397) on seeds 1 and 3, in about 36 thousand steps. To watch:

```bash
./build/bin/watch models/imitation.ckpt [seed=1] [ms per frame=50] [draw every N steps=1]
./build/bin/watch models/imitation_astar.ckpt 1 120 1 astar   # the network imitating safe A*
```

Full games exceed 35 thousand steps; the default (50 ms, every step) is for watching the start; use `120 1` for slower and, to reach the end, something like `20 50` (20 ms, one frame every 50 steps). Ctrl+C stops and restores the cursor.

## Final runs

Logs of the runs made with the reorganized code are in [results/](results/); see [results/README.md](results/README.md) for the summary table.

## Next steps

1. Learn the loop guard and the safety test, which today live in the code (see the section above), or measure how much the network loses without them.
2. Investigate the game that died and, if it is a generalization error, train with more games or by sampling the long stretches.
3. Repeat training with several seeds, since the A\* network's result comes from one training run and 12 games.
4. If invariance matters, try imitating the expert with egocentric features and check how much is lost from the 393.8 to 397.
