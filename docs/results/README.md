# Final runs

Logs of the main experiments, run with the binaries built from the reorganized code (`build/bin`). Each run used a scratch working directory so the committed checkpoints in `models/` were not overwritten.

| Log | Command | Headline |
|---|---|---|
| [safe_modes.log](safe_modes.log) | `safe_modes 12` | Cycle 39,574 steps, shortcuts 15,943, A\* 15,792; 397 points, 0 deaths in 12 games |
| [imitate_astar.log](imitate_astar.log) | `imitate_astar` | Network imitating safe A\*: 397.0 points, 15,792 steps, 0/12 deaths, raw and filtered |
| [selector.log](selector.log) | `selector` | Learned selector picks A\* in 98% of steps: 15,792 steps, same as pure A\* |
| [scale_test.log](scale_test.log) | `scale_test` | 20x20-trained network equals the teacher on 10x10, 20x20 and 40x40 (full board, same steps) |
| [bfs_safe.log](bfs_safe.log) | `bfs_safe 12` | BFS with tail test: 392.7 points, 23,327 steps, 12/12 games end in death before filling the board |

Training is not deterministic, so retraining can change losses slightly; the evaluation numbers above match [../REPORT.md](../REPORT.md). The `bfs_safe` log uses 12 games, the report's table 10.
