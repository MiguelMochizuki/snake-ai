# Final runs

Logs of the main experiments, run with the binaries built from the reorganized code (`build/bin`). Each run used a scratch working directory so the committed checkpoints in `models/` were not overwritten.

| Log | Command | Headline |
|---|---|---|
| [safe_modes.log](safe_modes.log) | `safe_modes 12` | Cycle 39,574 steps, shortcuts 15,943, A\* 15,792; 397 points, 0 deaths in 12 games |
| [imitate_astar.log](imitate_astar.log) | `imitate_astar` | Network imitating safe A\*: 397.0 points, 15,792 steps, 0/12 deaths, raw and filtered |
| [selector.log](selector.log) | `selector` | Learned selector picks A\* in 98% of steps: 15,792 steps, same as pure A\* |
| [scale_test.log](scale_test.log) | `scale_test` | 20x20-trained network equals the teacher on 10x10, 20x20 and 40x40 (full board, same steps) |
| [lookahead_oracle_validate_K1.log](lookahead_oracle_validate_K1.log), K3, K6; [lookahead_oracle_test_K6.log](lookahead_oracle_test_K6.log) | `lookahead_oracle <validate\|test> K` | Lookahead with the true future foods (upper bound only): -27.1%, -32.6%, -34.8% on validation, -35.74% on test (K=6) |
| [lookahead_teacher_validate_K1_M1.log](lookahead_teacher_validate_K1_M1.log), K3_M4, K6_M4, K6_M8; [lookahead_teacher_test_K1_M1.log](lookahead_teacher_test_K1_M1.log) | `lookahead_teacher <validate\|test> K M` | Honest lookahead (sampled futures): K=1 saves 27.1% on validation and 27.40% on test; larger K is worse |
| [timed_astar_validate.log](timed_astar_validate.log), [timed_astar_test.log](timed_astar_test.log) | `timed_astar <validate\|test>` | A\* with a breadth-first search in time: no better than plain A\* (+0.4%, +2.71%) |
| [es_finetune.log](es_finetune.log) | `es_finetune 200 12 --sigma=0.2 --lr=0.2 --final` | Imitation network fine-tuned with evolution strategies: 14,206 steps against 16,016 on 20x20 test seeds (-11.30%, 0 deaths); `models/es_finetuned.ckpt` |
| [bfs_safe.log](bfs_safe.log) | `bfs_safe 12` | BFS with tail test: 392.7 points, 23,327 steps, 12/12 games end in death before filling the board |

Training is not deterministic, so retraining can change losses slightly; the evaluation numbers above match [../REPORT.md](../REPORT.md). The `bfs_safe` log uses 12 games, the report's table 10.
