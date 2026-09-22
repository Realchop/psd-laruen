# Run manifest

Generated 2026-09-21 23:28:56 UTC by `scripts/run_manifest.py`.

Training artefacts are gitignored, so this file is the record that the runs
happened. Each row pairs what training wrote on disk with the job id SLURM
recorded independently; `sacct -j <jobid>` reproduces the scheduler's side
for as long as the cluster keeps its accounting rows.

## Summary

- Runs on disk: **88**
- Runs matched to a SLURM job id: **88**
- Job ids still present in `sacct`: **88**
- Runs whose event-file hostname matches the node SLURM recorded: **88/88**
- Checkpoints: **176** (0.4 GiB), each SHA-256'd below
- First run started: **2026-08-22 19:41:29 UTC**
- Last run started: **2026-09-17 00:57:59 UTC**

## Runs

| run | slurm | matched by | node | started (UTC) | min | ep | lr | best val_esr | ckpts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lightning_logs/cabinet-ampformer_187ms/version_0` | `742728` | checkpoint-path | msp3-3 ✓ | 2026-08-24 20:22:57 | 13.2 | 179 | 0.001 | 1.042e-02 | 2 |
| `lightning_logs/cabinet-ampformer_187ms/version_1` | `915935` | checkpoint-path | msp3-5 ✓ | 2026-09-17 00:24:19 | 1.5 | 20 | 0.001 | 1.061e-01 | 2 |
| `lightning_logs/cabinet-ampformer_48ms/version_0` | `742727` | checkpoint-path | msp3-0 ✓ | 2026-08-24 20:11:54 | 15.4 | 218 | 0.001 | 9.665e-03 | 2 |
| `lightning_logs/cabinet-ampformer_48ms/version_1` | `915934` | checkpoint-path | msp3-4 ✓ | 2026-09-17 00:23:50 | 1.5 | 20 | 0.001 | 8.748e-02 | 2 |
| `lightning_logs/cabinet-wavenet_186ms/version_0` | `742726` | checkpoint-path | msp3-0 ✓ | 2026-08-24 20:11:54 | 162.3 | 245 | 0.001 | 3.807e-05 | 2 |
| `lightning_logs/cabinet-wavenet_186ms/version_1` | `915933` | checkpoint-path | msp3-3 ✓ | 2026-09-17 00:23:46 | 13.4 | 20 | 0.001 | 2.365e-03 | 2 |
| `lightning_logs/cabinet-wavenet_46ms/version_0` | `742725` | checkpoint-path | msp3-0 ✓ | 2026-08-24 20:11:54 | 104 | 186 | 0.001 | 9.904e-05 | 2 |
| `lightning_logs/cabinet-wavenet_46ms/version_1` | `915932` | checkpoint-path | msp3-2 ✓ | 2026-09-17 00:23:10 | 11.2 | 20 | 0.001 | 1.346e-03 | 2 |
| `lightning_logs/clean-ampformer_187ms/version_0` | `742720` | checkpoint-path | msp3-5 ✓ | 2026-08-24 19:46:33 | 5.6 | 76 | 0.001 | 5.991e-08 | 2 |
| `lightning_logs/clean-ampformer_187ms/version_1` | `915880` | checkpoint-path | msp3-4 ✓ | 2026-09-17 00:05:03 | 1.5 | 20 | — | 1.981e-08 | 2 |
| `lightning_logs/clean-ampformer_187ms/version_2` | `915931` | checkpoint-path | msp3-4 ✓ | 2026-09-17 00:22:10 | 1.5 | 20 | 0.001 | 2.123e-08 | 2 |
| `lightning_logs/clean-ampformer_48ms/version_0` | `742719` | checkpoint-path | msp3-1 ✓ | 2026-08-24 19:43:24 | 8.8 | 119 | 0.001 | 2.952e-08 | 2 |
| `lightning_logs/clean-ampformer_48ms/version_1` | `915879` | checkpoint-path | msp3-2 ✓ | 2026-09-17 00:03:47 | 1.5 | 20 | — | 1.988e-08 | 2 |
| `lightning_logs/clean-ampformer_48ms/version_2` | `915930` | checkpoint-path | msp3-2 ✓ | 2026-09-17 00:21:29 | 1.5 | 20 | 0.001 | 2.127e-08 | 2 |
| `lightning_logs/clean-wavenet_186ms/version_0` | `742718` | checkpoint-path | msp3-5 ✓ | 2026-08-24 19:37:56 | 73.9 | 111 | 0.001 | 3.111e-05 | 2 |
| `lightning_logs/clean-wavenet_186ms/version_1` | `915878` | tag+window | msp3-6 ✓ | 2026-09-17 00:03:22 | 8.3 | 13 | — | 1.173e+00 | 2 |
| `lightning_logs/clean-wavenet_186ms/version_2` | `915929` | checkpoint-path | msp3-2 ✓ | 2026-09-17 00:21:29 | 13.3 | 20 | 0.001 | 7.163e-06 | 2 |
| `lightning_logs/clean-wavenet_46ms/version_0` | `742717` | checkpoint-path | msp3-2 ✓ | 2026-08-24 16:46:00 | 75 | 133 | 0.001 | 1.614e-05 | 2 |
| `lightning_logs/clean-wavenet_46ms/version_1` | `915849` | tag+window | msp3-2 ✓ | 2026-09-17 00:02:54 | 8.8 | 16 | — | 8.502e-01 | 2 |
| `lightning_logs/clean-wavenet_46ms/version_2` | `915928` | checkpoint-path | msp3-2 ✓ | 2026-09-17 00:13:40 | 11.3 | 20 | 0.001 | 1.489e-05 | 2 |
| `lightning_logs/metal-ampformer_187ms/version_0` | `742732` | checkpoint-path | msp3-5 ✓ | 2026-08-24 20:52:09 | 29.2 | 400 | 0.001 | 6.530e-03 | 2 |
| `lightning_logs/metal-ampformer_187ms/version_1` | `744422` | checkpoint-path | msp3-0 ✓ | 2026-08-25 11:11:24 | 27.9 | 383 | 0.001 | 5.654e-03 | 2 |
| `lightning_logs/metal-ampformer_187ms/version_2` | `915943` | checkpoint-path | msp3-4 ✓ | 2026-09-17 00:39:22 | 1.5 | 20 | 0.001 | 3.036e-01 | 2 |
| `lightning_logs/metal-ampformer_48ms/version_0` | `742731` | checkpoint-path | msp3-7 ✓ | 2026-08-24 20:45:08 | 29.8 | 400 | 0.001 | 1.241e-02 | 2 |
| `lightning_logs/metal-ampformer_48ms/version_1` | `744421` | checkpoint-path | msp3-0 ✓ | 2026-08-25 10:44:11 | 26.8 | 360 | 0.001 | 1.263e-02 | 2 |
| `lightning_logs/metal-ampformer_48ms/version_2` | `915942` | checkpoint-path | msp3-2 ✓ | 2026-09-17 00:39:03 | 1.5 | 20 | 0.001 | 3.217e-01 | 2 |
| `lightning_logs/metal-ampformer_48ms-lr1e-2/version_0` | `915953` | checkpoint-path | msp3-6 ✓ | 2026-09-17 00:57:59 | 1.5 | 20 | 0.01 | 8.441e-01 | 2 |
| `lightning_logs/metal-ampformer_48ms-lr1e-3/version_0` | `915951` | checkpoint-path | msp3-3 ✓ | 2026-09-17 00:57:12 | 1.6 | 20 | 0.001 | 6.661e-01 | 2 |
| `lightning_logs/metal-ampformer_48ms-lr1e-4/version_0` | `915949` | checkpoint-path | msp3-2 ✓ | 2026-09-17 00:48:50 | 1.5 | 20 | 0.0001 | 8.285e-01 | 2 |
| `lightning_logs/metal-ampformer_48ms-lr3e-3/version_0` | `915952` | checkpoint-path | msp3-2 ✓ | 2026-09-17 00:57:44 | 1.5 | 20 | 0.003 | 8.242e-01 | 2 |
| `lightning_logs/metal-ampformer_48ms-lr3e-4/version_0` | `915950` | checkpoint-path | msp3-6 ✓ | 2026-09-17 00:56:19 | 1.5 | 20 | 0.0003 | 3.727e-01 | 2 |
| `lightning_logs/metal-wavenet_186ms/version_0` | `742730` | tag+window | msp3-0 ✓ | 2026-08-24 20:31:06 | 210.1 | 315 | 0.001 | 8.409e-04 | 2 |
| `lightning_logs/metal-wavenet_186ms/version_1` | `744406` | checkpoint-path | msp3-2 ✓ | 2026-08-25 09:57:43 | 267.5 | 400 | 0.001 | 7.458e-04 | 2 |
| `lightning_logs/metal-wavenet_186ms/version_2` | `915941` | checkpoint-path | msp3-6 ✓ | 2026-09-17 00:28:48 | 13.4 | 20 | 0.001 | 8.213e-03 | 2 |
| `lightning_logs/metal-wavenet_46ms/version_0` | `742729` | checkpoint-path | msp3-3 ✓ | 2026-08-24 20:27:36 | 84.8 | 150 | 0.001 | 6.224e-03 | 2 |
| `lightning_logs/metal-wavenet_46ms/version_1` | `915940` | checkpoint-path | msp3-5 ✓ | 2026-09-17 00:27:40 | 11.3 | 20 | 0.001 | 1.725e-02 | 2 |
| `lightning_logs/metal-wavenet_46ms-lr1e-2/version_0` | `915948` | checkpoint-path | msp3-6 ✓ | 2026-09-17 00:42:25 | 11.3 | 20 | 0.01 | 1.244e-02 | 2 |
| `lightning_logs/metal-wavenet_46ms-lr1e-3/version_0` | `915946` | checkpoint-path | msp3-4 ✓ | 2026-09-17 00:41:19 | 11.3 | 20 | 0.001 | 1.689e-02 | 2 |
| `lightning_logs/metal-wavenet_46ms-lr1e-4/version_0` | `915944` | checkpoint-path | msp3-5 ✓ | 2026-09-17 00:39:19 | 11.3 | 20 | 0.0001 | 7.176e-02 | 2 |
| `lightning_logs/metal-wavenet_46ms-lr3e-3/version_0` | `915947` | checkpoint-path | msp3-2 ✓ | 2026-09-17 00:41:53 | 11.3 | 20 | 0.003 | 1.396e-02 | 2 |
| `lightning_logs/metal-wavenet_46ms-lr3e-4/version_0` | `915945` | checkpoint-path | msp3-2 ✓ | 2026-09-17 00:40:54 | 11.3 | 20 | 0.0003 | 2.482e-02 | 2 |
| `lightning_logs/preamp-ampformer_187ms/version_0` | `742724` | checkpoint-path | msp3-7 ✓ | 2026-08-24 20:00:55 | 29.7 | 400 | 0.001 | 8.732e-03 | 2 |
| `lightning_logs/preamp-ampformer_187ms/version_1` | `744420` | checkpoint-path | msp3-2 ✓ | 2026-08-25 10:20:47 | 42.9 | 570 | 0.001 | 7.699e-03 | 2 |
| `lightning_logs/preamp-ampformer_187ms/version_2` | `915939` | checkpoint-path | msp3-6 ✓ | 2026-09-17 00:27:04 | 1.5 | 20 | 0.001 | 9.966e-02 | 2 |
| `lightning_logs/preamp-ampformer_48ms/version_0` | `742723` | checkpoint-path | msp3-5 ✓ | 2026-08-24 19:52:33 | 29.7 | 400 | 0.001 | 8.906e-03 | 2 |
| `lightning_logs/preamp-ampformer_48ms/version_1` | `744419` | checkpoint-path | msp3-1 ✓ | 2026-08-25 10:08:44 | 47.9 | 641 | 0.001 | 6.946e-03 | 2 |
| `lightning_logs/preamp-ampformer_48ms/version_2` | `915938` | checkpoint-path | msp3-5 ✓ | 2026-09-17 00:26:03 | 1.5 | 20 | 0.001 | 9.386e-02 | 2 |
| `lightning_logs/preamp-wavenet_186ms/version_0` | `742722` | tag+window | msp3-1 ✓ | 2026-08-24 19:52:35 | 119.8 | 179 | 0.001 | 4.720e-04 | 2 |
| `lightning_logs/preamp-wavenet_186ms/version_1` | `742722` | jobname+window | msp3-1 ✓ | 2026-08-24 21:54:44 | 126.4 | 189 | 0.001 | 4.808e-04 | 2 |
| `lightning_logs/preamp-wavenet_186ms/version_2` | `744405` | checkpoint-path | msp3-2 ✓ | 2026-08-25 09:09:56 | 267.6 | 400 | 0.001 | 2.250e-04 | 2 |
| `lightning_logs/preamp-wavenet_186ms/version_3` | `915937` | checkpoint-path | msp3-4 ✓ | 2026-09-17 00:25:31 | 13.4 | 20 | 0.001 | 2.566e-03 | 2 |
| `lightning_logs/preamp-wavenet_46ms/version_0` | `742721` | jobname+window | msp3-1 ✓ | 2026-08-24 19:52:03 | 120 | 212 | 0.001 | 5.010e-04 | 2 |
| `lightning_logs/preamp-wavenet_46ms/version_1` | `742721` | checkpoint-path | msp3-3 ✓ | 2026-08-24 21:54:44 | 226.6 | 400 | 0.001 | 2.867e-04 | 2 |
| `lightning_logs/preamp-wavenet_46ms/version_2` | `744418` | jobname+window | msp3-1 ✓ | 2026-08-25 10:07:05 | 119.6 | 212 | 0.001 | 8.520e-04 | 2 |
| `lightning_logs/preamp-wavenet_46ms/version_3` | `744418` | checkpoint-path | msp3-1 ✓ | 2026-08-25 12:13:25 | 109.7 | 194 | 0.001 | 9.111e-04 | 2 |
| `lightning_logs/preamp-wavenet_46ms/version_4` | `915936` | checkpoint-path | msp3-2 ✓ | 2026-09-17 00:25:07 | 11.3 | 20 | 0.001 | 3.117e-03 | 2 |
| `lightning_logs_40ep_noWarmup/cabinet-ampformer_187ms/version_0` | `739714` | checkpoint-path | msp3-1 ✓ | 2026-08-23 19:02:40 | 2.9 | 40 | 0.001 | 3.173e-02 | 2 |
| `lightning_logs_40ep_noWarmup/cabinet-ampformer_48ms/version_0` | `739713` | checkpoint-path | msp3-1 ✓ | 2026-08-23 19:00:19 | 2.9 | 40 | 0.001 | 2.640e-02 | 2 |
| `lightning_logs_40ep_noWarmup/cabinet-wavenet_186ms/version_0` | `739712` | checkpoint-path | msp3-2 ✓ | 2026-08-23 18:55:09 | 26.5 | 40 | 0.001 | 2.895e-04 | 2 |
| `lightning_logs_40ep_noWarmup/cabinet-wavenet_46ms/version_0` | `739711` | checkpoint-path | msp3-2 ✓ | 2026-08-23 18:48:39 | 22.4 | 40 | 0.001 | 1.235e-03 | 2 |
| `lightning_logs_40ep_noWarmup/clean-ampformer_187ms/version_0` | `739706` | checkpoint-path | msp3-2 ✓ | 2026-08-23 18:28:45 | 3 | 40 | 0.001 | 5.869e-08 | 2 |
| `lightning_logs_40ep_noWarmup/clean-ampformer_48ms/version_0` | `739705` | checkpoint-path | msp3-2 ✓ | 2026-08-23 18:25:27 | 3 | 40 | 0.001 | 1.133e-07 | 2 |
| `lightning_logs_40ep_noWarmup/clean-wavenet_186ms/version_0` | `739704` | checkpoint-path | msp3-1 ✓ | 2026-08-23 18:06:17 | 26.6 | 40 | 0.001 | 5.426e-05 | 2 |
| `lightning_logs_40ep_noWarmup/clean-wavenet_46ms/version_0` | `739703` | checkpoint-path | msp3-1 ✓ | 2026-08-23 17:29:47 | 22.5 | 40 | 0.001 | 5.751e-05 | 2 |
| `lightning_logs_40ep_noWarmup/metal-ampformer_187ms/version_0` | `739718` | checkpoint-path | msp3-0 ✓ | 2026-08-23 19:09:58 | 3 | 40 | 0.001 | 6.972e-02 | 2 |
| `lightning_logs_40ep_noWarmup/metal-ampformer_48ms/version_0` | `739717` | checkpoint-path | msp3-3 ✓ | 2026-08-23 19:06:54 | 2.9 | 40 | 0.001 | 8.165e-02 | 2 |
| `lightning_logs_40ep_noWarmup/metal-wavenet_186ms/version_0` | `739716` | checkpoint-path | msp3-1 ✓ | 2026-08-23 19:06:18 | 26.7 | 40 | 0.001 | 7.117e-03 | 2 |
| `lightning_logs_40ep_noWarmup/metal-wavenet_46ms/version_0` | `739715` | checkpoint-path | msp3-1 ✓ | 2026-08-23 19:03:39 | 22.6 | 40 | 0.001 | 1.528e-02 | 2 |
| `lightning_logs_40ep_noWarmup/preamp-ampformer_187ms/version_0` | `739710` | checkpoint-path | msp3-2 ✓ | 2026-08-23 18:45:09 | 3 | 40 | 0.001 | 3.253e-01 | 2 |
| `lightning_logs_40ep_noWarmup/preamp-ampformer_48ms/version_0` | `739709` | checkpoint-path | msp3-2 ✓ | 2026-08-23 18:41:47 | 2.9 | 40 | 0.001 | 9.776e-02 | 2 |
| `lightning_logs_40ep_noWarmup/preamp-wavenet_186ms/version_0` | `739708` | checkpoint-path | msp3-1 ✓ | 2026-08-23 18:33:16 | 26.7 | 40 | 0.001 | 5.549e-02 | 2 |
| `lightning_logs_40ep_noWarmup/preamp-wavenet_46ms/version_0` | `739707` | checkpoint-path | msp3-2 ✓ | 2026-08-23 18:32:17 | 22.7 | 40 | 0.001 | 1.018e-01 | 2 |
| `lightning_logs_misaligned_grid/cabinet-ampformer_187ms/version_0` | `739683` | checkpoint-path | msp3-3 ✓ | 2026-08-23 17:22:01 | 3 | 40 | 0.001 | 3.653e-02 | 2 |
| `lightning_logs_misaligned_grid/cabinet-ampformer_48ms/version_0` | `739682` | checkpoint-path | msp3-1 ✓ | 2026-08-23 17:21:46 | 3.1 | 40 | 0.001 | 3.662e-02 | 2 |
| `lightning_logs_misaligned_grid/cabinet-wavenet_186ms/version_0` | `739681` | tag+window | msp3-3 ✓ | 2026-08-23 17:21:46 | 7 | 11 | 0.001 | 1.813e-02 | 2 |
| `lightning_logs_misaligned_grid/cabinet-wavenet_46ms/version_0` | `739680` | tag+window | msp3-3 ✓ | 2026-08-23 17:21:46 | 7.2 | 13 | 0.001 | 9.973e-03 | 2 |
| `lightning_logs_misaligned_grid/clean-ampformer_187ms/version_0` | `739620` | checkpoint-path | msp3-1 ✓ | 2026-08-23 16:49:01 | 3 | 40 | 0.001 | 4.243e-08 | 2 |
| `lightning_logs_misaligned_grid/clean-ampformer_48ms/version_0` | `739619` | checkpoint-path | msp3-1 ✓ | 2026-08-23 16:45:29 | 3 | 40 | 0.001 | 7.666e-08 | 2 |
| `lightning_logs_misaligned_grid/clean-wavenet_186ms/version_0` | `739618` | checkpoint-path | msp3-1 ✓ | 2026-08-23 16:45:13 | 26.7 | 40 | 0.001 | 1.227e-04 | 2 |
| `lightning_logs_misaligned_grid/clean-wavenet_46ms/version_0` | `739617` | checkpoint-path | msp3-1 ✓ | 2026-08-23 16:42:53 | 22.7 | 40 | 0.001 | 2.494e-04 | 2 |
| `lightning_logs_misaligned_grid/metal-ampformer_48ms/version_0` | `739686` | checkpoint-path | msp3-3 ✓ | 2026-08-23 17:25:31 | 3.1 | 40 | 0.001 | 6.747e-02 | 2 |
| `lightning_logs_misaligned_grid/metal-wavenet_186ms/version_0` | `739685` | tag+window | msp3-1 ✓ | 2026-08-23 17:25:16 | 3.6 | 6 | 0.001 | 1.132e-01 | 2 |
| `lightning_logs_misaligned_grid/metal-wavenet_46ms/version_0` | `739684` | tag+window | msp3-3 ✓ | 2026-08-23 17:25:15 | 3.6 | 7 | 0.001 | 1.569e-01 | 2 |
| `lightning_logs_misaligned_grid/preamp-ampformer_187ms/version_0` | `739624` | checkpoint-path | msp3-3 ✓ | 2026-08-23 17:02:49 | 3 | 40 | 0.001 | 2.187e-01 | 2 |
| `lightning_logs_misaligned_grid/preamp-ampformer_48ms/version_0` | `737363` | checkpoint-path | msp3-0 ✓ | 2026-08-22 19:41:29 | 0.2 | 1 | 0.001 | 1.308e+00 | 2 |
| `lightning_logs_misaligned_grid/preamp-ampformer_48ms/version_1` | `739623` | checkpoint-path | msp3-3 ✓ | 2026-08-23 17:02:49 | 3.1 | 40 | 0.001 | 3.078e-01 | 2 |
| `lightning_logs_misaligned_grid/preamp-wavenet_186ms/version_0` | `739622` | checkpoint-path | msp3-1 ✓ | 2026-08-23 16:52:31 | 27 | 40 | 0.001 | 1.668e-02 | 2 |
| `lightning_logs_misaligned_grid/preamp-wavenet_46ms/version_0` | `739621` | checkpoint-path | msp3-1 ✓ | 2026-08-23 16:50:32 | 22.8 | 40 | 0.001 | 5.567e-02 | 2 |

† job id parsed from a log file but no longer in the accounting database.

`checkpoint-path` means the job printed the directory it wrote to, so the
pairing is exact. `tag+window` means the scheduler killed the job before it
could print that, and the run is tied to it by grid cell plus the fact that
its logging began inside that job's allocation on that same node.
`jobname+window` is the same match for a run whose log file is gone, using
the name the scheduler itself stored. A ✓ on the node marks a run whose
event filename names the machine SLURM says it allocated.

## SLURM records

Verbatim `sacct` rows for the job ids above, also in `results/slurm_jobs.tsv`. A job that was preempted and requeued kept its id and appears once per allocation.

| jobid | name | state | start | end | elapsed | node |
| --- | --- | --- | --- | --- | --- | --- |
| 742728 | cabinet-ampformer_187ms | COMPLETED | 2026-08-24T20:22:30 | 2026-08-24T20:36:09 | 00:13:39 | msp3-3 |
| 915935 | amp | COMPLETED | 2026-09-17T00:23:53 | 2026-09-17T00:25:53 | 00:02:00 | msp3-5 |
| 742727 | cabinet-ampformer_48ms | COMPLETED | 2026-08-24T20:11:29 | 2026-08-24T20:27:17 | 00:15:48 | msp3-0 |
| 915934 | amp | COMPLETED | 2026-09-17T00:23:41 | 2026-09-17T00:25:21 | 00:01:40 | msp3-4 |
| 742726 | cabinet-wavenet_186ms | COMPLETED | 2026-08-24T20:11:29 | 2026-08-24T22:54:10 | 02:42:41 | msp3-0 |
| 915933 | amp | COMPLETED | 2026-09-17T00:23:18 | 2026-09-17T00:37:09 | 00:13:51 | msp3-3 |
| 742725 | cabinet-wavenet_46ms | COMPLETED | 2026-08-24T20:11:29 | 2026-08-24T21:55:54 | 01:44:25 | msp3-0 |
| 915932 | amp | COMPLETED | 2026-09-17T00:23:02 | 2026-09-17T00:34:23 | 00:11:21 | msp3-2 |
| 742720 | clean-ampformer_187ms | COMPLETED | 2026-08-24T19:46:27 | 2026-08-24T19:52:11 | 00:05:44 | msp3-5 |
| 915880 | amp | COMPLETED | 2026-09-17T00:04:33 | 2026-09-17T00:06:34 | 00:02:01 | msp3-4 |
| 915931 | amp | COMPLETED | 2026-09-17T00:22:00 | 2026-09-17T00:23:40 | 00:01:40 | msp3-4 |
| 742719 | clean-ampformer_48ms | COMPLETED | 2026-08-24T19:42:57 | 2026-08-24T19:52:11 | 00:09:14 | msp3-1 |
| 915879 | amp | COMPLETED | 2026-09-17T00:03:37 | 2026-09-17T00:05:18 | 00:01:41 | msp3-2 |
| 915930 | amp | COMPLETED | 2026-09-17T00:21:20 | 2026-09-17T00:23:01 | 00:01:41 | msp3-2 |
| 742718 | clean-wavenet_186ms | COMPLETED | 2026-08-24T19:37:26 | 2026-08-24T20:51:52 | 01:14:26 | msp3-5 |
| 915878 | amp | CANCELLED by 10206 | 2026-09-17T00:02:39 | 2026-09-17T00:11:43 | 00:09:04 | msp3-6 |
| 915929 | amp | COMPLETED | 2026-09-17T00:21:11 | 2026-09-17T00:34:51 | 00:13:40 | msp3-2 |
| 742717 | clean-wavenet_46ms | COMPLETED | 2026-08-24T16:45:31 | 2026-08-24T18:00:59 | 01:15:28 | msp3-2 |
| 915849 | amp | CANCELLED by 10206 | 2026-09-17T00:02:09 | 2026-09-17T00:11:43 | 00:09:34 | msp3-2 |
| 915928 | amp | COMPLETED | 2026-09-17T00:13:18 | 2026-09-17T00:24:58 | 00:11:40 | msp3-2 |
| 742732 | metal-ampformer_187ms | COMPLETED | 2026-08-24T20:52:03 | 2026-08-24T21:21:23 | 00:29:20 | msp3-5 |
| 744422 | metal-ampformer_187ms | COMPLETED | 2026-08-25T11:11:19 | 2026-08-25T11:39:21 | 00:28:02 | msp3-0 |
| 915943 | amp | COMPLETED | 2026-09-17T00:39:13 | 2026-09-17T00:40:52 | 00:01:39 | msp3-4 |
| 742731 | metal-ampformer_48ms | COMPLETED | 2026-08-24T20:45:02 | 2026-08-24T21:14:58 | 00:29:56 | msp3-7 |
| 744421 | metal-ampformer_48ms | COMPLETED | 2026-08-25T10:43:44 | 2026-08-25T11:10:58 | 00:27:14 | msp3-0 |
| 915942 | amp | COMPLETED | 2026-09-17T00:38:43 | 2026-09-17T00:40:32 | 00:01:49 | msp3-2 |
| 915953 | amp | COMPLETED | 2026-09-17T00:57:50 | 2026-09-17T00:59:28 | 00:01:38 | msp3-6 |
| 915951 | amp | COMPLETED | 2026-09-17T00:56:50 | 2026-09-17T00:58:49 | 00:01:59 | msp3-3 |
| 915949 | amp | COMPLETED | 2026-09-17T00:48:43 | 2026-09-17T00:50:22 | 00:01:39 | msp3-2 |
| 915952 | amp | COMPLETED | 2026-09-17T00:57:33 | 2026-09-17T00:59:13 | 00:01:40 | msp3-2 |
| 915950 | amp | COMPLETED | 2026-09-17T00:56:07 | 2026-09-17T00:57:48 | 00:01:41 | msp3-6 |
| 742730 | metal-wavenet_186ms | PREEMPTED | 2026-08-24T20:31:01 | 2026-08-25T00:01:20 | 03:30:19 | msp3-0 |
| 742730 | metal-wavenet_186ms | FAILED | 2026-08-25T00:03:49 | 2026-08-25T00:03:50 | 00:00:01 | sof1-h200-4 |
| 744406 | metal-wavenet_186ms | COMPLETED | 2026-08-25T09:57:37 | 2026-08-25T14:25:16 | 04:27:39 | msp3-2 |
| 915941 | amp | COMPLETED | 2026-09-17T00:28:37 | 2026-09-17T00:42:13 | 00:13:36 | msp3-6 |
| 742729 | metal-wavenet_46ms | COMPLETED | 2026-08-24T20:27:30 | 2026-08-24T21:52:24 | 01:24:54 | msp3-3 |
| 915940 | amp | COMPLETED | 2026-09-17T00:27:34 | 2026-09-17T00:39:01 | 00:11:27 | msp3-5 |
| 915948 | amp | COMPLETED | 2026-09-17T00:42:14 | 2026-09-17T00:53:44 | 00:11:30 | msp3-6 |
| 915946 | amp | COMPLETED | 2026-09-17T00:41:13 | 2026-09-17T00:52:37 | 00:11:24 | msp3-4 |
| 915944 | amp | COMPLETED | 2026-09-17T00:39:13 | 2026-09-17T00:50:41 | 00:11:28 | msp3-5 |
| 915947 | amp | COMPLETED | 2026-09-17T00:41:43 | 2026-09-17T00:53:13 | 00:11:30 | msp3-2 |
| 915945 | amp | COMPLETED | 2026-09-17T00:40:43 | 2026-09-17T00:52:13 | 00:11:30 | msp3-2 |
| 742724 | preamp-ampformer_187ms | COMPLETED | 2026-08-24T20:00:28 | 2026-08-24T20:30:40 | 00:30:12 | msp3-7 |
| 744420 | preamp-ampformer_187ms | COMPLETED | 2026-08-25T10:20:40 | 2026-08-25T11:03:42 | 00:43:02 | msp3-2 |
| 915939 | amp | COMPLETED | 2026-09-17T00:26:42 | 2026-09-17T00:28:36 | 00:01:54 | msp3-6 |
| 742723 | preamp-ampformer_48ms | COMPLETED | 2026-08-24T19:52:27 | 2026-08-24T20:22:13 | 00:29:46 | msp3-5 |
| 744419 | preamp-ampformer_48ms | COMPLETED | 2026-08-25T10:08:39 | 2026-08-25T10:56:37 | 00:47:58 | msp3-1 |
| 915938 | amp | COMPLETED | 2026-09-17T00:25:54 | 2026-09-17T00:27:33 | 00:01:39 | msp3-5 |
| 742722 | preamp-wavenet_186ms | PREEMPTED | 2026-08-24T19:52:27 | 2026-08-24T21:52:28 | 02:00:01 | msp3-1 |
| 742722 | preamp-wavenet_186ms | PREEMPTED | 2026-08-24T21:54:38 | 2026-08-25T00:01:14 | 02:06:36 | msp3-1 |
| 742722 | preamp-wavenet_186ms | FAILED | 2026-08-25T00:03:19 | 2026-08-25T00:03:20 | 00:00:01 | sof1-h200-4 |
| 744405 | preamp-wavenet_186ms | COMPLETED | 2026-08-25T09:09:32 | 2026-08-25T13:37:31 | 04:27:59 | msp3-2 |
| 915937 | amp | COMPLETED | 2026-09-17T00:25:22 | 2026-09-17T00:38:54 | 00:13:32 | msp3-4 |
| 742721 | preamp-wavenet_46ms | PREEMPTED | 2026-08-24T19:51:57 | 2026-08-24T21:52:08 | 02:00:11 | msp3-1 |
| 742721 | preamp-wavenet_46ms | COMPLETED | 2026-08-24T21:54:38 | 2026-08-25T01:41:21 | 03:46:43 | msp3-3 |
| 744418 | preamp-wavenet_46ms | PREEMPTED | 2026-08-25T10:06:38 | 2026-08-25T12:06:39 | 02:00:01 | msp3-1 |
| 744418 | preamp-wavenet_46ms | COMPLETED | 2026-08-25T12:12:59 | 2026-08-25T14:03:09 | 01:50:10 | msp3-1 |
| 915936 | amp | COMPLETED | 2026-09-17T00:24:59 | 2026-09-17T00:36:28 | 00:11:29 | msp3-2 |
| 739714 | cabinet-ampformer_187ms | COMPLETED | 2026-08-23T19:02:33 | 2026-08-23T19:05:33 | 00:03:00 | msp3-1 |
| 739713 | cabinet-ampformer_48ms | COMPLETED | 2026-08-23T19:00:03 | 2026-08-23T19:03:14 | 00:03:11 | msp3-1 |
| 739712 | cabinet-wavenet_186ms | COMPLETED | 2026-08-23T18:55:03 | 2026-08-23T19:21:40 | 00:26:37 | msp3-2 |
| 739711 | cabinet-wavenet_46ms | COMPLETED | 2026-08-23T18:48:32 | 2026-08-23T19:11:04 | 00:22:32 | msp3-2 |
| 739706 | clean-ampformer_187ms | COMPLETED | 2026-08-23T18:28:31 | 2026-08-23T18:31:45 | 00:03:14 | msp3-2 |
| 739705 | clean-ampformer_48ms | COMPLETED | 2026-08-23T18:25:00 | 2026-08-23T18:28:27 | 00:03:27 | msp3-2 |
| 739704 | clean-wavenet_186ms | COMPLETED | 2026-08-23T18:05:59 | 2026-08-23T18:32:56 | 00:26:57 | msp3-1 |
| 739703 | clean-wavenet_46ms | COMPLETED | 2026-08-23T17:29:26 | 2026-08-23T17:52:20 | 00:22:54 | msp3-1 |
| 739718 | metal-ampformer_187ms | COMPLETED | 2026-08-23T19:09:34 | 2026-08-23T19:12:58 | 00:03:24 | msp3-0 |
| 739717 | metal-ampformer_48ms | COMPLETED | 2026-08-23T19:06:33 | 2026-08-23T19:09:48 | 00:03:15 | msp3-3 |
| 739716 | metal-wavenet_186ms | COMPLETED | 2026-08-23T19:06:03 | 2026-08-23T19:33:01 | 00:26:58 | msp3-1 |
| 739715 | metal-wavenet_46ms | COMPLETED | 2026-08-23T19:03:33 | 2026-08-23T19:26:17 | 00:22:44 | msp3-1 |
| 739710 | preamp-ampformer_187ms | COMPLETED | 2026-08-23T18:45:02 | 2026-08-23T18:48:08 | 00:03:06 | msp3-2 |
| 739709 | preamp-ampformer_48ms | COMPLETED | 2026-08-23T18:41:32 | 2026-08-23T18:44:44 | 00:03:12 | msp3-2 |
| 739708 | preamp-wavenet_186ms | COMPLETED | 2026-08-23T18:33:01 | 2026-08-23T19:00:01 | 00:27:00 | msp3-1 |
| 739707 | preamp-wavenet_46ms | COMPLETED | 2026-08-23T18:32:01 | 2026-08-23T18:54:57 | 00:22:56 | msp3-2 |
| 739683 | cabinet-ampformer_187ms | COMPLETED | 2026-08-23T17:21:56 | 2026-08-23T17:24:59 | 00:03:03 | msp3-3 |
| 739682 | cabinet-ampformer_48ms | COMPLETED | 2026-08-23T17:21:26 | 2026-08-23T17:24:51 | 00:03:25 | msp3-1 |
| 739681 | cabinet-wavenet_186ms | CANCELLED by 10206 | 2026-08-23T17:21:26 | 2026-08-23T17:28:59 | 00:07:33 | msp3-3 |
| 739680 | cabinet-wavenet_46ms | CANCELLED by 10206 | 2026-08-23T17:21:26 | 2026-08-23T17:28:59 | 00:07:33 | msp3-3 |
| 739620 | clean-ampformer_187ms | COMPLETED | 2026-08-23T16:48:54 | 2026-08-23T16:52:05 | 00:03:11 | msp3-1 |
| 739619 | clean-ampformer_48ms | COMPLETED | 2026-08-23T16:45:23 | 2026-08-23T16:48:32 | 00:03:09 | msp3-1 |
| 739618 | clean-wavenet_186ms | COMPLETED | 2026-08-23T16:44:53 | 2026-08-23T17:11:58 | 00:27:05 | msp3-1 |
| 739617 | clean-wavenet_46ms | COMPLETED | 2026-08-23T16:42:23 | 2026-08-23T17:05:33 | 00:23:10 | msp3-1 |
| 739686 | metal-ampformer_48ms | COMPLETED | 2026-08-23T17:25:26 | 2026-08-23T17:28:41 | 00:03:15 | msp3-3 |
| 739685 | metal-wavenet_186ms | CANCELLED by 10206 | 2026-08-23T17:24:56 | 2026-08-23T17:28:59 | 00:04:03 | msp3-1 |
| 739684 | metal-wavenet_46ms | CANCELLED by 10206 | 2026-08-23T17:24:56 | 2026-08-23T17:28:59 | 00:04:03 | msp3-3 |
| 739624 | preamp-ampformer_187ms | COMPLETED | 2026-08-23T17:02:25 | 2026-08-23T17:05:52 | 00:03:27 | msp3-3 |
| 737363 | smoke | COMPLETED | 2026-08-22T19:41:03 | 2026-08-22T19:41:43 | 00:00:40 | msp3-0 |
| 739623 | preamp-ampformer_48ms | COMPLETED | 2026-08-23T17:02:25 | 2026-08-23T17:05:56 | 00:03:31 | msp3-3 |
| 739622 | preamp-wavenet_186ms | COMPLETED | 2026-08-23T16:52:24 | 2026-08-23T17:19:30 | 00:27:06 | msp3-1 |
| 739621 | preamp-wavenet_46ms | COMPLETED | 2026-08-23T16:50:24 | 2026-08-23T17:13:22 | 00:22:58 | msp3-1 |

## Checkpoint digests

```
57ec99917174a086a72b07836c07dc0d7808a10658a629c80cdddd3b678206d8       2546763  lightning_logs/cabinet-ampformer_187ms/version_0/checkpoints/epoch=138-val_esr=0.01042.ckpt
57ec99917174a086a72b07836c07dc0d7808a10658a629c80cdddd3b678206d8       2546763  lightning_logs/cabinet-ampformer_187ms/version_0/checkpoints/last.ckpt
9b7bb358dcd2b2e411b0740fcdaff9f2ce8bc9d0332a7308f133ea99c98d6892       2546763  lightning_logs/cabinet-ampformer_187ms/version_1/checkpoints/epoch=19-val_esr=0.10613.ckpt
9b7bb358dcd2b2e411b0740fcdaff9f2ce8bc9d0332a7308f133ea99c98d6892       2546763  lightning_logs/cabinet-ampformer_187ms/version_1/checkpoints/last.ckpt
3544812ab928ca7c9b016bb1e7e8fa8da0b8a64a669276bcaca41ba6b4ce9406       2546763  lightning_logs/cabinet-ampformer_48ms/version_0/checkpoints/epoch=177-val_esr=0.00967.ckpt
3544812ab928ca7c9b016bb1e7e8fa8da0b8a64a669276bcaca41ba6b4ce9406       2546763  lightning_logs/cabinet-ampformer_48ms/version_0/checkpoints/last.ckpt
5ee4f27e72f25ccfe76a0161035eb47589c9b38e193f1030135be6a6b6c6d957       2546763  lightning_logs/cabinet-ampformer_48ms/version_1/checkpoints/epoch=19-val_esr=0.08748.ckpt
5ee4f27e72f25ccfe76a0161035eb47589c9b38e193f1030135be6a6b6c6d957       2546763  lightning_logs/cabinet-ampformer_48ms/version_1/checkpoints/last.ckpt
988fa64594e2354a6989c0b06d9301afd9e9a54031843d64b3eb04fef019d890       3093335  lightning_logs/cabinet-wavenet_186ms/version_0/checkpoints/epoch=204-val_esr=0.00004.ckpt
988fa64594e2354a6989c0b06d9301afd9e9a54031843d64b3eb04fef019d890       3093335  lightning_logs/cabinet-wavenet_186ms/version_0/checkpoints/last.ckpt
149c2f6e083997445099d0105b2c5a187f1dc4842b148eb8641f57d2c9e40d6f       3093335  lightning_logs/cabinet-wavenet_186ms/version_1/checkpoints/epoch=19-val_esr=0.00237.ckpt
149c2f6e083997445099d0105b2c5a187f1dc4842b148eb8641f57d2c9e40d6f       3093335  lightning_logs/cabinet-wavenet_186ms/version_1/checkpoints/last.ckpt
ee03c7f2e9d5249d5e3dab26acbb9f5cc123eb0993acf35b92298cc53daaa18e       2611415  lightning_logs/cabinet-wavenet_46ms/version_0/checkpoints/epoch=145-val_esr=0.00010.ckpt
ee03c7f2e9d5249d5e3dab26acbb9f5cc123eb0993acf35b92298cc53daaa18e       2611415  lightning_logs/cabinet-wavenet_46ms/version_0/checkpoints/last.ckpt
7e23842c6a448fcb5f9f91415849c0b5702bcfd7d8ea70708c84a3f11dcb8617       2611415  lightning_logs/cabinet-wavenet_46ms/version_1/checkpoints/epoch=19-val_esr=0.00135.ckpt
7e23842c6a448fcb5f9f91415849c0b5702bcfd7d8ea70708c84a3f11dcb8617       2611415  lightning_logs/cabinet-wavenet_46ms/version_1/checkpoints/last.ckpt
6e08f981271dca72e77dd8a3ea4ad985fc699200bc273c0d212d3e67c1f07d37       2546763  lightning_logs/clean-ampformer_187ms/version_0/checkpoints/epoch=35-val_esr=0.00000.ckpt
6e08f981271dca72e77dd8a3ea4ad985fc699200bc273c0d212d3e67c1f07d37       2546763  lightning_logs/clean-ampformer_187ms/version_0/checkpoints/last.ckpt
1f0356d2db7a378ffc9589c542b7850f20d60a58830d227c7bd7d7b583900b26       2546763  lightning_logs/clean-ampformer_187ms/version_1/checkpoints/epoch=19-val_esr=0.00000.ckpt
1f0356d2db7a378ffc9589c542b7850f20d60a58830d227c7bd7d7b583900b26       2546763  lightning_logs/clean-ampformer_187ms/version_1/checkpoints/last.ckpt
29c1a3e62d8560f08c30de9c173a5e8b76e176dd35f6f1173e7a412b11181a9b       2546763  lightning_logs/clean-ampformer_187ms/version_2/checkpoints/epoch=19-val_esr=0.00000.ckpt
29c1a3e62d8560f08c30de9c173a5e8b76e176dd35f6f1173e7a412b11181a9b       2546763  lightning_logs/clean-ampformer_187ms/version_2/checkpoints/last.ckpt
02c9c54aa693243d1f9ef9891cc35b2e4db28ea289017f2fcc8993a64d0d77a1       2546763  lightning_logs/clean-ampformer_48ms/version_0/checkpoints/epoch=78-val_esr=0.00000.ckpt
02c9c54aa693243d1f9ef9891cc35b2e4db28ea289017f2fcc8993a64d0d77a1       2546763  lightning_logs/clean-ampformer_48ms/version_0/checkpoints/last.ckpt
afc0b9a3da2e1160215f96e743c55e49fa38b2b1cdd68a2e5c7cb285ffa5602c       2546763  lightning_logs/clean-ampformer_48ms/version_1/checkpoints/epoch=19-val_esr=0.00000.ckpt
afc0b9a3da2e1160215f96e743c55e49fa38b2b1cdd68a2e5c7cb285ffa5602c       2546763  lightning_logs/clean-ampformer_48ms/version_1/checkpoints/last.ckpt
bf3deb8c7c0e2f806f3409d993b25dc5bf47454c4b7b5609b0acb6f0240fe228       2546763  lightning_logs/clean-ampformer_48ms/version_2/checkpoints/epoch=19-val_esr=0.00000.ckpt
bf3deb8c7c0e2f806f3409d993b25dc5bf47454c4b7b5609b0acb6f0240fe228       2546763  lightning_logs/clean-ampformer_48ms/version_2/checkpoints/last.ckpt
039906cf97a3b8e0c6ad731ba801ff49fc9aa85020c6dc90cc26473d8b0ba441       3093335  lightning_logs/clean-wavenet_186ms/version_0/checkpoints/epoch=70-val_esr=0.00003.ckpt
039906cf97a3b8e0c6ad731ba801ff49fc9aa85020c6dc90cc26473d8b0ba441       3093335  lightning_logs/clean-wavenet_186ms/version_0/checkpoints/last.ckpt
ed350eb238d5db646107b7a63153a5c26a773724b4555560ab09030f0a5a1d36       3093335  lightning_logs/clean-wavenet_186ms/version_1/checkpoints/epoch=11-val_esr=1.17347.ckpt
ed350eb238d5db646107b7a63153a5c26a773724b4555560ab09030f0a5a1d36       3093335  lightning_logs/clean-wavenet_186ms/version_1/checkpoints/last.ckpt
20b7c1df3f028e59907efeed930b0a2de1944c1b4854297793f95025b47adf65       3093335  lightning_logs/clean-wavenet_186ms/version_2/checkpoints/epoch=19-val_esr=0.00001.ckpt
20b7c1df3f028e59907efeed930b0a2de1944c1b4854297793f95025b47adf65       3093335  lightning_logs/clean-wavenet_186ms/version_2/checkpoints/last.ckpt
182fb58fe7cfa3f615560a1a524d385572a034559128e3b276378d5bafbb6cee       2611415  lightning_logs/clean-wavenet_46ms/version_0/checkpoints/epoch=92-val_esr=0.00002.ckpt
182fb58fe7cfa3f615560a1a524d385572a034559128e3b276378d5bafbb6cee       2611415  lightning_logs/clean-wavenet_46ms/version_0/checkpoints/last.ckpt
5f15120e5dc9bd118c94e17b1481017793c195567334a43a623fba6e381d508d       2611415  lightning_logs/clean-wavenet_46ms/version_1/checkpoints/epoch=14-val_esr=0.85017.ckpt
5f15120e5dc9bd118c94e17b1481017793c195567334a43a623fba6e381d508d       2611415  lightning_logs/clean-wavenet_46ms/version_1/checkpoints/last.ckpt
df9fdd2a4eca4c334b348b016ffc6da00b7aca648ff302b663fdf0eef7a6f7ee       2611415  lightning_logs/clean-wavenet_46ms/version_2/checkpoints/epoch=19-val_esr=0.00001.ckpt
df9fdd2a4eca4c334b348b016ffc6da00b7aca648ff302b663fdf0eef7a6f7ee       2611415  lightning_logs/clean-wavenet_46ms/version_2/checkpoints/last.ckpt
5636eb77a02393fdfa8613edc412099ae951e2205ceef672b36022d344409b9c       2546763  lightning_logs/metal-ampformer_187ms/version_0/checkpoints/epoch=382-val_esr=0.00653.ckpt
5636eb77a02393fdfa8613edc412099ae951e2205ceef672b36022d344409b9c       2546763  lightning_logs/metal-ampformer_187ms/version_0/checkpoints/last.ckpt
337410ae1e277c4b4dc40de52cea2e9c1f87854aa18fb2a97c83d5f0cd57934b       2546763  lightning_logs/metal-ampformer_187ms/version_1/checkpoints/epoch=342-val_esr=0.00565.ckpt
337410ae1e277c4b4dc40de52cea2e9c1f87854aa18fb2a97c83d5f0cd57934b       2546763  lightning_logs/metal-ampformer_187ms/version_1/checkpoints/last.ckpt
5f8b43ce9228221f8769847976b8029ae5bd849a4cc542d07a49af8a8c4d49ac       2546763  lightning_logs/metal-ampformer_187ms/version_2/checkpoints/epoch=19-val_esr=0.30359.ckpt
5f8b43ce9228221f8769847976b8029ae5bd849a4cc542d07a49af8a8c4d49ac       2546763  lightning_logs/metal-ampformer_187ms/version_2/checkpoints/last.ckpt
d7ca4793abcabf0f26935e6a6c52664a51d125ff2aa28b2b82a1e12d76d252c9       2546763  lightning_logs/metal-ampformer_48ms/version_0/checkpoints/epoch=386-val_esr=0.01241.ckpt
d7ca4793abcabf0f26935e6a6c52664a51d125ff2aa28b2b82a1e12d76d252c9       2546763  lightning_logs/metal-ampformer_48ms/version_0/checkpoints/last.ckpt
7d37ead5ff895e30397089916fa7b8224bbae5184d2c4a109cbaab4cdd70097c       2546763  lightning_logs/metal-ampformer_48ms/version_1/checkpoints/epoch=319-val_esr=0.01263.ckpt
7d37ead5ff895e30397089916fa7b8224bbae5184d2c4a109cbaab4cdd70097c       2546763  lightning_logs/metal-ampformer_48ms/version_1/checkpoints/last.ckpt
9fd083b723aee7e58832487fa95c33e35749e8283001e3b8ea52d8aea4936d2c       2546763  lightning_logs/metal-ampformer_48ms/version_2/checkpoints/epoch=19-val_esr=0.32166.ckpt
9fd083b723aee7e58832487fa95c33e35749e8283001e3b8ea52d8aea4936d2c       2546763  lightning_logs/metal-ampformer_48ms/version_2/checkpoints/last.ckpt
93561115f65cb29985212b045b4f28ad23de7d7b520e3f065098e2b2c13d8776       2546763  lightning_logs/metal-ampformer_48ms-lr1e-2/version_0/checkpoints/epoch=18-val_esr=0.84412.ckpt
93561115f65cb29985212b045b4f28ad23de7d7b520e3f065098e2b2c13d8776       2546763  lightning_logs/metal-ampformer_48ms-lr1e-2/version_0/checkpoints/last.ckpt
cf51b269a21d229d5ffdb3cca245c251b587b73c835cd7abac91af73a3d5ac3b       2546763  lightning_logs/metal-ampformer_48ms-lr1e-3/version_0/checkpoints/epoch=19-val_esr=0.66610.ckpt
cf51b269a21d229d5ffdb3cca245c251b587b73c835cd7abac91af73a3d5ac3b       2546763  lightning_logs/metal-ampformer_48ms-lr1e-3/version_0/checkpoints/last.ckpt
ffc27c977528e2c9a19b17a5d6841ccb4951c8e933edb6e7963ea9776d3642e7       2546763  lightning_logs/metal-ampformer_48ms-lr1e-4/version_0/checkpoints/epoch=19-val_esr=0.82851.ckpt
ffc27c977528e2c9a19b17a5d6841ccb4951c8e933edb6e7963ea9776d3642e7       2546763  lightning_logs/metal-ampformer_48ms-lr1e-4/version_0/checkpoints/last.ckpt
ff4834a1c9b64c6215cbb07dbbfc996d3066831fa640eff48d18ef98c46e0fea       2546763  lightning_logs/metal-ampformer_48ms-lr3e-3/version_0/checkpoints/epoch=19-val_esr=0.82422.ckpt
ff4834a1c9b64c6215cbb07dbbfc996d3066831fa640eff48d18ef98c46e0fea       2546763  lightning_logs/metal-ampformer_48ms-lr3e-3/version_0/checkpoints/last.ckpt
2e4d955e509415517f5b88e58796cbb162b25abcb420325a912b11e6f3eabe4d       2546763  lightning_logs/metal-ampformer_48ms-lr3e-4/version_0/checkpoints/epoch=19-val_esr=0.37271.ckpt
2e4d955e509415517f5b88e58796cbb162b25abcb420325a912b11e6f3eabe4d       2546763  lightning_logs/metal-ampformer_48ms-lr3e-4/version_0/checkpoints/last.ckpt
2914d07209b88d3907052e3b1bf6e4c3a23b14eabca7c4e291ba8742d0557213       3093335  lightning_logs/metal-wavenet_186ms/version_0/checkpoints/epoch=313-val_esr=0.00084.ckpt
2914d07209b88d3907052e3b1bf6e4c3a23b14eabca7c4e291ba8742d0557213       3093335  lightning_logs/metal-wavenet_186ms/version_0/checkpoints/last.ckpt
0c4b07dab4655dee41badb8735fa6ca837a9e4c5dd1bda1c00934204cfef170f       3093335  lightning_logs/metal-wavenet_186ms/version_1/checkpoints/epoch=388-val_esr=0.00075.ckpt
0c4b07dab4655dee41badb8735fa6ca837a9e4c5dd1bda1c00934204cfef170f       3093335  lightning_logs/metal-wavenet_186ms/version_1/checkpoints/last.ckpt
31e550e174fed62b6f6a455105ce7ec1133804cdab20068682e9da5f54fd3bc1       3093335  lightning_logs/metal-wavenet_186ms/version_2/checkpoints/epoch=19-val_esr=0.00821.ckpt
31e550e174fed62b6f6a455105ce7ec1133804cdab20068682e9da5f54fd3bc1       3093335  lightning_logs/metal-wavenet_186ms/version_2/checkpoints/last.ckpt
4b05bde358914e7133334f16820fd0ced3dd6d2b38d18341d4132d5a06f57d54       2611415  lightning_logs/metal-wavenet_46ms/version_0/checkpoints/epoch=109-val_esr=0.00622.ckpt
4b05bde358914e7133334f16820fd0ced3dd6d2b38d18341d4132d5a06f57d54       2611415  lightning_logs/metal-wavenet_46ms/version_0/checkpoints/last.ckpt
d9304162f3564582ff3ca61486be0ee4f2f08d3df78360a6af40a921171c86a9       2611415  lightning_logs/metal-wavenet_46ms/version_1/checkpoints/epoch=19-val_esr=0.01725.ckpt
d9304162f3564582ff3ca61486be0ee4f2f08d3df78360a6af40a921171c86a9       2611415  lightning_logs/metal-wavenet_46ms/version_1/checkpoints/last.ckpt
c376f05dd429a4f364562f86f3dc778de9c2a03aefeb800232926c77e1a040d7       2611415  lightning_logs/metal-wavenet_46ms-lr1e-2/version_0/checkpoints/epoch=19-val_esr=0.01244.ckpt
c376f05dd429a4f364562f86f3dc778de9c2a03aefeb800232926c77e1a040d7       2611415  lightning_logs/metal-wavenet_46ms-lr1e-2/version_0/checkpoints/last.ckpt
15763314232822c20689b9f4e663622de9e05c6d4176609d9bdcc4fa95ae7bfc       2611415  lightning_logs/metal-wavenet_46ms-lr1e-3/version_0/checkpoints/epoch=19-val_esr=0.01689.ckpt
15763314232822c20689b9f4e663622de9e05c6d4176609d9bdcc4fa95ae7bfc       2611415  lightning_logs/metal-wavenet_46ms-lr1e-3/version_0/checkpoints/last.ckpt
0b5c9b1b9a024a9b2d7e7973c3e3fa84149bf676102b22d3b26a97692abbd761       2611415  lightning_logs/metal-wavenet_46ms-lr1e-4/version_0/checkpoints/epoch=18-val_esr=0.07176.ckpt
0b5c9b1b9a024a9b2d7e7973c3e3fa84149bf676102b22d3b26a97692abbd761       2611415  lightning_logs/metal-wavenet_46ms-lr1e-4/version_0/checkpoints/last.ckpt
698d0b14ee8cc1a599da8bc7fca4b5a07f196696fb1ddbe6f31daa530a6d806f       2611415  lightning_logs/metal-wavenet_46ms-lr3e-3/version_0/checkpoints/epoch=19-val_esr=0.01396.ckpt
698d0b14ee8cc1a599da8bc7fca4b5a07f196696fb1ddbe6f31daa530a6d806f       2611415  lightning_logs/metal-wavenet_46ms-lr3e-3/version_0/checkpoints/last.ckpt
f36fa8cedded168c97ca89b989ebae77fe1d211c61d63571931608858c596952       2611415  lightning_logs/metal-wavenet_46ms-lr3e-4/version_0/checkpoints/epoch=18-val_esr=0.02482.ckpt
f36fa8cedded168c97ca89b989ebae77fe1d211c61d63571931608858c596952       2611415  lightning_logs/metal-wavenet_46ms-lr3e-4/version_0/checkpoints/last.ckpt
b79a95e45fa5fd1fc1d6242d1b10fb89097e947417db9f244d4221892f4af2e8       2546763  lightning_logs/preamp-ampformer_187ms/version_0/checkpoints/epoch=390-val_esr=0.00873.ckpt
b79a95e45fa5fd1fc1d6242d1b10fb89097e947417db9f244d4221892f4af2e8       2546763  lightning_logs/preamp-ampformer_187ms/version_0/checkpoints/last.ckpt
f2c69831e6de8be74e55ff3827ae257eb293f1a236c402099d9c947d481d5fb3       2546827  lightning_logs/preamp-ampformer_187ms/version_1/checkpoints/epoch=529-val_esr=0.00770.ckpt
f2c69831e6de8be74e55ff3827ae257eb293f1a236c402099d9c947d481d5fb3       2546827  lightning_logs/preamp-ampformer_187ms/version_1/checkpoints/last.ckpt
ffd9aaecbd57cf2de917bdcf7502f64ef155f1589e838a4ca6eb4263a02e3f4f       2546763  lightning_logs/preamp-ampformer_187ms/version_2/checkpoints/epoch=19-val_esr=0.09966.ckpt
ffd9aaecbd57cf2de917bdcf7502f64ef155f1589e838a4ca6eb4263a02e3f4f       2546763  lightning_logs/preamp-ampformer_187ms/version_2/checkpoints/last.ckpt
73169b8a3be9745a8ec7669f743e6e3f51ba4b0eb4dd9a38df90ec66a334f30f       2546763  lightning_logs/preamp-ampformer_48ms/version_0/checkpoints/epoch=395-val_esr=0.00891.ckpt
73169b8a3be9745a8ec7669f743e6e3f51ba4b0eb4dd9a38df90ec66a334f30f       2546763  lightning_logs/preamp-ampformer_48ms/version_0/checkpoints/last.ckpt
bfe197074754454fb16df1132ad54d200f6a1814a9620b21ac25b89626869f3d       2546827  lightning_logs/preamp-ampformer_48ms/version_1/checkpoints/epoch=600-val_esr=0.00695.ckpt
bfe197074754454fb16df1132ad54d200f6a1814a9620b21ac25b89626869f3d       2546827  lightning_logs/preamp-ampformer_48ms/version_1/checkpoints/last.ckpt
8fb64c5e8ecba615ca8221a33fe80f3dd449ed9760022be2b856c792785cd1f0       2546763  lightning_logs/preamp-ampformer_48ms/version_2/checkpoints/epoch=19-val_esr=0.09386.ckpt
8fb64c5e8ecba615ca8221a33fe80f3dd449ed9760022be2b856c792785cd1f0       2546763  lightning_logs/preamp-ampformer_48ms/version_2/checkpoints/last.ckpt
b0e6a4f315a7444c12b31a7944f5cc6eb49739718435ab33dda68d6bc0446481       3093335  lightning_logs/preamp-wavenet_186ms/version_0/checkpoints/epoch=175-val_esr=0.00047.ckpt
b0e6a4f315a7444c12b31a7944f5cc6eb49739718435ab33dda68d6bc0446481       3093335  lightning_logs/preamp-wavenet_186ms/version_0/checkpoints/last.ckpt
f18a83e9bf44e1cc1d6af94f69f8d85bdae6eeeab87038fe0cf125eb56ac73f0       3093335  lightning_logs/preamp-wavenet_186ms/version_1/checkpoints/epoch=184-val_esr=0.00048.ckpt
f18a83e9bf44e1cc1d6af94f69f8d85bdae6eeeab87038fe0cf125eb56ac73f0       3093335  lightning_logs/preamp-wavenet_186ms/version_1/checkpoints/last.ckpt
42d47006f877453d655f77b001837b03a981f5d71951a446e947efdf2c61629a       3093335  lightning_logs/preamp-wavenet_186ms/version_2/checkpoints/epoch=386-val_esr=0.00022.ckpt
42d47006f877453d655f77b001837b03a981f5d71951a446e947efdf2c61629a       3093335  lightning_logs/preamp-wavenet_186ms/version_2/checkpoints/last.ckpt
362fc56b772d051678503746b3b17e0700d1dc1678d28b63343d3645db16df25       3093335  lightning_logs/preamp-wavenet_186ms/version_3/checkpoints/epoch=19-val_esr=0.00257.ckpt
362fc56b772d051678503746b3b17e0700d1dc1678d28b63343d3645db16df25       3093335  lightning_logs/preamp-wavenet_186ms/version_3/checkpoints/last.ckpt
d63a875ee66e626d4a101340469f9aa43dda8e9c57503f508035a837ea6446f5       2611415  lightning_logs/preamp-wavenet_46ms/version_0/checkpoints/epoch=190-val_esr=0.00050.ckpt
d63a875ee66e626d4a101340469f9aa43dda8e9c57503f508035a837ea6446f5       2611415  lightning_logs/preamp-wavenet_46ms/version_0/checkpoints/last.ckpt
1bf3e20a3386b5016cb5337fbd84a0ea8e4732d8b4c10c105906c6b37efc2520       2611415  lightning_logs/preamp-wavenet_46ms/version_1/checkpoints/epoch=377-val_esr=0.00029.ckpt
1bf3e20a3386b5016cb5337fbd84a0ea8e4732d8b4c10c105906c6b37efc2520       2611415  lightning_logs/preamp-wavenet_46ms/version_1/checkpoints/last.ckpt
5090fc6f3e986d71248a12213dc0d910cc51f40e83489d43280bed79f53fabfa       2611415  lightning_logs/preamp-wavenet_46ms/version_2/checkpoints/epoch=210-val_esr=0.00085.ckpt
5090fc6f3e986d71248a12213dc0d910cc51f40e83489d43280bed79f53fabfa       2611415  lightning_logs/preamp-wavenet_46ms/version_2/checkpoints/last.ckpt
c5a29ab47015c4bb08e8073f7c65a61951b445968496f447747a894a4697b4eb       2611415  lightning_logs/preamp-wavenet_46ms/version_3/checkpoints/epoch=153-val_esr=0.00091.ckpt
c5a29ab47015c4bb08e8073f7c65a61951b445968496f447747a894a4697b4eb       2611415  lightning_logs/preamp-wavenet_46ms/version_3/checkpoints/last.ckpt
01e54bbec21c8d8579ebd48fd4792e170c30f9034f2b026bd2b814ae953901d9       2611415  lightning_logs/preamp-wavenet_46ms/version_4/checkpoints/epoch=18-val_esr=0.00312.ckpt
01e54bbec21c8d8579ebd48fd4792e170c30f9034f2b026bd2b814ae953901d9       2611415  lightning_logs/preamp-wavenet_46ms/version_4/checkpoints/last.ckpt
ce731608e6e840c1ca7e2a34e9532ca9f00aa7dc10311668ba2d5f6177f599e5       2545997  lightning_logs_40ep_noWarmup/cabinet-ampformer_187ms/version_0/checkpoints/epoch=39-val_esr=0.03173.ckpt
ce731608e6e840c1ca7e2a34e9532ca9f00aa7dc10311668ba2d5f6177f599e5       2545997  lightning_logs_40ep_noWarmup/cabinet-ampformer_187ms/version_0/checkpoints/last.ckpt
3e4386d44c7e07390c820a65f18d2fb6ea13a8be32d7574c31e9137502e11749       2545997  lightning_logs_40ep_noWarmup/cabinet-ampformer_48ms/version_0/checkpoints/epoch=38-val_esr=0.02640.ckpt
3e4386d44c7e07390c820a65f18d2fb6ea13a8be32d7574c31e9137502e11749       2545997  lightning_logs_40ep_noWarmup/cabinet-ampformer_48ms/version_0/checkpoints/last.ckpt
f9f98ced9de408ec33cd44ceac70af77ffa2be0416693a0d000c4b9bbca6e3e0       3092569  lightning_logs_40ep_noWarmup/cabinet-wavenet_186ms/version_0/checkpoints/epoch=39-val_esr=0.00029.ckpt
f9f98ced9de408ec33cd44ceac70af77ffa2be0416693a0d000c4b9bbca6e3e0       3092569  lightning_logs_40ep_noWarmup/cabinet-wavenet_186ms/version_0/checkpoints/last.ckpt
55e4d2c0e0d23d185d98fcb09b6aa95ae867235270a1db5970283eab5f104c2b       2610649  lightning_logs_40ep_noWarmup/cabinet-wavenet_46ms/version_0/checkpoints/epoch=36-val_esr=0.00123.ckpt
55e4d2c0e0d23d185d98fcb09b6aa95ae867235270a1db5970283eab5f104c2b       2610649  lightning_logs_40ep_noWarmup/cabinet-wavenet_46ms/version_0/checkpoints/last.ckpt
381cdc9da5c2ba06677ec71bff6ca1c1c930ffa004aaee6beb8568e142c25a61       2545997  lightning_logs_40ep_noWarmup/clean-ampformer_187ms/version_0/checkpoints/epoch=35-val_esr=0.00000.ckpt
381cdc9da5c2ba06677ec71bff6ca1c1c930ffa004aaee6beb8568e142c25a61       2545997  lightning_logs_40ep_noWarmup/clean-ampformer_187ms/version_0/checkpoints/last.ckpt
6567a67b9022333458109538b2ffb93447fd01236bbd76c357cae12eca91638b       2545997  lightning_logs_40ep_noWarmup/clean-ampformer_48ms/version_0/checkpoints/epoch=25-val_esr=0.00000.ckpt
6567a67b9022333458109538b2ffb93447fd01236bbd76c357cae12eca91638b       2545997  lightning_logs_40ep_noWarmup/clean-ampformer_48ms/version_0/checkpoints/last.ckpt
5900ae6f6cb4a75a6dde6269eda6dc473f333ae3c2816620e78066538bd577aa       3092569  lightning_logs_40ep_noWarmup/clean-wavenet_186ms/version_0/checkpoints/epoch=12-val_esr=0.00005.ckpt
5900ae6f6cb4a75a6dde6269eda6dc473f333ae3c2816620e78066538bd577aa       3092569  lightning_logs_40ep_noWarmup/clean-wavenet_186ms/version_0/checkpoints/last.ckpt
fc0eb0778a80985be99248b6dc460aebb8d26272fe9e405f9cafe83d72114200       2610649  lightning_logs_40ep_noWarmup/clean-wavenet_46ms/version_0/checkpoints/epoch=21-val_esr=0.00006.ckpt
fc0eb0778a80985be99248b6dc460aebb8d26272fe9e405f9cafe83d72114200       2610649  lightning_logs_40ep_noWarmup/clean-wavenet_46ms/version_0/checkpoints/last.ckpt
fc5477578b5e47e52e09d909cd2954898b603bd37bfcbaf4aa817a104a6d5c9c       2545997  lightning_logs_40ep_noWarmup/metal-ampformer_187ms/version_0/checkpoints/epoch=38-val_esr=0.06972.ckpt
fc5477578b5e47e52e09d909cd2954898b603bd37bfcbaf4aa817a104a6d5c9c       2545997  lightning_logs_40ep_noWarmup/metal-ampformer_187ms/version_0/checkpoints/last.ckpt
50d675be656bea6c51818dab70d40c50102627f611f38d0c6d0ffa4aecb27f8b       2545997  lightning_logs_40ep_noWarmup/metal-ampformer_48ms/version_0/checkpoints/epoch=39-val_esr=0.08165.ckpt
50d675be656bea6c51818dab70d40c50102627f611f38d0c6d0ffa4aecb27f8b       2545997  lightning_logs_40ep_noWarmup/metal-ampformer_48ms/version_0/checkpoints/last.ckpt
094f43931366e9689fbaa0deb9b60b64042a51eb8cda29df25bd600ab8739376       3092569  lightning_logs_40ep_noWarmup/metal-wavenet_186ms/version_0/checkpoints/epoch=39-val_esr=0.00712.ckpt
094f43931366e9689fbaa0deb9b60b64042a51eb8cda29df25bd600ab8739376       3092569  lightning_logs_40ep_noWarmup/metal-wavenet_186ms/version_0/checkpoints/last.ckpt
246e11eb3cb5c00ce8a3bb1f0113241531e4174550f1e635ca46dfcd82a1ed1f       2610649  lightning_logs_40ep_noWarmup/metal-wavenet_46ms/version_0/checkpoints/epoch=32-val_esr=0.01528.ckpt
246e11eb3cb5c00ce8a3bb1f0113241531e4174550f1e635ca46dfcd82a1ed1f       2610649  lightning_logs_40ep_noWarmup/metal-wavenet_46ms/version_0/checkpoints/last.ckpt
d3deac362e742c78e6e2861fd0c9e73315bd1cb9e4c274c36d2a6666fbcd210c       2545997  lightning_logs_40ep_noWarmup/preamp-ampformer_187ms/version_0/checkpoints/epoch=39-val_esr=0.32526.ckpt
d3deac362e742c78e6e2861fd0c9e73315bd1cb9e4c274c36d2a6666fbcd210c       2545997  lightning_logs_40ep_noWarmup/preamp-ampformer_187ms/version_0/checkpoints/last.ckpt
c9e144f39784763d95ba80cd8ad4d88bf41e776bd4221cccbd1adb2a91a183f9       2545997  lightning_logs_40ep_noWarmup/preamp-ampformer_48ms/version_0/checkpoints/epoch=39-val_esr=0.09776.ckpt
c9e144f39784763d95ba80cd8ad4d88bf41e776bd4221cccbd1adb2a91a183f9       2545997  lightning_logs_40ep_noWarmup/preamp-ampformer_48ms/version_0/checkpoints/last.ckpt
73fa3b7c2f9a1675f6c3e3d356514210691d5361136d0d1cdf8513ea28dd4781       3092569  lightning_logs_40ep_noWarmup/preamp-wavenet_186ms/version_0/checkpoints/epoch=34-val_esr=0.05549.ckpt
73fa3b7c2f9a1675f6c3e3d356514210691d5361136d0d1cdf8513ea28dd4781       3092569  lightning_logs_40ep_noWarmup/preamp-wavenet_186ms/version_0/checkpoints/last.ckpt
c1435c73e707d6131f7dad160b0a362ca573fe57a0bec1395d247d9978f98d6c       2610649  lightning_logs_40ep_noWarmup/preamp-wavenet_46ms/version_0/checkpoints/epoch=26-val_esr=0.10184.ckpt
c1435c73e707d6131f7dad160b0a362ca573fe57a0bec1395d247d9978f98d6c       2610649  lightning_logs_40ep_noWarmup/preamp-wavenet_46ms/version_0/checkpoints/last.ckpt
bd9669d9daddb545e960f7780042b3d407d9ad82bce2a8ce3f8b73bf8031b8a0       2545997  lightning_logs_misaligned_grid/cabinet-ampformer_187ms/version_0/checkpoints/epoch=38-val_esr=0.03653.ckpt
bd9669d9daddb545e960f7780042b3d407d9ad82bce2a8ce3f8b73bf8031b8a0       2545997  lightning_logs_misaligned_grid/cabinet-ampformer_187ms/version_0/checkpoints/last.ckpt
bdf589a96f3bfe7ed1e91dbffa8ea888b4a4b74924970f08d6acf957766003ae       2545997  lightning_logs_misaligned_grid/cabinet-ampformer_48ms/version_0/checkpoints/epoch=38-val_esr=0.03662.ckpt
bdf589a96f3bfe7ed1e91dbffa8ea888b4a4b74924970f08d6acf957766003ae       2545997  lightning_logs_misaligned_grid/cabinet-ampformer_48ms/version_0/checkpoints/last.ckpt
1c46495769684102728ebc55bff42890ce29e50d58b7a5228f9c86b7ee2acf44       3092569  lightning_logs_misaligned_grid/cabinet-wavenet_186ms/version_0/checkpoints/epoch=7-val_esr=0.01813.ckpt
1c46495769684102728ebc55bff42890ce29e50d58b7a5228f9c86b7ee2acf44       3092569  lightning_logs_misaligned_grid/cabinet-wavenet_186ms/version_0/checkpoints/last.ckpt
1a4a6781ce17e85aefe825c1f8b00122f1bf07da10190efb4ce05e128ccc805e       2610649  lightning_logs_misaligned_grid/cabinet-wavenet_46ms/version_0/checkpoints/epoch=10-val_esr=0.00997.ckpt
1a4a6781ce17e85aefe825c1f8b00122f1bf07da10190efb4ce05e128ccc805e       2610649  lightning_logs_misaligned_grid/cabinet-wavenet_46ms/version_0/checkpoints/last.ckpt
a389e5e0186d9fa087c8e98f934ce9b1c247831104f046a20bebfad494f88d5a       2545997  lightning_logs_misaligned_grid/clean-ampformer_187ms/version_0/checkpoints/epoch=31-val_esr=0.00000.ckpt
a389e5e0186d9fa087c8e98f934ce9b1c247831104f046a20bebfad494f88d5a       2545997  lightning_logs_misaligned_grid/clean-ampformer_187ms/version_0/checkpoints/last.ckpt
e8d77415d1e11d5ff93f376f09a7d341d6379a00d1b265c5432b4da3709cd433       2545997  lightning_logs_misaligned_grid/clean-ampformer_48ms/version_0/checkpoints/epoch=23-val_esr=0.00000.ckpt
e8d77415d1e11d5ff93f376f09a7d341d6379a00d1b265c5432b4da3709cd433       2545997  lightning_logs_misaligned_grid/clean-ampformer_48ms/version_0/checkpoints/last.ckpt
5bf4af479e0e78fbba25dbcb9b4742f1b445b2839b58ece794d099ac926533bf       3092569  lightning_logs_misaligned_grid/clean-wavenet_186ms/version_0/checkpoints/epoch=31-val_esr=0.00012.ckpt
5bf4af479e0e78fbba25dbcb9b4742f1b445b2839b58ece794d099ac926533bf       3092569  lightning_logs_misaligned_grid/clean-wavenet_186ms/version_0/checkpoints/last.ckpt
90dc52e6760de1edc2f7ed579b656c970010b44e88777bfc866a76f79be8c9c0       2610649  lightning_logs_misaligned_grid/clean-wavenet_46ms/version_0/checkpoints/epoch=16-val_esr=0.00025.ckpt
90dc52e6760de1edc2f7ed579b656c970010b44e88777bfc866a76f79be8c9c0       2610649  lightning_logs_misaligned_grid/clean-wavenet_46ms/version_0/checkpoints/last.ckpt
590b3ea9df7d49b402909ab8105fd2ca7f112a5b4e52ad66bcfa744584147072       2545997  lightning_logs_misaligned_grid/metal-ampformer_48ms/version_0/checkpoints/epoch=39-val_esr=0.06747.ckpt
590b3ea9df7d49b402909ab8105fd2ca7f112a5b4e52ad66bcfa744584147072       2545997  lightning_logs_misaligned_grid/metal-ampformer_48ms/version_0/checkpoints/last.ckpt
0fd2572f40a16eb4db4839c567944583d4e4f6f3728d2b15c9e5bcfb050ed293       3092569  lightning_logs_misaligned_grid/metal-wavenet_186ms/version_0/checkpoints/epoch=4-val_esr=0.11316.ckpt
0fd2572f40a16eb4db4839c567944583d4e4f6f3728d2b15c9e5bcfb050ed293       3092569  lightning_logs_misaligned_grid/metal-wavenet_186ms/version_0/checkpoints/last.ckpt
a3385a602371d6305762972ae7a4367a21426026a047d55bb5fe2101080cb095       2610649  lightning_logs_misaligned_grid/metal-wavenet_46ms/version_0/checkpoints/epoch=5-val_esr=0.15692.ckpt
a3385a602371d6305762972ae7a4367a21426026a047d55bb5fe2101080cb095       2610649  lightning_logs_misaligned_grid/metal-wavenet_46ms/version_0/checkpoints/last.ckpt
cf031d501b867967d722bb1558f1559b2775404faff29bcb5073b8a686f9e3da       2545997  lightning_logs_misaligned_grid/preamp-ampformer_187ms/version_0/checkpoints/epoch=39-val_esr=0.21871.ckpt
cf031d501b867967d722bb1558f1559b2775404faff29bcb5073b8a686f9e3da       2545997  lightning_logs_misaligned_grid/preamp-ampformer_187ms/version_0/checkpoints/last.ckpt
5636c842f49559e45d9eeac74f3f5c686e51097dbb3e8f0d937d9d486440048e       2545933  lightning_logs_misaligned_grid/preamp-ampformer_48ms/version_0/checkpoints/epoch=0-val_esr=1.30754.ckpt
27b8e67d8be68c0a78a136ba4453bde383325d3f84e8b5c27e5e9efa4a80d986       2545997  lightning_logs_misaligned_grid/preamp-ampformer_48ms/version_0/checkpoints/last.ckpt
89a19c8e6c66173cd4276cb9bd43f34e1f1680ba38a63fecb5748711ce95c518       2545997  lightning_logs_misaligned_grid/preamp-ampformer_48ms/version_1/checkpoints/epoch=39-val_esr=0.30781.ckpt
89a19c8e6c66173cd4276cb9bd43f34e1f1680ba38a63fecb5748711ce95c518       2545997  lightning_logs_misaligned_grid/preamp-ampformer_48ms/version_1/checkpoints/last.ckpt
32ffc619867c66402dbee87682558648f19185a77539590b2d85e83cd94b8a11       3092569  lightning_logs_misaligned_grid/preamp-wavenet_186ms/version_0/checkpoints/epoch=29-val_esr=0.01668.ckpt
32ffc619867c66402dbee87682558648f19185a77539590b2d85e83cd94b8a11       3092569  lightning_logs_misaligned_grid/preamp-wavenet_186ms/version_0/checkpoints/last.ckpt
3aca1a06ef64ec7ea4a0671956e35ee824bccc9d89e78e276b01b99f5c6b05df       2610649  lightning_logs_misaligned_grid/preamp-wavenet_46ms/version_0/checkpoints/epoch=36-val_esr=0.05567.ckpt
3aca1a06ef64ec7ea4a0671956e35ee824bccc9d89e78e276b01b99f5c6b05df       2610649  lightning_logs_misaligned_grid/preamp-wavenet_46ms/version_0/checkpoints/last.ckpt
```
