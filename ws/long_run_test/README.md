# Long-run baseline vs TransferQueue alignment test

Correctness experiment comparing the **legacy Ray-object-store baseline** and
the **TransferQueue (TQ)** path: do they produce the same training
trajectory on a real RL job over hundreds of steps?

This README is the onboarding doc for new sessions. If you're an AI dropping
in fresh, read this top-to-bottom before touching anything.

## Goal

Train Qwen3-8B on `dapo-math-17k` with GRPO, validate on `aime-2024` every
25 steps, and overlay per-step metrics from both runs. If the curves line up
within stochastic noise, TQ is correct.

**Target endpoint:** both runs reach `global_step ≥ 200`, then
`compare_metrics.py` renders the final overlay PNG.

## Files in this dir

| File / dir | What it is |
|---|---|
| `acc_text_7b_baseline.sh` | Launch script for the legacy path. Calls `verl.trainer.main_ppo`. |
| `acc_text_7b_tq.sh` | Launch script for the TQ path. Calls `verl.trainer.main_ppo_sync` and brings up `mooncake_master` first. |
| `compare_metrics.py` | Parses logs from both runs and renders a subplot grid overlay PNG. |
| `run_logs_baseline/launch_*.log` | One log per launch — a single long-run spans many launches (see slurm note below). |
| `run_logs_tq/launch_*.log` | Same for TQ. |
| `baseline_vs_tq.png` | Current comparison plot. |
| `README.md` | This file. |

Checkpoints (not in this dir):
`checkpoint/pinjiex_dapo_longrun_0508/qwen3-8B-grpo-megatron-vllm-{legacy,tq}/`.
`save_freq=25` and `resume_mode=auto`, so every fresh launch continues from
the latest `global_step_*` directory automatically.

## Environment & required skills

The jump node has filesystem + git only. **No CUDA, no torch, no ray.**
Everything that needs GPUs runs inside an enroot container on a
slurm-allocated compute node.

Use the **`cn` skill** for any command that needs the dev environment:

```bash
cn 'ray status'
cn 'ps -ef | grep -E "main_ppo|mooncake_master" | grep -v grep'
cn 'cd /lustre/.../verl && bash ws/long_run_test/acc_text_7b_baseline.sh'
```

The container's default cwd is `/lustre/fsw/coreai_devtech_hugectr/pinjiex/`,
not the verl repo — always `cd $VERL && …` first.

### Slurm 2 h time limit

The interactive partition kills jobs at exactly 2 h. Effects you will see:
- Training process disappears mid-step at ~2 h from job start.
- `ray status` from inside the dying job goes silent; `cn` errors with
  `no running SLURM job for pinjiex`.
- A fresh slurm allocation re-creates the ray cluster from scratch (node
  IDs in `ray status` change).

**Re-allocate yourself.** From the verl repo root on the jump node:

```bash
bash submit_ray.sh 2     # submits sbatch slurm/start_ray.sh 2
```

This is the same as the user typing it manually. The script returns
immediately; the slurm job needs a few minutes to be allocated and
`start_ray.sh` takes another ~3-5 min to bring up ray inside the container.
Poll `squeue --me` for `R` state, then poll `cn 'ray status'` until both
nodes are active before re-launching the training script. `resume_mode=auto`
picks up from the last checkpoint (loses at most 24 steps).

## How to start (or restart) a run

```bash
VERL=/lustre/fsw/coreai_devtech_hugectr/pinjiex/workspace/rl/verl
SIDE=baseline   # or tq
TS=$(date +%Y%m%d_%H%M%S)
LOG=$VERL/ws/long_run_test/run_logs_${SIDE}/launch_${TS}.log
cn "cd $VERL && nohup bash ws/long_run_test/acc_text_7b_${SIDE}.sh \
    > $LOG 2>&1 < /dev/null & disown"
```

Notes:
- The `cn` / ssh call exits **255** after `disown`. That is expected — the
  detached bash + python keep running inside the container.
- Verify it's alive with:
  ```bash
  cn "ps -ef | grep -E 'acc_text_7b_${SIDE}|main_ppo' | grep -v grep | head -3"
  ```
- TQ also spawns a `mooncake_master` on the head node — check that PID too.

## How to read progress

Logs are on lustre — read directly from the jump node, no `cn` needed.

Last step reached for a given side:
```bash
SIDE=tq
grep -oE "step:[0-9]+" ws/long_run_test/run_logs_${SIDE}/launch_*.log \
  | sort -u -t: -k2 -n | tail -3
```

Latest saved checkpoint (definitive source of truth for `resume_mode=auto`):
```bash
cat checkpoint/pinjiex_dapo_longrun_0508/qwen3-8B-grpo-megatron-vllm-legacy/latest_checkpointed_iteration.txt
cat checkpoint/pinjiex_dapo_longrun_0508/qwen3-8B-grpo-megatron-vllm-tq/latest_checkpointed_iteration.txt
```

## Alignment metrics

Defined in `compare_metrics.py::METRICS`, in plot order:

- **Reward / quality (top of figure)** — `critic/rewards/mean`,
  `val-aux/math_dapo/reward/mean@1`, `critic/score/mean`,
  `val-core/math_dapo/acc/mean@1`.
- **Loss / optimization** — `actor/pg_loss`, `actor/loss`, `actor/grad_norm`.
- **PPO dynamics** — `actor/entropy`, `actor/ppo_kl`, `actor/pg_clipfrac`.
- **Advantage scale** — `critic/advantages/mean`.
- **Generation quality** — `response_length/mean`,
  `response_length/clip_ratio`, `response/aborted_ratio`.
- **Rollout vs training consistency** — `training/rollout_probs_diff_mean`,
  `training/rollout_probs_diff_max`, `rollout_corr/kl`,
  `training/rollout_actor_probs_pearson_corr`.

### Quirks

- **Val-metric renaming.** TQ (2026-05-08 code) emits
  `val-core/math_dapo/reward/mean@1`. Baseline (2026-05-13 code) emits
  `val-aux/math_dapo/reward/mean@1` plus a separate
  `val-core/math_dapo/acc/mean@1`. They are the same underlying numbers —
  `compare_metrics.py::METRIC_ALIASES` maps the two names onto one panel.
- **TQ has no native `val acc` panel** (it only logs `val reward`). The
  `val-core/.../acc/mean@1` panel is baseline-only.
- `critic/rewards/mean == critic/score/mean` as long as
  `use_kl_in_reward=False` (current setting). Both are plotted intentionally
  so a future KL-in-reward run would show divergence here.

## Rendering the plot

```bash
python3 ws/long_run_test/compare_metrics.py \
    --baseline ws/long_run_test/run_logs_baseline \
    --tq       ws/long_run_test/run_logs_tq \
    --output   ws/long_run_test/baseline_vs_tq.png
```

Useful flags: `--max-step 200` (clip x-axis), `--metrics k1 k2 …`
(override metric list), `--ncols N`.

Merge rule for multiple `launch_*.log` per side: files sorted by mtime
ascending, then `dict.update` per step → **for duplicate steps the later
launch wins** (i.e. "data after the resume" replaces data before).

## Workflow for the headline 200-step comparison

1. Run **TQ** in 2 h slots, re-launching after each slurm cycle, until TQ's
   `latest_checkpointed_iteration.txt` ≥ 200.
2. Switch to **baseline**, same loop, until baseline ≥ 200.
3. Render the final overlay with `--max-step 200`.
4. Inspect: curves should overlay within stochastic noise. Systematic drift
   in any panel is a red flag and should be investigated before declaring TQ
   correct.

When the user says "切换" / "switch", swap the side that runs next; the
checkpoints + `resume_mode=auto` make the swap free.

### Self-driven loop (preferred)

Claude should run this loop autonomously rather than waiting on the user
each cycle:

1. On each scheduled wakeup (~30 min cadence), check `squeue --me` and the
   current side's `latest_checkpointed_iteration.txt`.
2. If slurm is empty → `bash submit_ray.sh 2`, poll until ray is up, then
   re-launch the same side.
3. If the current side reached its target step → swap to the other side
   (re-launch via `cn` with the other `acc_text_7b_*.sh`).
4. If both sides reached the target → render the final plot with
   `--max-step 200` and report.

Schedule the next wakeup at the end of each tick. Don't stop polling until
both sides hit 200.

## Don't touch

- `checkpoint/.../qwen3-8B-grpo-megatron-vllm-{legacy,tq}/global_step_*` —
  required for resume.
- `run_logs_*/launch_*.log` — `compare_metrics.py`'s merge logic depends on
  the full set of mtime-ordered launches. Failed launches with 0 steps are
  fine to keep; they're auto-skipped.

## Collaboration rules (from `.claude/CLAUDE.md`)

- The user is on the jump node, the AI runs on the jump node too. GPU work
  goes through `cn`.
- For Debug / feature dev / perf testing / correctness testing requests,
  propose a plan with options first; only implement after the user agrees.
- Working language is Chinese; technical terms / commands stay English.
