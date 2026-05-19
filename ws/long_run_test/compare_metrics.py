#!/usr/bin/env python3
"""Compare per-step metrics between two long-run experiments (e.g. baseline vs tq).

Each run lives in a directory containing one or more ``launch_YYYYMMDD_HHMMSS.log``
files (a single long-run can span multiple launches because of resumes). For every
run we merge its logs by file mtime ascending; when the same training step appears
in multiple launches the later launch wins (use the data after the resume).

The output is a single PNG grid: one subplot per metric, both runs overlaid.

Example
-------
    python ws/long_run_test/compare_metrics.py \\
        --baseline ws/long_run_test/run_logs_baseline \\
        --tq       ws/long_run_test/run_logs_tq \\
        --output   ws/long_run_test/baseline_vs_tq.png
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# match a `step:N - ` token preceded by a non-word char; this avoids matching the
# `step:` substring inside `training/global_step:N`.
STEP_RE = re.compile(r"(?<![\w/])step:(\d+)\s+-\s+(.*)")

# Curated set of metrics that matter for correctness alignment.
# Ordered so the first two rows (reward / quality) show up at the top of the figure.
# Dropped: critic/score/{max,min} (=+/-1 constants in GRPO with {-1,+1} reward),
# critic/advantages/{max,min} (symmetric clip constants).
# Note: critic/score/mean and critic/rewards/mean are numerically identical when
# use_kl_in_reward=False but diverge once a KL-in-reward penalty is enabled, so
# we keep both as separate panels.
METRICS: List[str] = [
    # Reward / quality — the two headline metrics first (training reward then
    # validation reward), then the related but secondary panels.
    "critic/rewards/mean",
    "val-aux/math_dapo/reward/mean@1",      # sparse: only at every test_freq step
    "critic/score/mean",
    "val-core/math_dapo/acc/mean@1",        # sparse: only at every test_freq step
    # Loss / optimization
    "actor/pg_loss",
    "actor/loss",
    "actor/grad_norm",
    # PPO training dynamics
    "actor/entropy",
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    # Advantage scale
    "critic/advantages/mean",
    # Generation quality
    "response_length/mean",
    "response_length/clip_ratio",
    "response/aborted_ratio",
    # Rollout vs training consistency (numerical alignment)
    "training/rollout_probs_diff_mean",
    "training/rollout_probs_diff_max",
    "rollout_corr/kl",
    "training/rollout_actor_probs_pearson_corr",
]


# Metric name aliases. When ``build_series`` looks up a metric it tries the
# canonical key first, then each alias. The first key with any data in a given
# run wins for that run — so baseline and tq can use different names and still
# end up on the same panel.
#
# Background for the val reward alias: an older verl trainer (used by the tq
# run on 5/8) emitted validation reward as ``val-core/<src>/reward/mean@1``.
# A newer trainer (used by baseline on 5/13) splits it into
# ``val-aux/<src>/reward/mean@1`` plus a separate ``val-core/<src>/acc/mean@1``.
# The two ``reward/mean@1`` series are numerically the same metric (mean reward
# over @1 sample per validation prompt) so aliasing them is safe.
METRIC_ALIASES: Dict[str, List[str]] = {
    "val-aux/math_dapo/reward/mean@1": ["val-core/math_dapo/reward/mean@1"],
}


def parse_log(path: str) -> Dict[int, Dict[str, float]]:
    """Parse a single launch log; return {step: {metric: value}}."""
    out: Dict[int, Dict[str, float]] = {}
    with open(path, errors="ignore") as f:
        for raw in f:
            line = ANSI_RE.sub("", raw)
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            body = m.group(2).rstrip("\n")
            for chunk in body.split(" - "):
                chunk = chunk.strip()
                if not chunk or ":" not in chunk:
                    continue
                key, _, val = chunk.partition(":")
                key = key.strip()
                val = val.strip()
                try:
                    fval = float(val)
                except ValueError:
                    continue
                out.setdefault(step, {})[key] = fval
    return out


def merge_logs(log_dir: str) -> Dict[int, Dict[str, float]]:
    """Merge launch_*.log under ``log_dir``; later launches overwrite earlier ones.

    Files are sorted by mtime ascending so that ``dict.update`` (last write wins)
    yields the "data after the resume" for duplicate steps.
    """
    files = sorted(
        glob.glob(os.path.join(log_dir, "launch_*.log")),
        key=os.path.getmtime,
    )
    if not files:
        raise FileNotFoundError(f"no launch_*.log under {log_dir}")
    merged: Dict[int, Dict[str, float]] = {}
    print(f"  merging {len(files)} file(s) from {log_dir}:")
    for f in files:
        per_file = parse_log(f)
        merged.update(per_file)
        print(f"    + {os.path.basename(f):40s}  steps={len(per_file)}")
    return merged


def build_series(
    merged: Dict[int, Dict[str, float]], metric: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) for ``metric``; falls back through METRIC_ALIASES."""
    candidates: Sequence[str] = [metric, *METRIC_ALIASES.get(metric, [])]
    for cand in candidates:
        steps = sorted(s for s, fields in merged.items() if cand in fields)
        if steps:
            return np.array(steps), np.array([merged[s][cand] for s in steps])
    return np.array([]), np.array([])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="dir containing baseline launch_*.log")
    ap.add_argument("--tq",       required=True, help="dir containing tq launch_*.log")
    ap.add_argument("--output",   default="baseline_vs_tq.png", help="output PNG path")
    ap.add_argument("--metrics",  nargs="*", default=None,
                    help="override the metric list (space-separated keys)")
    ap.add_argument("--ncols",    type=int, default=3, help="subplot grid columns")
    ap.add_argument("--max-step", type=int, default=None,
                    help="clip x-axis to step <= MAX_STEP for both runs")
    args = ap.parse_args()

    print("parsing baseline…")
    baseline = merge_logs(args.baseline)
    print("parsing tq…")
    tq = merge_logs(args.tq)
    print(f"baseline: {len(baseline)} steps, range "
          f"[{min(baseline)}, {max(baseline)}]")
    print(f"tq:       {len(tq)} steps, range "
          f"[{min(tq)}, {max(tq)}]")

    metrics: List[str] = args.metrics or METRICS

    n = len(metrics)
    ncols = max(1, args.ncols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 3.0 * nrows), squeeze=False,
    )

    def _is_sparse(x: np.ndarray) -> bool:
        """Mark a series sparse if it has <=15 points or density <25% of its span.

        Sparse series (e.g. val-* metrics logged only every test_freq steps) get
        rendered with markers so the few points are visible.
        """
        if len(x) <= 15:
            return True
        span = float(x[-1] - x[0] + 1)
        return len(x) < 0.25 * span if span > 0 else True

    legend_drawn = False
    for i, metric in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        bx, by = build_series(baseline, metric)
        tx, ty = build_series(tq, metric)
        if args.max_step is not None:
            mask_b = bx <= args.max_step
            bx, by = bx[mask_b], by[mask_b]
            mask_t = tx <= args.max_step
            tx, ty = tx[mask_t], ty[mask_t]
        sparse = _is_sparse(bx) or _is_sparse(tx)
        kwargs_b = dict(label="baseline", color="C0", lw=1.0)
        kwargs_t = dict(label="tq",       color="C1", lw=1.0, alpha=0.85)
        if sparse:
            kwargs_b.update(marker="o", markersize=4)
            kwargs_t.update(marker="s", markersize=4)
        if len(bx):
            ax.plot(bx, by, **kwargs_b)
        if len(tx):
            ax.plot(tx, ty, **kwargs_t)
        ax.set_title(metric, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=8)
        if not legend_drawn and (len(bx) or len(tx)):
            ax.legend(fontsize=8, loc="best")
            legend_drawn = True

    # Hide unused axes when n is not a perfect multiple of ncols.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("baseline vs tq long-run metrics", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.output, dpi=120)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
