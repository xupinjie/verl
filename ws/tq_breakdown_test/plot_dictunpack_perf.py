"""3-way reward + wall-time + phase-breakdown plot for the dict-unpack feature.

Lines:
  - baseline       (run_logs_baseline/, 30B-VL + onethinker, no TQ)
  - TQ no-GDR      (run_logs_tq/, TQ + SimpleStorage)
  - TQ-Moon+DictU  (run_logs_tq_mooncake/launch_mixeddict_10step_*, TQ + MooncakeStore
                    with dict-unpack feature: tensors -> RDMA sub-keys,
                    non-tensors pickled into one uint8 blob also via RDMA)

Three panels:
  1. training reward (critic/rewards/mean)
  2. per-step total wall time (s)
  3. phase breakdown - avg over steps 2-3 for all backends (fair comparison)
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"(?<![\w/])step:(\d+)\s+-\s+(.*)")

PHASES = ["gen", "old_log_prob", "update_actor", "update_weights"]
PHASE_COLORS = {
    "gen":             "#5b8def",
    "old_log_prob":    "#f4a261",
    "update_actor":    "#2a9d8f",
    "update_weights":  "#e76f51",
}

# Fair comparison: avg over steps 2 and 3 only (cold-start step 1 dropped),
# matching baseline / TQ+Simple which only have 3 steps total.
AVG_STEPS = (2, 3)
MAX_STEP_SHOW = 10  # x-axis on line panels

RUNS = [
    ("baseline (no TQ)",
     "ws/tq_breakdown_test/run_logs_baseline/launch_10step_20260520_063824.log",
     "#d62728", "^"),
    ("TQ + SimpleStorage",
     "ws/tq_breakdown_test/run_logs_tq/launch_10step_20260520_110303.log",
     "#666666", "s"),
    ("TQ + Mooncake + Dict-unpack",
     "ws/tq_breakdown_test/run_logs_tq_mooncake/launch_mixeddict_10step_20260520_051703.log",
     "#1f77b4", "o"),
]


def parse_file(path):
    out = {}
    with open(path, errors="ignore") as f:
        for line in f:
            line = ANSI.sub("", line)
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            for chunk in m.group(2).rstrip("\n").split(" - "):
                chunk = chunk.strip()
                if not chunk or ":" not in chunk:
                    continue
                k, _, v = chunk.partition(":")
                try:
                    out.setdefault(step, {})[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return out


def series(merged, metric, max_step=None):
    steps = sorted(s for s, fields in merged.items() if metric in fields)
    x = np.array(steps)
    y = np.array([merged[s][metric] for s in steps])
    if max_step is not None and len(x) > 0:
        m = x <= max_step
        x, y = x[m], y[m]
    return x, y


def phase_avg(merged, target_steps=AVG_STEPS):
    """Avg seconds per phase across the specified step list, when available."""
    out = {}
    for phase in PHASES:
        key = f"timing_s/{phase}"
        vals = [merged[s][key] for s in target_steps
                if s in merged and key in merged[s]]
        if vals:
            out[phase] = float(np.mean(vals))
    step_vals = [merged[s]["timing_s/step"] for s in target_steps
                 if s in merged and "timing_s/step" in merged[s]]
    if step_vals:
        out["step_total"] = float(np.mean(step_vals))
    out["n_steps_used"] = len(step_vals)
    return out


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else Path("ws/tq_breakdown_test/dictunpack_perf_3way.png")

    parsed = []
    for label, log, color, marker in RUNS:
        data = parse_file(log)
        parsed.append((label, data, color, marker))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # Panel 1: training reward
    ax = axes[0]
    for label, data, color, marker in parsed:
        xs, ys = series(data, "critic/rewards/mean", max_step=MAX_STEP_SHOW)
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, label=label, color=color, lw=1.5,
                marker=marker, markersize=7, alpha=0.9)
    ax.set_title("training reward (critic/rewards/mean) - higher is better", fontsize=10)
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("mean reward", fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks(range(1, MAX_STEP_SHOW + 1))

    # Panel 2: per-step wall time
    ax = axes[1]
    for label, data, color, marker in parsed:
        xs, ys = series(data, "timing_s/step", max_step=MAX_STEP_SHOW)
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, label=label, color=color, lw=1.5,
                marker=marker, markersize=7, alpha=0.9)
    ax.set_title("per-step wall time (s) - lower is better", fontsize=10)
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("seconds", fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks(range(1, MAX_STEP_SHOW + 1))

    # Panel 3: phase breakdown - avg over steps 2-3 for ALL backends (fair)
    ax = axes[2]
    n_runs = len(parsed)
    x = np.arange(n_runs)
    bottoms = np.zeros(n_runs)
    for phase in PHASES:
        ys = np.array([phase_avg(d).get(phase, 0.0) for _, d, _, _ in parsed])
        ax.bar(x, ys, width=0.55, bottom=bottoms,
               color=PHASE_COLORS[phase], label=phase,
               edgecolor="white", linewidth=0.5)
        for xi, (yi, bi) in enumerate(zip(ys, bottoms)):
            if yi >= 20:
                ax.text(xi, bi + yi / 2, f"{yi:.0f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += ys

    for xi, b in enumerate(bottoms):
        ax.text(xi, b + 8, f"{b:.0f} s", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _, _, _ in parsed], fontsize=8)
    ax.set_ylabel(f"seconds (avg of step {AVG_STEPS[0]}-{AVG_STEPS[1]})", fontsize=9)
    ax.set_title(f"phase breakdown - avg over step {AVG_STEPS[0]}-{AVG_STEPS[1]} (cold start excluded)",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, max(bottoms) * 1.18)

    fig.suptitle(
        "30B-VL + onethinker - Mooncake dict-unpack vs baselines (3-way)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
