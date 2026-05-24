"""Per-step phase timeline for N backends, 10 steps shown.

Derived from scripts/plot_timeline.py — original took (baseline_log, tq_log, out)
and rendered 2 rows per step; this one takes a RUNS list and renders one row
per (step, run) so we can directly compare the Mooncake data-path refactor
against `baseline (no TQ)` and `TQ + SimpleStorage`.

Phase / segment legend is unchanged; only the input loading + row layout
changed.

Run:
    python ws/tq_mooncake_datapath_refactor_test/plot_timeline_genbreak.py
Output:
    ws/tq_mooncake_datapath_refactor_test/timeline_genbreak_10step.png
"""

import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

PHASES = ["gen", "old_log_prob", "adv", "update_actor"]
PHASE_COLORS = {
    "gen": "#d62728",
    "old_log_prob": "#1f77b4",
    "adv": "#8c564b",
    "update_actor": "#2ca02c",
}

LINE_RE = re.compile(
    r"\[TRANSFER_TIMING\]\s+step=(\d+)\s+phase=(\w+)\s+op=(\w+)\s+elapsed_s=([\d.]+)\s+bytes=(\d+)\s+mode=(\w+)"
)


def _latest_refactor_log() -> str:
    here = Path(__file__).resolve().parent
    cands = sorted(glob.glob(str(here / "run_logs" / "launch_10step_*.log")),
                   key=os.path.getmtime, reverse=True)
    if not cands:
        raise FileNotFoundError("No launch_10step_*.log under run_logs/")
    return cands[0]


# (label, log_path) — order = row order within each step
RUNS = [
    ("baseline (no TQ)",
     "ws/tq_breakdown_test/run_logs_baseline/launch_10step_20260520_063824.log"),
    ("TQ + SimpleStorage",
     "ws/tq_breakdown_test/run_logs_tq/launch_10step_20260520_110303.log"),
    ("TQ + Mooncake (data-path refactor)",
     _latest_refactor_log()),
]
MAX_STEPS = 10


def parse(log_path):
    """{step: {phase: {op: elapsed_s}}}"""
    data = defaultdict(lambda: defaultdict(dict))
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        step, phase, op, elapsed = int(m.group(1)), m.group(2), m.group(3), float(m.group(4))
        data[step][phase][op] = elapsed
    return dict(data)


def phase_segments(phase, phase_data):
    """Return list of (length_s, kind).

    kind in {'compute','rollout','postprocess','concat','transfer','other'}.
    Sums of segments may be less than phase_total when no breakdown exists.
    """
    total = phase_data.get("phase_total", 0.0)

    if phase == "gen" and "rollout_time" in phase_data:
        rollout = phase_data["rollout_time"]
        pp = phase_data.get("postprocess_time", 0.0)
        concat = phase_data.get("concat_time", 0.0)
        transfer = phase_data.get("transfer_time", 0.0)
        rest = max(total - rollout - pp - concat - transfer, 0.0)
        return [
            (rollout, "rollout"),
            (pp, "postprocess"),
            (concat, "concat"),
            (transfer, "transfer"),
            (rest, "other"),
        ]

    if "compute_time" in phase_data and "transfer_time" in phase_data:
        c = phase_data["compute_time"]
        t = phase_data["transfer_time"]
        rest = max(total - c - t, 0.0)
        return [(c, "compute"), (t, "transfer"), (rest, "other")]

    if "compute_rpc" in phase_data:
        c = phase_data["compute_rpc"]
        t = phase_data.get("transfer_time", phase_data.get("tq_get", 0.0) + phase_data.get("tq_put", 0.0))
        rest = max(total - c - t, 0.0)
        return [(c, "compute"), (t, "transfer"), (rest, "other")]

    if "tq_get" in phase_data and "tq_put" in phase_data:
        t = phase_data["tq_get"] + phase_data["tq_put"]
        rest = max(total - t, 0.0)
        return [(rest, "other"), (t, "transfer")]

    if "transfer_time" in phase_data and total > 0:
        t = phase_data["transfer_time"]
        c = max(total - t, 0.0)
        return [(c, "compute"), (t, "transfer")]

    return [(total, "other")]


SKIP_LABEL_PHASES = {"adv"}

SEGMENT_STYLE = {
    "compute":     dict(alpha=1.00, hatch=None),
    "rollout":     dict(alpha=1.00, hatch=None),
    "postprocess": dict(alpha=0.60, hatch="..."),
    "concat":      dict(alpha=0.80, hatch="xx"),
    "transfer":    dict(alpha=1.00, hatch="///"),
    "other":       dict(alpha=0.30, hatch=None),
}

LABEL_KINDS = {"postprocess", "concat", "transfer"}


def draw_row(ax, y, step_data, color_alpha=1.0):
    x = 0.0
    annotations = []
    for phase in PHASES:
        if phase not in step_data:
            continue
        color = PHASE_COLORS[phase]
        for length, kind in phase_segments(phase, step_data[phase]):
            if length <= 0:
                continue
            style = SEGMENT_STYLE.get(kind, SEGMENT_STYLE["other"])
            ax.barh(
                y, length, left=x, height=0.7,
                color=color, edgecolor="black", linewidth=0.5,
                alpha=style["alpha"] * color_alpha,
                hatch=style["hatch"],
            )
            if kind in LABEL_KINDS and phase not in SKIP_LABEL_PHASES:
                annotations.append((x + length / 2, length, kind))
            x += length

    for cx, length, kind in annotations:
        label = f"{length:.2f}s" if length < 1 else f"{length:.1f}s"
        ax.annotate(
            label,
            xy=(cx, y + 0.35),
            xytext=(cx, y + 0.55),
            ha="center", va="bottom",
            fontsize=7.0, color="black",
            arrowprops=dict(arrowstyle="-", lw=0.4, color="gray") if length < 5 else None,
        )


def main():
    out_png = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else Path(__file__).resolve().parent / "timeline_genbreak_10step.png"

    parsed = [(label, parse(log)) for label, log in RUNS]
    steps = sorted(set.intersection(*[set(d.keys()) for _, d in parsed]))
    steps = [s for s in steps if s <= MAX_STEPS]
    if not steps:
        print("no overlapping steps", file=sys.stderr)
        sys.exit(1)

    n_runs_per_step = len(parsed)
    row_h = 1.0
    pair_gap = 0.55
    n_steps = len(steps)
    fig, ax = plt.subplots(
        figsize=(18, (n_runs_per_step * row_h + pair_gap) * n_steps + 2.0)
    )

    y_labels = []
    y_ticks = []
    y_top = (n_runs_per_step * row_h + pair_gap) * n_steps
    for s in steps:
        for label, d in parsed:
            y_top -= row_h
            draw_row(ax, y_top, d.get(s, {}))
            y_labels.append(f"step {s} • {label}")
            y_ticks.append(y_top)
        if s != steps[-1]:
            ax.axhline(y_top - pair_gap / 2, color="lightgray", linewidth=0.6, linestyle="--")
            y_top -= pair_gap

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("seconds from step start")
    ax.set_xlim(left=0)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    # legends
    phase_handles = [mpatches.Patch(color=PHASE_COLORS[p], label=p) for p in PHASES]
    style_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", label="compute / rollout (solid)"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="...", alpha=0.60, label="postprocess (worker concat)"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="xx", alpha=0.80, label="concat (controller)"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="transfer (Ray / TQ)"),
        mpatches.Patch(facecolor="white", edgecolor="black", alpha=0.30, label="other / overhead"),
    ]
    leg1 = ax.legend(handles=phase_handles, title="phase", loc="upper right", bbox_to_anchor=(1, 1))
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, title="segment", loc="lower right", bbox_to_anchor=(1, 0))

    # per-row total annotation
    for i, (yt, label) in enumerate(zip(y_ticks, y_labels)):
        s = steps[i // n_runs_per_step]
        d = parsed[i % n_runs_per_step][1]
        sd = d.get(s, {})
        total = sum(sd[p].get("phase_total", 0.0) for p in PHASES if p in sd)
        ax.text(total + 4, yt, f"{total:.0f}s", va="center", fontsize=9)

    config_line = (
        "GRPO • OneThinker • Qwen3-VL-30B-A3B-Instruct • 2 nodes × 8 H100 • Megatron (TP=4 CP=2 EP=8) + vLLM (TP=2)"
    )
    ax.set_title(
        f"Per-step phase timeline — 10 steps × 3 backends\n{config_line}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
