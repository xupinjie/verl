"""Plot a phase-level timeline from [TRANSFER_TIMING] log lines.

Usage:
    python scripts/plot_timeline.py BASELINE_LOG TQ_LOG OUT_PNG
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    """Return list of (length_s, kind) where kind in
    {'compute','rollout','postprocess','concat','transfer','other'}.

    Sums of segments may be less than phase_total when no breakdown exists.
    """
    total = phase_data.get("phase_total", 0.0)

    # gen with the new fine-grained breakdown (rollout/postprocess/concat/transfer)
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
        # tq_get + tq_put or transfer_time
        t = phase_data.get("transfer_time", phase_data.get("tq_get", 0.0) + phase_data.get("tq_put", 0.0))
        rest = max(total - c - t, 0.0)
        return [(c, "compute"), (t, "transfer"), (rest, "other")]

    if "tq_get" in phase_data and "tq_put" in phase_data:
        t = phase_data["tq_get"] + phase_data["tq_put"]
        rest = max(total - t, 0.0)
        return [(rest, "other"), (t, "transfer")]

    # phase_total + transfer_time only (legacy fallback): infer compute = total - transfer
    if "transfer_time" in phase_data and total > 0:
        t = phase_data["transfer_time"]
        c = max(total - t, 0.0)
        return [(c, "compute"), (t, "transfer")]

    return [(total, "other")]


SKIP_LABEL_PHASES = {"adv"}  # too small to label

# Visual style per segment kind. Solid for actual compute work; hatched for overhead.
SEGMENT_STYLE = {
    "compute":     dict(alpha=1.00, hatch=None),
    "rollout":     dict(alpha=1.00, hatch=None),
    "postprocess": dict(alpha=0.60, hatch="..."),
    "concat":      dict(alpha=0.80, hatch="xx"),
    "transfer":    dict(alpha=1.00, hatch="///"),
    "other":       dict(alpha=0.30, hatch=None),
}

# Which kinds get a numeric label printed above the bar.
LABEL_KINDS = {"postprocess", "concat", "transfer"}


def draw_row(ax, y, step_data, color_alpha=1.0):
    x = 0.0
    annotations = []  # (center_x, length, kind)
    for phase in PHASES:
        if phase not in step_data:
            continue
        color = PHASE_COLORS[phase]
        for length, kind in phase_segments(phase, step_data[phase]):
            if length <= 0:
                continue
            style = SEGMENT_STYLE.get(kind, SEGMENT_STYLE["other"])
            ax.barh(
                y, length,
                left=x, height=0.7,
                color=color,
                edgecolor="black", linewidth=0.5,
                alpha=style["alpha"] * color_alpha,
                hatch=style["hatch"],
            )
            if kind in LABEL_KINDS and phase not in SKIP_LABEL_PHASES:
                annotations.append((x + length / 2, length, kind))
            x += length

    # Annotate select segments with their seconds value above the bar
    for cx, length, kind in annotations:
        label = f"{length:.2f}s" if length < 1 else f"{length:.1f}s"
        ax.annotate(
            label,
            xy=(cx, y + 0.35),
            xytext=(cx, y + 0.55),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="black",
            arrowprops=dict(arrowstyle="-", lw=0.4, color="gray") if length < 5 else None,
        )


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    baseline_log, tq_log, out_png = sys.argv[1:]
    bl = parse(baseline_log)
    tq = parse(tq_log)

    steps = sorted(set(bl.keys()) & set(tq.keys()))
    if not steps:
        print("no overlapping steps", file=sys.stderr)
        sys.exit(1)

    # Two rows per step plus a small gap between pairs
    row_h = 1.0
    pair_gap = 0.35
    n_steps = len(steps)
    fig, ax = plt.subplots(figsize=(18, (2 * row_h + pair_gap) * n_steps + 1.8))

    y_labels = []
    y_ticks = []
    y_top = (2 * row_h + pair_gap) * n_steps
    for s in steps:
        y_top -= row_h
        draw_row(ax, y_top, bl[s])
        y_labels.append(f"step {s} • baseline")
        y_ticks.append(y_top)
        y_top -= row_h
        draw_row(ax, y_top, tq[s])
        y_labels.append(f"step {s} • TQ")
        y_ticks.append(y_top)
        # separator between step pairs
        if s != steps[-1]:
            ax.axhline(y_top - pair_gap / 2, color="lightgray", linewidth=0.6, linestyle="--")
            y_top -= pair_gap

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("seconds from step start")
    ax.set_xlim(left=0)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    # legend
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

    # annotate per-row total
    for i, (yt, label) in enumerate(zip(y_ticks, y_labels)):
        s = steps[i // 2]
        d = (bl if i % 2 == 0 else tq)[s]
        total = sum(d[p].get("phase_total", 0.0) for p in PHASES if p in d)
        ax.text(total + 4, yt, f"{total:.0f}s", va="center", fontsize=9)

    config_line = (
        "GRPO • OneThinker • Qwen3-VL-30B-A3B-Instruct • 2 nodes × 8 H100 • Megatron (TP=4 CP=2 EP=8) + vLLM (TP=2)"
    )
    ax.set_title(
        f"Per-step phase timeline — baseline (legacy Ray) vs TQ\n{config_line}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
