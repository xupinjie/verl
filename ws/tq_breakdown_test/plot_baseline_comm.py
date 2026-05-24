"""Baseline-only single-step timeline showing communication share.

Numbers come from ws/tq_breakdown_test/transferqueue_speedup_analysis.md
(3-step averages from the 2026-05-07 Qwen3-VL-30B-A3B-Instruct run).

Run:
    python ws/tq_breakdown_test/plot_baseline_comm.py
Output:
    ws/tq_breakdown_test/baseline_step_breakdown.png
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# (label, seconds, kind)  kind in {compute, comm_concat, comm_xfer, comm_pp}
# Segment values are 3-step averages from transferqueue_speedup_analysis.md.
# We deliberately drop the "other / non-RPC residual" buckets:
#   * for gen the 4 segments are each "cross-worker max" so they don't
#     strictly add to phase_total (see analysis §6);
#   * for olp / ua the residual is ~7%, not part of the story.
# Bar width = sum of named segments; comm % is computed from that sum.
PHASES = {
    "gen (~238s)": [
        ("rollout",      178.4, "compute"),
        ("postproc",       2.1, "comm_pp"),
        ("concat",        15.7, "comm_concat"),
        ("ray xfer",      41.6, "comm_xfer"),
    ],
    "old_log_prob (~108s)": [
        ("compute",       64.1, "compute"),
        ("transfer",      44.0, "comm_xfer"),
    ],
    "adv (~11s)": [
        ("compute",       11.0, "compute"),
    ],
    "update_actor (~148s)": [
        ("compute",      101.6, "compute"),
        ("transfer",      46.2, "comm_xfer"),
    ],
}

KIND_STYLE = {
    "compute":     dict(color="#3a7ca5", hatch=None,  label="compute"),
    "comm_pp":     dict(color="#f4a261", hatch="//",  label="postproc (controller wrap-up)"),
    "comm_concat": dict(color="#e76f51", hatch="\\\\",label="DataProto.concat (controller bottleneck)"),
    "comm_xfer":   dict(color="#c1272d", hatch="xx",  label="Ray Object Store transfer"),
}

COMM_KINDS = {"comm_pp", "comm_concat", "comm_xfer"}


def main() -> None:
    fig, (ax, ax_legend) = plt.subplots(
        2, 1, figsize=(15, 5.0),
        gridspec_kw={"height_ratios": [4.5, 1], "hspace": 0.55},
    )

    y = 0
    bar_h = 0.55
    phase_gap = 1.0  # in seconds, visual breathing room between phases

    cursor = 0.0
    phase_bounds = []  # (start, end, name)
    total_comm = 0.0
    total_step = 0.0

    for phase_name, segs in PHASES.items():
        seg_start = cursor
        for label, sec, kind in segs:
            style = KIND_STYLE[kind]
            ax.barh(
                y, sec, left=cursor, height=bar_h,
                color=style["color"], hatch=style["hatch"],
                edgecolor="white", linewidth=0.6,
            )
            # Only annotate segments wide enough to read; let the legend
            # explain everything else.
            if sec >= 30:
                # Hatched (communication) cells use white hatch lines, so
                # white text disappears into them — switch to black.
                txt_color = "black" if kind in COMM_KINDS else "white"
                ax.text(
                    cursor + sec / 2, y,
                    f"{label}\n{sec:.0f}s",
                    ha="center", va="center",
                    fontsize=9, color=txt_color, fontweight="bold",
                )
            elif sec >= 14 and kind in COMM_KINDS:
                # narrow but communication-related: callout below the bar
                ax.annotate(
                    f"{label} {sec:.0f}s",
                    xy=(cursor + sec / 2, y - bar_h / 2),
                    xytext=(cursor + sec / 2, y - bar_h / 2 - 0.45),
                    ha="center", va="top", fontsize=8, color="#444",
                    arrowprops=dict(arrowstyle="-", color="#888", lw=0.5),
                )
            cursor += sec
            if kind in COMM_KINDS:
                total_comm += sec
            total_step += sec
        phase_bounds.append((seg_start, cursor, phase_name))
        cursor += phase_gap  # visual gap (not counted into totals)

    # Phase labels above the bar
    for start, end, name in phase_bounds:
        ax.annotate(
            "", xy=(start, y + bar_h / 2 + 0.18), xytext=(end, y + bar_h / 2 + 0.18),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.7),
        )
        ax.text(
            (start + end) / 2, y + bar_h / 2 + 0.30, name,
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222",
        )

    # Headline callout
    comm_pct = 100 * total_comm / total_step
    headline = (
        f"verl baseline · 1 training step ≈ {total_step:.0f}s     "
        f"│     communication / data-movement = {total_comm:.0f}s  "
        f"({comm_pct:.0f}% of wall-clock)"
    )
    ax.set_title(headline, fontsize=13, fontweight="bold", pad=22, loc="left")
    ax.text(
        0, y + bar_h / 2 + 0.70,
        "Qwen3-VL-30B-A3B-Instruct · 2 nodes × 8 H100 · GRPO bsz=128 n=8 · 3-step avg",
        ha="left", fontsize=9, color="#666", style="italic",
    )

    ax.set_xlim(-2, cursor)
    ax.set_ylim(-1.3, 1.3)
    ax.set_yticks([])
    ax.set_xlabel("time within one step (seconds)", fontsize=10, labelpad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

    # Legend in its own axis underneath
    ax_legend.axis("off")
    handles = [
        mpatches.Patch(
            facecolor=s["color"], hatch=s["hatch"], edgecolor="white", label=s["label"]
        )
        for s in KIND_STYLE.values()
    ]
    ax_legend.legend(
        handles=handles, loc="center", ncol=5, frameon=False, fontsize=9,
        handlelength=2.2, handleheight=1.2, columnspacing=1.6,
    )

    out = Path(__file__).parent / "baseline_step_breakdown.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
