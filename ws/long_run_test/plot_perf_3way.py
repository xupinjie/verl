"""3-way per-step transfer-op timing comparison.

Lines:
  - legacy baseline (Ray Object Store, run_logs_baseline/)
  - TQ no-GDR       (Mooncake KV, use_gdr=false, run_logs_tq/)
  - TQ-GDR+KEEP     (Mooncake KV, use_gdr=true + VERL_TQ_GDR_KEEP_CUDA=1,
                     run_logs_tq_gdr_keep/launch_20260519_030602.log)

Both legacy and TQ are continuous trajectories merged across resume launches.
TQ-GDR+KEEP is a single fresh-start 50-step run.
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r"\[TRANSFER_TIMING\] step=(?P<step>\d+) phase=(?P<phase>[a-z_]+) "
    r"op=(?P<op>[a-z_]+) elapsed_s=(?P<elapsed>[0-9.]+) bytes=\d+ mode=(?P<mode>[a-z]+)"
)

MAX_STEP = 50


def parse(p):
    d = {}
    with Path(p).open() as f:
        for L in f:
            m = LINE_RE.search(L)
            if not m:
                continue
            k = (m["phase"], m["op"], int(m["step"]))
            v = float(m["elapsed"])
            if k not in d or v > d[k]:
                d[k] = v
    return d


def merge_directory(dir_path):
    files = sorted(Path(dir_path).glob("launch_*.log"), key=lambda p: p.stat().st_mtime)
    merged = {}
    for f in files:
        merged.update(parse(f))
    return merged


def series(data, phase, op, max_step=MAX_STEP):
    pts = sorted((s, v * 1000) for (p, o, s), v in data.items() if p == phase and o == op)
    return [(s, v) for s, v in pts if s <= max_step]


PANELS = [
    ("gen.transfer_time",           "gen",          "transfer_time"),
    ("old_log_prob.tq_get",         "old_log_prob", "tq_get"),
    ("old_log_prob.tq_put",         "old_log_prob", "tq_put"),
    ("old_log_prob.transfer_time",  "old_log_prob", "transfer_time"),
    ("adv.tq_get",                  "adv",          "tq_get"),
    ("adv.tq_put",                  "adv",          "tq_put"),
    ("adv.transfer_time",           "adv",          "transfer_time"),
    ("update_actor.transfer_time",  "update_actor", "transfer_time"),
]


def main():
    out = Path(sys.argv[1])
    base_dir = Path("ws/long_run_test/run_logs_baseline")
    tq_dir = Path("ws/long_run_test/run_logs_tq")
    gdr_log = Path("ws/long_run_test/run_logs_tq_gdr_keep/launch_20260519_030602.log")

    baseline = merge_directory(base_dir)
    tq = merge_directory(tq_dir)
    gdr = parse(gdr_log)

    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=False)
    axes = axes.flatten()

    for ax, (title, phase, op) in zip(axes, PANELS):
        # legacy baseline (red)
        pts = series(baseline, phase, op)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="#d62728", linewidth=1.2, marker="^", markersize=3.5,
                    alpha=0.8, label="legacy baseline (Ray Object Store)")

        # TQ no-GDR (gray)
        pts = series(tq, phase, op)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="#666", linewidth=1.2, marker="s", markersize=3.5,
                    alpha=0.8, label="TQ no-GDR")

        # TQ-GDR + KEEP_CUDA (blue)
        pts = series(gdr, phase, op)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="#1f77b4", linewidth=1.8, marker="o", markersize=4,
                    label="TQ-GDR + KEEP_CUDA")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("elapsed (ms)")
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Per-step transfer ops (step 1-{MAX_STEP}) — legacy vs TQ vs TQ-GDR+KEEP_CUDA",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
