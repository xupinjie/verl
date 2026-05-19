"""3-way accuracy + per-step wall-time alignment plot.

Lines:
  - legacy baseline (run_logs_baseline/)
  - TQ no-GDR       (run_logs_tq/)
  - TQ-GDR          (run_logs_tq_gdr_keep/launch_20260519_030602.log)

Three panels: training reward, validation reward, per-step wall time.
"""

import sys
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"(?<![\w/])step:(\d+)\s+-\s+(.*)")
ALIASES = {"val-aux/math_dapo/reward/mean@1": ["val-core/math_dapo/reward/mean@1"]}

# tqdm progress line:
#   Training Progress:  10%|█         | 5/50 [06:42<58:48, 78.41s/it]
PROGRESS_RE = re.compile(
    r"Training Progress:\s+\d+%\|[^|]*\|\s+(\d+)/\d+\s+\[(\d+:\d+(?::\d+)?)<"
)


def _to_seconds(s):
    parts = list(map(int, s.split(":")))
    while len(parts) < 3:
        parts = [0, *parts]
    h, m, sec = parts
    return h * 3600 + m * 60 + sec


def parse_step_times(path):
    """Return {step: per_step_seconds} parsed from tqdm Training Progress lines.

    Cumulative elapsed at step N gives wall time AFTER N is done; per-step delta
    is elapsed(N) - elapsed(N-1).
    """
    cum = {}
    with Path(path).open(errors="ignore") as f:
        for line in f:
            line = ANSI.sub("", line)
            for m in PROGRESS_RE.finditer(line):
                step = int(m.group(1))
                t = _to_seconds(m.group(2))
                if step not in cum or t > cum[step]:
                    cum[step] = t
    return {n: cum[n] - cum.get(n - 1, 0) for n in cum if cum[n] - cum.get(n - 1, 0) > 0}


def parse_step_times_dir(dir_path):
    """Merge per-step times across multiple resume launches."""
    merged = {}
    for f in sorted(Path(dir_path).glob("launch_*.log"), key=lambda p: p.stat().st_mtime):
        # Each launch resets its tqdm clock to 0; the step numbers inside are absolute.
        # Compute per-step deltas within the launch and merge.
        per = parse_step_times(f)
        merged.update(per)  # later launch wins on overlap
    return merged


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


def merge_dir(dir_path):
    files = sorted(Path(dir_path).glob("launch_*.log"), key=lambda p: p.stat().st_mtime)
    merged = {}
    for f in files:
        merged.update(parse_file(f))
    return merged


def series(merged, metric, max_step=None):
    for cand in [metric, *ALIASES.get(metric, [])]:
        steps = sorted(s for s, fields in merged.items() if cand in fields)
        if steps:
            x = np.array(steps)
            y = np.array([merged[s][cand] for s in steps])
            if max_step is not None:
                m = x <= max_step
                x, y = x[m], y[m]
            return x, y
    return np.array([]), np.array([])


METRICS = ["critic/rewards/mean", "val-aux/math_dapo/reward/mean@1"]
MAX_STEP = 50

BASE_DIR = "ws/long_run_test/run_logs_baseline"
TQ_DIR = "ws/long_run_test/run_logs_tq"
GDR_LOG = "ws/long_run_test/run_logs_tq_gdr_keep/launch_20260519_030602.log"


def main():
    out = Path(sys.argv[1])
    metric_runs = [
        ("legacy baseline", merge_dir(BASE_DIR),                 "#d62728", "^"),
        ("TQ no-GDR",       merge_dir(TQ_DIR),                   "#666",    "s"),
        ("TQ-GDR",          parse_file(GDR_LOG),                 "#1f77b4", "o"),
    ]
    time_runs = [
        ("legacy baseline", parse_step_times_dir(BASE_DIR),       "#d62728", "^"),
        ("TQ no-GDR",       parse_step_times_dir(TQ_DIR),         "#666",    "s"),
        ("TQ-GDR",          parse_step_times(GDR_LOG),            "#1f77b4", "o"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Panels 0 & 1: accuracy metrics.
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        for label, data, color, marker in metric_runs:
            xs, ys = series(data, metric, max_step=MAX_STEP)
            if len(xs) == 0:
                continue
            sparse = len(xs) <= 15 or (xs[-1] - xs[0] + 1) > 4 * len(xs)
            kw = dict(label=label, color=color, lw=1.3)
            if sparse:
                kw.update(marker=marker, markersize=5)
            ax.plot(xs, ys, **kw)
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("step", fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=8)
        if i == 0:
            ax.legend(fontsize=8, loc="lower right")

    # Panel 2: per-step wall-clock time (seconds).
    # The y-axis is clipped to the steady-state band so checkpoint / validation
    # spikes (~160-170s at step 10/25/50) don't crush the resolution of the
    # actual 75-85s body of the data.
    ax = axes[2]
    for label, times, color, marker in time_runs:
        if not times:
            continue
        steps = sorted(s for s in times if s <= MAX_STEP and s > 1)
        ys = [times[s] for s in steps]
        ax.plot(steps, ys, label=label, color=color, lw=1.3,
                marker=marker, markersize=4, alpha=0.85)
    ax.set_title("per-step wall time (s, clipped — spikes are val/ckpt)", fontsize=10)
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("seconds", fontsize=9)
    ax.set_ylim(72, 95)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=8)

    fig.suptitle(
        f"3-way alignment over first {MAX_STEP} steps — legacy vs TQ vs TQ-GDR",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
