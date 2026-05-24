"""Pie chart of baseline DataProto payload composition (per step, per phase).

Numbers per step 1, SEND→old_log_prob:
  * directly measured by _log_transfer_size (tensor batch only): 862 MB
  * pixel_values in non_tensor_batch (estimated from dataset sampling):
      median per-prompt 16 MB × 128 prompts = ~2.0 GB (fp32)

Categories:
  * multimodal     : pixel_values + extra padding inflation that VL forces on
                     prompts/attention_mask/position_ids
  * router_replay  : routed_experts uint8 (MoE topk per token per layer)
  * trajectory     : responses, masks, log_probs, indices, advantages
"""

from pathlib import Path
import matplotlib.pyplot as plt


# Bytes are in MB
CATEGORIES = {
    "Multimodal\n(pixel_values + padding inflation)": {
        "value_mb": 2000 + 276,   # 2.0 GB pixel_values + (311 - 35) MB padding inflation
        "color": "#c1272d",
        "explode": 0.04,
        "detail": "pixel_values ~2 GB (median, fp32)\n+ VL-inflated prompt/attn/pos ~276 MB",
    },
    "Router replay\n(routed_experts uint8)": {
        "value_mb": 467,
        "color": "#e76f51",
        "explode": 0.04,
        "detail": "48 MoE layers × topk 8 × 1.28M tokens",
    },
    "Trajectory\n(responses, masks, log_probs, ...)": {
        "value_mb": 120,
        "color": "#3a7ca5",
        "explode": 0.0,
        "detail": "pure text ids + masks + scores + indices",
    },
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    labels = list(CATEGORIES.keys())
    sizes = [c["value_mb"] for c in CATEGORIES.values()]
    colors = [c["color"] for c in CATEGORIES.values()]
    explode = [c["explode"] for c in CATEGORIES.values()]
    total = sum(sizes)

    def autopct(pct):
        mb = pct * total / 100
        if mb >= 1000:
            return f"{pct:.1f}%\n{mb/1024:.2f} GB"
        return f"{pct:.1f}%\n{mb:.0f} MB"

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct=autopct, startangle=90, counterclock=False,
        pctdistance=0.72,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10.5),
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")
        t.set_fontsize(10)

    ax.set_title(
        f"verl baseline · payload per phase transfer ≈ {total/1024:.2f} GB\n"
        f"Qwen3-VL-30B-A3B-Instruct · bsz=128 × n=8 · step 1",
        fontsize=12.5, fontweight="bold", pad=18,
    )

    # Footnote with caveats
    foot = (
        "Multimodal share estimated from dataset analysis (sampled 150 images + 80 "
        "videos in OneThinker, applied Qwen3-VL processor at fp32, fps=2).\n"
        "Per-video size is long-tailed (10 MB → 1.98 GB); median used. "
        "bfloat16 would halve the multimodal slice."
    )
    fig.text(0.5, 0.02, foot, ha="center", fontsize=8.2, style="italic", color="#555")

    plt.subplots_adjust(top=0.86, bottom=0.13)

    out = Path(__file__).parent / "baseline_payload_pie.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved {out}  (total {total/1024:.2f} GB)")


if __name__ == "__main__":
    main()
