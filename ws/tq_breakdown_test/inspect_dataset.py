"""Dig into images / videos columns of OneThinker train.parquet.

We want to know:
  * what fraction of samples have images / videos
  * are images inline bytes or external paths?
  * for inline bytes, how big per image
  * for external paths, what resolution / frame count → resulting patch count

Then we can estimate the Qwen3-VL pixel_values payload per batch.
"""

from pathlib import Path
from collections import Counter
import io

import pyarrow.parquet as pq

PARQ = Path("/lustre/fsw/coreai_devtech_hugectr/pinjiex/datas/verl/onethinker_parquet/train.parquet")


def main() -> None:
    pf = pq.ParquetFile(PARQ)
    print(f"file: {PARQ}  ({PARQ.stat().st_size/1024**2:.1f} MB, {pf.metadata.num_rows} rows)")

    # Stream rows. row_group=1 since num_row_groups == num_rows we read in chunks.
    # Easiest: read entire table column-by-column (parquet is column-oriented anyway).
    print("\n== reading 'images' and 'videos' columns ==")
    t = pf.read(columns=["images", "videos", "data_source"])
    images = t.column("images").to_pylist()
    videos = t.column("videos").to_pylist()
    sources = t.column("data_source").to_pylist()

    n = len(images)
    n_with_img = sum(1 for x in images if x)
    n_with_vid = sum(1 for x in videos if x)
    print(f"  samples with non-empty images: {n_with_img}/{n} ({100*n_with_img/n:.1f}%)")
    print(f"  samples with non-empty videos: {n_with_vid}/{n} ({100*n_with_vid/n:.1f}%)")

    # Inline image bytes vs path
    img_inline_bytes = []
    img_paths = []
    img_count_per_sample = []
    for entry in images:
        if not entry:
            continue
        img_count_per_sample.append(len(entry))
        for item in entry:
            b = item.get("bytes")
            p = item.get("path")
            if b:
                img_inline_bytes.append(len(b))
            if p:
                img_paths.append(p)
    print(f"\n  total image refs: {sum(img_count_per_sample)}  (across {n_with_img} samples)")
    print(f"  images per sample: min={min(img_count_per_sample) if img_count_per_sample else 0}  "
          f"max={max(img_count_per_sample) if img_count_per_sample else 0}  "
          f"avg={(sum(img_count_per_sample)/len(img_count_per_sample) if img_count_per_sample else 0):.2f}")
    if img_inline_bytes:
        avg_b = sum(img_inline_bytes) / len(img_inline_bytes)
        print(f"  inline image bytes: {len(img_inline_bytes)} entries, avg={avg_b/1024:.1f} KB, "
              f"max={max(img_inline_bytes)/1024:.1f} KB")
    if img_paths:
        print(f"  external image paths: {len(img_paths)} entries; first 5:")
        for p in img_paths[:5]:
            print(f"    {p}")

    # Video info
    vid_paths = []
    vid_count = []
    for entry in videos:
        if not entry:
            continue
        vid_count.append(len(entry))
        for item in entry:
            v = item.get("video")
            if v:
                vid_paths.append(v)
    print(f"\n  total video refs: {sum(vid_count)}")
    if vid_paths:
        print(f"  first 5 video paths/URLs:")
        for p in vid_paths[:5]:
            print(f"    {p[:120]}...")

    # Cross-tab with data_source
    print("\n  modality x data_source breakdown:")
    counter = Counter()
    for src, img, vid in zip(sources, images, videos):
        kind = "text"
        if vid:
            kind = "video"
        elif img:
            kind = "image"
        counter[(src, kind)] += 1
    for (src, kind), c in sorted(counter.items(), key=lambda x: -x[1])[:25]:
        print(f"    {src:35s} {kind:6s}  {c:6d}  ({100*c/n:.1f}%)")

    # Inspect one inline image to estimate dimensions
    if img_inline_bytes:
        for entry in images:
            if entry and entry[0].get("bytes"):
                b = entry[0]["bytes"]
                try:
                    from PIL import Image
                    im = Image.open(io.BytesIO(b))
                    print(f"\n  sample inline image: format={im.format} mode={im.mode} size={im.size}")
                except Exception as e:
                    print(f"\n  could not decode sample image: {e}")
                break


if __name__ == "__main__":
    main()
