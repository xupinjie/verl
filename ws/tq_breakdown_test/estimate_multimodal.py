"""Estimate multimodal payload per training batch for Qwen3-VL-30B + OneThinker.

We compute the bytes added to a DataProto when each batch of 128 prompts (with
n=8 responses) carries pixel_values for its mix of images / videos.

Approach:
  1. Sample image / video files from parquet → measure original dimensions
  2. Apply Qwen3-VL image processor's resize rule
       min_pixels / max_pixels with 28x28 patch alignment
  3. Compute pixel_values bytes:
       (#patches) × patch_size² × in_channels × dtype_size
     where patch_size=14*2 = 28 (Qwen-VL uses 14 in conv + spatial_merge 2)
  4. Sum over a batch with realistic mix (74% image, 26% video)
"""

from pathlib import Path
import io
import random

from PIL import Image
import pyarrow.parquet as pq

PARQ = Path("/lustre/fsw/coreai_devtech_hugectr/pinjiex/datas/verl/onethinker_parquet/train.parquet")
LUSTRE_PREFIX = Path("/lustre/fsw/coreai_devtech_hugectr/pinjiex/datas/verl")
CONTAINER_PREFIX = Path("/data/verl")  # paths in parquet use this prefix

# Qwen3-VL image processor (from preprocessor_config.json + run config)
PATCH_SIZE = 16            # confirmed: preprocessor_config.json patch_size=16
SPATIAL_MERGE = 2          # confirmed: merge_size=2
FACTOR = PATCH_SIZE * SPATIAL_MERGE  # =32; resize aligned to this
MIN_PIXELS = 4 * FACTOR * FACTOR   # 4 visual tokens
MAX_PIXELS = 1280 * FACTOR * FACTOR  # ~1.3M pixels (Qwen-VL convention)
# pixel_values dtype: Qwen-VL ImageProcessorFast returns float32 (see HF code)
DTYPE_BYTES = 4

# Video config: verl default fps=2, max_frames=768; in practice clamped much lower
VIDEO_FPS = 2.0            # confirmed: vision_utils.py default
VIDEO_MAX_FRAMES = 768
TEMPORAL_PATCH = 2         # temporal_patch_size from config

# Batch
BATCH_PROMPTS = 128


def container_to_lustre(p: str) -> Path:
    p = Path(p)
    rel = p.relative_to(CONTAINER_PREFIX)
    return LUSTRE_PREFIX / rel


def round_by_factor(x: float, factor: int) -> int:
    """Match Qwen-VL's smart_resize."""
    return max(factor, round(x / factor) * factor)


def smart_resize(h: int, w: int, factor: int = FACTOR,
                 min_px: int = MIN_PIXELS, max_px: int = MAX_PIXELS) -> tuple[int, int]:
    """Replicate Qwen-VL's smart_resize."""
    if h * w > max_px:
        beta = ((h * w) / max_px) ** 0.5
        h, w = h / beta, w / beta
    elif h * w < min_px:
        beta = (min_px / (h * w)) ** 0.5
        h, w = h * beta, w * beta
    h_r = round_by_factor(h, factor)
    w_r = round_by_factor(w, factor)
    return h_r, w_r


def pixel_values_bytes(h_r: int, w_r: int, n_frames: int = 1) -> int:
    """Bytes of `pixel_values` tensor produced by Qwen-VL processor for one media."""
    # After resize, grid: (h_r/PATCH_SIZE) × (w_r/PATCH_SIZE) spatial patches
    n_h = h_r // PATCH_SIZE
    n_w = w_r // PATCH_SIZE
    # Temporal: for image n_frames=1; for video, frames grouped by TEMPORAL_PATCH
    n_t = max(1, n_frames // TEMPORAL_PATCH)
    n_patches = n_h * n_w * n_t
    # Flat shape: [n_patches, channels(3) * temporal_patch * patch_size * patch_size]
    feat_per_patch = 3 * TEMPORAL_PATCH * PATCH_SIZE * PATCH_SIZE  # = 1176
    return n_patches * feat_per_patch * DTYPE_BYTES


def main() -> None:
    pf = pq.ParquetFile(PARQ)
    t = pf.read(columns=["images", "videos"])
    images = t.column("images").to_pylist()
    videos = t.column("videos").to_pylist()

    img_paths = [e[0]["path"] for e in images if e]
    vid_paths = [e[0]["video"].replace("file://", "") for e in videos if e]
    print(f"corpus: {len(img_paths)} images, {len(vid_paths)} videos")

    random.seed(0)
    sample_imgs = random.sample(img_paths, min(150, len(img_paths)))
    sample_vids = random.sample(vid_paths, min(80, len(vid_paths)))

    # Probe images
    print("\n== probing images ==")
    img_bytes_samples = []
    img_dim_samples = []
    missing = 0
    for p in sample_imgs:
        lp = container_to_lustre(p)
        if not lp.exists():
            missing += 1
            continue
        try:
            with Image.open(lp) as im:
                w, h = im.size
        except Exception:
            missing += 1
            continue
        h_r, w_r = smart_resize(h, w)
        b = pixel_values_bytes(h_r, w_r, n_frames=1)
        img_bytes_samples.append(b)
        img_dim_samples.append((h, w, h_r, w_r))
    print(f"  sampled {len(img_bytes_samples)} (missing/unreadable: {missing})")
    if img_bytes_samples:
        avg = sum(img_bytes_samples) / len(img_bytes_samples)
        mn = min(img_bytes_samples)
        mx = max(img_bytes_samples)
        print(f"  per-image pixel_values bytes: avg={avg/1024**2:.2f} MB  "
              f"min={mn/1024**2:.2f} MB  max={mx/1024**2:.2f} MB")
        # Show a few dim examples
        print("  sample original→resized dims:")
        for h, w, hr, wr in img_dim_samples[:5]:
            print(f"    {w}x{h}  →  {wr}x{hr}  ({hr*wr/1e6:.2f} MP)")

    # Probe videos (just get frame count & dim; resize independent of frame count)
    print("\n== probing videos ==")
    vid_bytes_samples = []
    vid_dim_samples = []
    v_missing = 0
    try:
        import cv2  # noqa: F401
        use_cv2 = True
    except ImportError:
        use_cv2 = False
        print("  WARNING: opencv not available, assuming default 32 frames @ 480x640")
    for p in sample_vids:
        lp = container_to_lustre(p)
        if not lp.exists():
            v_missing += 1
            continue
        if use_cv2:
            import cv2
            cap = cv2.VideoCapture(str(lp))
            if not cap.isOpened():
                v_missing += 1
                continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            duration_s = total_frames / fps if total_frames > 0 else 0
            # Replicate vision_utils default: sample at VIDEO_FPS, capped to total_frames/VIDEO_MAX_FRAMES
            target = int(round(duration_s * VIDEO_FPS))
            n_used = max(2, min(target, total_frames if total_frames > 0 else target, VIDEO_MAX_FRAMES))
            # Round down to even multiple of TEMPORAL_PATCH
            n_used = max(TEMPORAL_PATCH, (n_used // TEMPORAL_PATCH) * TEMPORAL_PATCH)
        else:
            h, w, total_frames, n_used = 480, 640, 0, 16
        h_r, w_r = smart_resize(h, w)
        b = pixel_values_bytes(h_r, w_r, n_frames=n_used)
        vid_bytes_samples.append(b)
        vid_dim_samples.append((h, w, total_frames, n_used, h_r, w_r))
    print(f"  sampled {len(vid_bytes_samples)} (missing/unreadable: {v_missing})")
    if vid_bytes_samples:
        avg = sum(vid_bytes_samples) / len(vid_bytes_samples)
        mn = min(vid_bytes_samples)
        mx = max(vid_bytes_samples)
        print(f"  per-video pixel_values bytes: avg={avg/1024**2:.2f} MB  "
              f"min={mn/1024**2:.2f} MB  max={mx/1024**2:.2f} MB")
        print("  sample dims (orig hxw, total_frames, used_frames):")
        for h, w, tf, nu, hr, wr in vid_dim_samples[:5]:
            print(f"    {w}x{h}  frames={tf}→{nu}  resized={wr}x{hr}")

    # Mixture per batch
    n_img = len(img_paths)
    n_vid = len(vid_paths)
    p_img = n_img / (n_img + n_vid)
    p_vid = n_vid / (n_img + n_vid)
    print(f"\n== batch projection (BATCH_PROMPTS={BATCH_PROMPTS}) ==")
    avg_img = sum(img_bytes_samples) / len(img_bytes_samples)
    avg_vid = sum(vid_bytes_samples) / len(vid_bytes_samples) if vid_bytes_samples else 0
    per_sample_avg = p_img * avg_img + p_vid * avg_vid
    batch_total = BATCH_PROMPTS * per_sample_avg
    print(f"  expected per-prompt pixel_values: "
          f"{p_img*100:.1f}% × {avg_img/1024**2:.2f} MB (image) + "
          f"{p_vid*100:.1f}% × {avg_vid/1024**2:.2f} MB (video) = {per_sample_avg/1024**2:.2f} MB")
    print(f"  expected batch pixel_values total: {batch_total/1024**2:.1f} MB  (dtype=float32)")
    print(f"  same in bfloat16:                  {batch_total/2/1024**2:.1f} MB")
    # Median is more robust than mean for skewed video distribution
    import statistics
    if vid_bytes_samples:
        med_v = statistics.median(vid_bytes_samples)
        med_i = statistics.median(img_bytes_samples)
        per_sample_med = p_img * med_i + p_vid * med_v
        print(f"  median-based per-prompt:           {per_sample_med/1024**2:.2f} MB "
              f"(img median {med_i/1024**2:.2f}, vid median {med_v/1024**2:.2f})")
        print(f"  median-based batch total:          {BATCH_PROMPTS*per_sample_med/1024**2:.1f} MB (fp32)")


if __name__ == "__main__":
    main()
