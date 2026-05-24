"""Verify cuMemGetAddressRange + BUFFER_ID behavior under expandable_segments.

verl enables expandable_segments during training (see verl/utils/device.py
set_expandable_segments + verl/workers/engine_workers.py:720), which is also the
phase when GDR PUT/GET runs. Need to verify the MR cache keying still works.

Two run modes (set via PYTORCH_CUDA_ALLOC_CONF env var BEFORE python starts):
- expandable_segments:False  → regular cudaMalloc segments
- expandable_segments:True   → reserved-VA + cuMemMap segments
"""

import ctypes
import os

import torch

libcuda = ctypes.CDLL("libcuda.so")
try:
    _cuMemGetAddressRange = libcuda.cuMemGetAddressRange_v2
except AttributeError:
    _cuMemGetAddressRange = libcuda.cuMemGetAddressRange
_cuMemGetAddressRange.argtypes = [
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_ulonglong,
]
_cuMemGetAddressRange.restype = ctypes.c_int

libcuda.cuPointerGetAttribute.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_ulonglong,
]
libcuda.cuPointerGetAttribute.restype = ctypes.c_int

CU_POINTER_ATTRIBUTE_BUFFER_ID = 7


def get_seg(ptr):
    base = ctypes.c_ulonglong(0)
    size = ctypes.c_size_t(0)
    rc = _cuMemGetAddressRange(ctypes.byref(base), ctypes.byref(size), ptr)
    return base.value, size.value, rc


def get_bid(ptr):
    bid = ctypes.c_ulonglong(0)
    rc = libcuda.cuPointerGetAttribute(
        ctypes.byref(bid), CU_POINTER_ATTRIBUTE_BUFFER_ID, ptr
    )
    return bid.value, rc


def describe(t, label):
    base, size, rc1 = get_seg(t.data_ptr())
    bid, rc2 = get_bid(t.data_ptr())
    print(
        f"  {label:<32} ptr=0x{t.data_ptr():x}  "
        f"seg=[0x{base:x}, +{size/1024**2:.2f}MB] rc={rc1}  "
        f"buf_id={bid} rc={rc2}"
    )
    return base, size, bid


def main():
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")
    torch.cuda.init()

    # Force expandable_segments to take effect at allocator init time
    cfg = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    print(f"Effective allocator config: {cfg}")
    print(f"torch.cuda.memory.list_gpu_processes()[:80]: {str(torch.cuda.memory.list_gpu_processes())[:80]}")

    print("\n--- Allocate 3 distinct tensors ---")
    a = torch.empty(50 * 256 * 1024, dtype=torch.float32, device="cuda")  # 50MB
    b = torch.empty(50 * 256 * 1024, dtype=torch.float32, device="cuda")  # 50MB
    c = torch.empty(200 * 256 * 1024, dtype=torch.float32, device="cuda")  # 200MB
    sa = describe(a, "a (50MB)")
    sb = describe(b, "b (50MB)")
    sc = describe(c, "c (200MB)")
    print(f"  → distinct segs: a={sa[0]:#x}/{sa[1]}  b={sb[0]:#x}/{sb[1]}  c={sc[0]:#x}/{sc[1]}")
    print(f"  → distinct buf_ids: {len({sa[2], sb[2], sc[2]})}")

    print("\n--- Memory snapshot (segment count + sizes) ---")
    snap = torch.cuda.memory_snapshot()
    print(f"  segments via memory_snapshot: {len(snap)}")
    for i, s in enumerate(snap[:5]):
        print(f"    seg {i}: addr=0x{s.get('address', 0):x} size={s.get('total_size', 0)/1024**2:.2f}MB "
              f"is_expandable={s.get('is_expandable', '?')}")

    print("\n--- Re-alloc same shape, check stability ---")
    del a, b
    a2 = torch.empty(50 * 256 * 1024, dtype=torch.float32, device="cuda")
    b2 = torch.empty(50 * 256 * 1024, dtype=torch.float32, device="cuda")
    sa2 = describe(a2, "a2 (50MB, after del)")
    sb2 = describe(b2, "b2 (50MB, after del)")
    print(f"  → ptr/seg stability:  a==a2 ptr={sa2[0] == sa[0]}  buf_id={sa2[2] == sa[2]}")

    print("\n--- empty_cache test ---")
    del c, a2, b2
    snap_before = torch.cuda.memory_snapshot()
    torch.cuda.empty_cache()
    snap_after = torch.cuda.memory_snapshot()
    print(f"  segments: before_empty_cache={len(snap_before)}  after={len(snap_after)}")
    d = torch.empty(50 * 256 * 1024, dtype=torch.float32, device="cuda")
    sd = describe(d, "d (50MB, after empty_cache)")
    print(f"  → if same VA as sa: same_ptr={sd[0] == sa[0]}  same_buf_id={sd[2] == sa[2]}")


if __name__ == "__main__":
    main()
