#!/usr/bin/env python3
"""Standalone mooncake UT: reproduce the AgentLoopWorker init crash without
verl / Ray / TransferQueue noise.

Usage:
    # First start mooncake_master in a separate terminal:
    #   mooncake_master --http_metadata_server_host=$HOSTIP \
    #                   --http_metadata_server_port=50050 \
    #                   --rpc_port=50051 ...
    # Then:
    MC_LOG_LEVEL=info python3 mooncake_ut.py
"""

import os, sys, socket, time, ctypes


def get_ip():
    h = socket.gethostname()
    try:
        return socket.gethostbyname(h)
    except Exception:
        # fall back: parse from `hostname -I`
        import subprocess
        out = subprocess.check_output(['hostname', '-I']).decode().split()
        return out[0]


def main(action='all'):
    ip = os.environ.get('HEAD_IP') or get_ip()
    print(f"== test config ==", flush=True)
    print(f"   local_hostname = {ip}", flush=True)
    print(f"   metadata_server = http://{ip}:50050/metadata", flush=True)
    print(f"   master_server  = {ip}:50051", flush=True)

    from mooncake.store import MooncakeDistributedStore, ReplicateConfig
    print(f"   mooncake imported", flush=True)

    # Same params as TQ's MooncakeStoreClient
    SEG = int(os.environ.get('SEG_SIZE', 8 * 1024 ** 3))   # 8 GiB
    BUF = int(os.environ.get('BUF_SIZE', 2 * 1024 ** 3))   # 2 GiB
    protocol = os.environ.get('PROTOCOL', 'rdma')
    device_name = ''

    print(f"   global_segment_size = {SEG} ({SEG/1024**3:.1f} GiB)", flush=True)
    print(f"   local_buffer_size  = {BUF} ({BUF/1024**3:.1f} GiB)", flush=True)
    print(f"   protocol           = {protocol}", flush=True)
    print(f"   device_name        = '{device_name}' (auto-discover)", flush=True)
    print()

    print("== step 1: instantiate store ==", flush=True)
    store = MooncakeDistributedStore()
    print(f"   store = {store}", flush=True)

    print("== step 2: setup ==", flush=True)
    t0 = time.time()
    ret = store.setup(
        ip,                                  # local_hostname
        f"http://{ip}:50050/metadata",       # metadata_server
        SEG,                                 # global_segment_size
        BUF,                                 # local_buffer_size
        protocol,                            # protocol
        device_name,                         # device_name
        f"{ip}:50051",                       # master_server_address
    )
    elapsed = time.time() - t0
    print(f"   setup returned {ret} after {elapsed:.2f}s", flush=True)
    if ret != 0:
        sys.exit(f"setup failed with code {ret}")

    print("== step 3: simulate ~20s of idle (we saw segfault ~21s after setup) ==", flush=True)
    for s in range(25):
        time.sleep(1)
        print(f"   alive at t+{s+1}s", flush=True)
        sys.stdout.flush()

    if action == 'all':
        print("== step 4: try a simple put/get ==", flush=True)
        rc = ReplicateConfig()
        rc.with_hard_pin = True

        # Allocate a small buffer
        data = (ctypes.c_uint8 * 1024)(*range(256, 256+1024 if False else 1024))
        ptr = ctypes.addressof(data)
        nbytes = ctypes.sizeof(data)

        # Register the buffer
        store.register_buffer(ptr, nbytes)
        print(f"   register_buffer({ptr}, {nbytes}) OK", flush=True)

        # batch_upsert_from with 1 entry
        rcs = store.batch_upsert_from(["ut_key_0"], [ptr], [nbytes], config=rc)
        print(f"   batch_upsert_from returned: {rcs}", flush=True)

    print("== UT FINISHED OK ==", flush=True)


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'all'
    main(action)
