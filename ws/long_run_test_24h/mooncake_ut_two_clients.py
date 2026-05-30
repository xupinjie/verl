"""Repro: simulate TaskRunner-style client #1 puts data, then AgentLoopWorker-style client #2 setup.

This is the actual sequence in the failing run:
  1. TaskRunner registers + does PutStart (we saw ~9 PutStart, 234 items)
  2. Several minutes later, AgentLoopWorker starts up, tries to setup mooncake → segfault
"""
import os, sys, socket, time, ctypes, ray


def get_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        import subprocess
        return subprocess.check_output(['hostname', '-I']).decode().split()[0]


@ray.remote(num_cpus=1)
class FirstClient:
    """Simulates TaskRunner: setup + many puts."""

    def __init__(self, ip):
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        self._store = MooncakeDistributedStore()
        ret = self._store.setup(
            ip, f"http://{ip}:50050/metadata",
            8 * 1024 ** 3, 2 * 1024 ** 3, "rdma", "",
            f"{ip}:50051",
        )
        print(f"[client1] setup ret={ret}", flush=True)
        assert ret == 0
        self._rc = ReplicateConfig()
        self._rc.with_hard_pin = True

    def put_many(self, n_puts: int = 10):
        # Allocate a buffer and put many small entries (like TaskRunner does for batch meta)
        data = (ctypes.c_uint8 * 4096)()
        ptr = ctypes.addressof(data)
        nbytes = ctypes.sizeof(data)
        self._store.register_buffer(ptr, nbytes)
        for i in range(n_puts):
            res = self._store.batch_upsert_from([f"c1_key_{i}"], [ptr], [nbytes], config=self._rc)
            print(f"[client1] put {i}: {res}", flush=True)
        return n_puts

    async def stay_alive(self, secs):
        import asyncio
        for s in range(secs):
            await asyncio.sleep(1)
        return "alive"


@ray.remote(num_cpus=1)
class SecondClient:
    """Simulates AgentLoopWorker: setup AFTER first client has put data."""

    def __init__(self, ip):
        import torch  # mirror AgentLoopWorker which imports torch
        print(f"[client2] torch imported, GPUs={torch.cuda.device_count()}", flush=True)

        from mooncake.store import MooncakeDistributedStore
        self._store = MooncakeDistributedStore()
        t0 = time.time()
        print(f"[client2] calling setup", flush=True)
        ret = self._store.setup(
            ip, f"http://{ip}:50050/metadata",
            8 * 1024 ** 3, 2 * 1024 ** 3, "rdma", "",
            f"{ip}:50051",
        )
        print(f"[client2] setup ret={ret} elapsed={time.time()-t0:.2f}s", flush=True)
        assert ret == 0

    async def heartbeat(self):
        import asyncio
        for s in range(35):
            await asyncio.sleep(1)
            print(f"[client2] alive t+{s+1}s", flush=True)
        return "OK"


def main():
    ip = os.environ.get('HEAD_IP') or get_ip()
    print(f"=== two-clients UT: ip={ip} ===", flush=True)
    ray.init()

    # Phase 1: client1 = TaskRunner-style
    print("=== Phase 1: spawn first client (TaskRunner-like) ===", flush=True)
    c1 = FirstClient.remote(ip)
    ray.get(c1.put_many.remote(10))
    print("[main] client1 has put 10 items", flush=True)
    # keep alive
    c1_alive = c1.stay_alive.remote(120)

    # Phase 2: wait a bit (mimic the gap)
    print("=== Phase 2: wait 2 min (mimic gap between TaskRunner setup and AgentLoopWorker spawn) ===", flush=True)
    time.sleep(5)  # shorter for test

    # Phase 3: client2 = AgentLoopWorker-style
    print("=== Phase 3: spawn second client (AgentLoopWorker-like) ===", flush=True)
    c2 = SecondClient.remote(ip)
    print(ray.get(c2.heartbeat.remote()))


if __name__ == '__main__':
    main()
