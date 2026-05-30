"""Repro: drive mooncake setup from inside a Ray ASYNC actor (mimics AgentLoopWorker)."""
import os, sys, socket, time, asyncio, ray


def get_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        import subprocess
        return subprocess.check_output(['hostname', '-I']).decode().split()[0]


@ray.remote(num_cpus=1)
class MooncakeAsyncActor:
    """Same shape as AgentLoopWorkerTQ: sync __init__ that calls mooncake setup,
    but async methods present so Ray wraps with an asyncio event loop."""

    def __init__(self, ip):
        import torch
        print(f"[actor] torch imported, GPUs: {torch.cuda.device_count()}", flush=True)
        from mooncake.store import MooncakeDistributedStore
        print(f"[actor] mooncake imported", flush=True)

        self._store = MooncakeDistributedStore()
        SEG = 8 * 1024 ** 3
        BUF = 2 * 1024 ** 3
        print(f"[actor] calling setup ip={ip}", flush=True)
        t0 = time.time()
        ret = self._store.setup(
            ip,
            f"http://{ip}:50050/metadata",
            SEG, BUF, "rdma", "",
            f"{ip}:50051",
        )
        print(f"[actor] setup returned {ret} after {time.time()-t0:.2f}s", flush=True)

    async def heartbeat(self):
        # Async method forces Ray to set up an asyncio event loop in the actor.
        for s in range(30):
            await asyncio.sleep(1)
            print(f"[actor] heartbeat t+{s+1}s", flush=True)
        return "OK"


def main():
    ip = os.environ.get('HEAD_IP') or get_ip()
    n = int(os.environ.get('N_ACTORS', '1'))
    print(f"=== driving {n} ASYNC ray actor(s) calling mooncake setup, ip={ip} ===", flush=True)

    ray.init()
    actors = [MooncakeAsyncActor.remote(ip) for _ in range(n)]
    # Now invoke an async method on each, which triggers asyncio loop spin-up
    futures = [a.heartbeat.remote() for a in actors]
    results = ray.get(futures)
    print(f"=== all actors returned: {results} ===", flush=True)


if __name__ == '__main__':
    main()
