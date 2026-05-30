"""Add verl + AgentLoopWorker imports to see if they break mooncake."""
import os, sys, socket, time, ray


def get_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        import subprocess
        return subprocess.check_output(['hostname', '-I']).decode().split()[0]


@ray.remote(num_cpus=1)
class MooncakeWithVerlActor:
    def __init__(self, ip):
        print("[actor] step A: import torch", flush=True)
        import torch
        print(f"[actor] step A done. GPUs={torch.cuda.device_count()}", flush=True)

        print("[actor] step B: import verl modules", flush=True)
        # Try to import the same heavy verl stuff AgentLoopWorker imports
        try:
            from verl.experimental.agent_loop.agent_loop import (
                AgentLoopWorker, AsyncLLMServerManager, RolloutTraceConfig,
            )
            print("[actor] step B done.", flush=True)
        except Exception as e:
            print(f"[actor] step B FAILED: {type(e).__name__}: {e}", flush=True)

        print("[actor] step C: import transfer_queue", flush=True)
        try:
            import transfer_queue as tq
            print("[actor] step C done.", flush=True)
        except Exception as e:
            print(f"[actor] step C FAILED: {type(e).__name__}: {e}", flush=True)

        print("[actor] step D: import mooncake", flush=True)
        from mooncake.store import MooncakeDistributedStore
        print("[actor] step D done.", flush=True)

        print("[actor] step E: mooncake setup", flush=True)
        self._store = MooncakeDistributedStore()
        t0 = time.time()
        ret = self._store.setup(
            ip,
            f"http://{ip}:50050/metadata",
            8 * 1024 ** 3, 2 * 1024 ** 3, "rdma", "",
            f"{ip}:50051",
        )
        print(f"[actor] step E done ret={ret} elapsed={time.time()-t0:.2f}s", flush=True)

    async def heartbeat(self):
        import asyncio
        for s in range(30):
            await asyncio.sleep(1)
            print(f"[actor] heartbeat t+{s+1}s", flush=True)
        return "OK"


def main():
    ip = os.environ.get('HEAD_IP') or get_ip()
    print(f"=== test: verl imports + mooncake setup in async ray actor, ip={ip} ===", flush=True)
    ray.init()
    actor = MooncakeWithVerlActor.remote(ip)
    print(ray.get(actor.heartbeat.remote()))


if __name__ == '__main__':
    main()
