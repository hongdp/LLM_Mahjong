"""Batched GPU inference for self-play rollouts (perf 2026-08-22).

CPU workers keep running the engine unchanged; their policy.act() becomes a
cross-process RPC: write (planes, scalars, mask, temperature) into a
per-worker shared-memory slot, enqueue the slot id, block on the slot's
Event. One spawned server process drains the queue (first request, then up
to `wait_ms` more or `max_batch`), runs ONE forward on the GPU, samples,
writes (action, logprob) back and sets the events.

Why cross-process instead of an in-process state machine: the game loop
(turn / interrupt windows / chankan) stays byte-for-byte the same code;
the price is ~0.1 ms IPC per decision, negligible against the 5-17 ms
batch-1 CPU latency of a 10-20M model. Workers block while waiting, so
oversubscribing CPU cores with more workers raises the batch size.

Equivalence: sampling moves to the server's GPU RNG, so trajectories are
statistically, not bit-wise, identical to the CPU path.
"""

import os
import queue as _queue
import time
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp   # shared tensors cross process boundaries

mp.set_sharing_strategy("file_system")

from src.agents.dnn.encoder import ACTION_DIM, TILE_TYPES

_CTX = mp.get_context("spawn")

# module-level handles; set in the parent BEFORE forking rollout workers so
# they inherit by fork (no pickling of shared tensors through Pool args,
# which is the fragile path noted in parallel_rollout).
HANDLES: dict = {}


class _Shared:
    def __init__(self, n_slots: int, n_planes: int, n_scalars: int):
        # v2 (2026-08-22): request flags in shared memory + ONE semaphore
        # instead of a pickling Queue: 48 producers contending on a pipe
        # lock cost more than the GPU forward itself.
        self.pending = torch.zeros(n_slots, dtype=torch.int32).share_memory_()
        self.planes = torch.zeros(n_slots, n_planes, TILE_TYPES).share_memory_()
        self.scalars = torch.zeros(n_slots, n_scalars).share_memory_()
        self.mask = torch.zeros(n_slots, ACTION_DIM, dtype=torch.bool).share_memory_()
        self.temp = torch.ones(n_slots).share_memory_()
        self.out_idx = torch.zeros(n_slots, dtype=torch.int64).share_memory_()
        self.out_lp = torch.zeros(n_slots).share_memory_()


def _server_main(shared, req_q, events, state_np, cfg, device, max_batch,
                 wait_ms, ready):
    torch.set_num_threads(2)
    if cfg.get("arch"):
        from src.agents.dnn.arch_zoo import ZOO
        net = ZOO[cfg["arch"]][0]()
    else:
        from src.agents.dnn.net import MahjongPolicyNet
        net = MahjongPolicyNet(channels=cfg["channels"], blocks=cfg["blocks"])
    from src.agents.dnn.net import load_compatible
    load_compatible(net, {k: torch.from_numpy(v) for k, v in state_np.items()})
    net = net.to(device).eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg.get("seed", 0)) * 31337 + 7)
    ready.set()
    wait_s = wait_ms / 1000.0
    try:
        _serve(shared, req_q, events, net, device, max_batch, wait_s, gen)
    except BaseException as e:          # never die silently: unblock workers
        import traceback; traceback.print_exc()
        shared.out_idx[:] = -1
        for ev in events:
            ev.set()
        raise


def _serve(shared, req_q, events, net, device, max_batch, wait_s, gen):
    sem, stop = req_q                       # (Semaphore, Event)
    n_slots = shared.pending.shape[0]
    target = max(1, min(max_batch, n_slots // 2))   # adaptive: half the slots
    # the flag array is mutated by other processes while we scan it: never
    # run multi-threaded torch ops on it (torch.nonzero hit an internal
    # assert); snapshot through numpy instead.
    pend_np = shared.pending.numpy()
    diag = bool(os.environ.get("INFER_DIAG"))
    T = {"wait": 0.0, "drain": 0.0, "gather": 0.0, "fwd": 0.0, "write": 0.0,
         "signal": 0.0, "n_batch": 0, "n_req": 0}
    with torch.no_grad():
        while True:
            sem.acquire()                   # at least one request
            if stop.is_set():
                if diag and T["n_batch"]:
                    nb = T["n_batch"]
                    print("[infer-server] batches", nb, "avg batch", round(T["n_req"] / nb, 1),
                          {k: round(v / nb * 1000, 2) for k, v in T.items()
                           if k not in ("n_batch", "n_req")}, "ms/batch", flush=True)
                return
            t0 = time.perf_counter()
            t_end = time.perf_counter() + wait_s
            while True:
                ids_np = np.flatnonzero(pend_np.copy())
                if len(ids_np) >= target or time.perf_counter() >= t_end:
                    break
                time.sleep(0.0002)
            ids_np = ids_np[:max_batch]
            t1 = time.perf_counter()
            # drain the semaphore for every request we are about to serve
            for _ in range(len(ids_np) - 1):
                sem.acquire()
            pend_np[ids_np] = 0
            t2 = time.perf_counter()
            idx = torch.from_numpy(ids_np.astype(np.int64))
            p = shared.planes[idx].to(device, non_blocking=True)
            s = shared.scalars[idx].to(device, non_blocking=True)
            m = shared.mask[idx].to(device, non_blocking=True)
            t = shared.temp[idx].to(device)
            t3 = time.perf_counter()
            logits = net(p, s, m)
            greedy = t <= 0
            probs = torch.softmax(logits / t.clamp(min=1e-6)[:, None], dim=1)
            probs = torch.nan_to_num(probs, nan=0.0)
            samp = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            act = torch.where(greedy, logits.argmax(1), samp)
            lp = torch.log_softmax(logits, 1).gather(1, act[:, None]).squeeze(1)
            act_c, lp_c = act.cpu(), lp.cpu()
            t4 = time.perf_counter()
            shared.out_idx[idx] = act_c
            shared.out_lp[idx] = lp_c
            t5 = time.perf_counter()
            for i in ids_np.tolist():
                events[i].set()
            if diag:
                t6 = time.perf_counter()
                T["wait"] += t1 - t0; T["drain"] += t2 - t1; T["gather"] += t3 - t2
                T["fwd"] += t4 - t3; T["write"] += t5 - t4; T["signal"] += t6 - t5
                T["n_batch"] += 1; T["n_req"] += len(ids_np)


class InferenceServer:
    """Parent-side handle. start() spawns the GPU process; workers forked
    afterwards find HANDLES populated and build RemotePolicy(rank)."""

    def __init__(self, state_np, cfg, n_slots: int, n_planes: int,
                 n_scalars: int, device: str = "cuda", max_batch: int = 256,
                 wait_ms: float = 4.0):
        self.shared = _Shared(n_slots, n_planes, n_scalars)
        self.req_q = (_CTX.Semaphore(0), _CTX.Event())   # (requests, stop)
        self.events = [_CTX.Event() for _ in range(n_slots)]
        ready = _CTX.Event()
        self.proc = _CTX.Process(
            target=_server_main,
            args=(self.shared, self.req_q, self.events, state_np, cfg, device,
                  max_batch, wait_ms, ready), daemon=True)
        self.proc.start()
        if not ready.wait(timeout=120):
            raise RuntimeError("inference server failed to start")
        HANDLES.update(shared=self.shared, req_q=self.req_q, events=self.events,
                       pid=self.proc.pid)

    def stop(self):
        self.req_q[1].set()
        self.req_q[0].release()
        self.proc.join(timeout=30)
        HANDLES.clear()


class RemotePolicy:
    """Drop-in for net.act() inside a forked worker (one slot per worker)."""

    def __init__(self, rank: int, encoder_variant: str = "v1"):
        self.rank = rank
        self.encoder_variant = encoder_variant
        self.shared = HANDLES["shared"]
        self.req_q = HANDLES["req_q"]
        self.event = HANDLES["events"][rank]

    def eval(self):
        return self

    @torch.no_grad()
    def act(self, planes, scalars, mask, temperature: float = 1.0):
        r = self.rank
        self.shared.planes[r].copy_(planes[0])
        self.shared.scalars[r].copy_(scalars[0])
        self.shared.mask[r].copy_(mask[0])
        self.shared.temp[r] = float(temperature)
        self.event.clear()
        self.shared.pending[r] = 1
        self.req_q[0].release()
        while not self.event.wait(timeout=5.0):
            pid = HANDLES.get("pid")
            if pid is not None:
                try:
                    os.kill(pid, 0)
                except OSError:
                    raise RuntimeError("inference server died")
        out = self.shared.out_idx[r:r + 1].clone()
        if int(out) < 0:
            raise RuntimeError("inference server reported failure")
        return out, self.shared.out_lp[r:r + 1].clone()
