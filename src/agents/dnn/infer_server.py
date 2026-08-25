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
        # multi-model (2026-08-22, user ask): which hosted model a slot wants
        self.model_id = torch.zeros(n_slots, dtype=torch.int32).share_memory_()
        # v3: per-slot done-generation counter; workers wait on ONE shared
        # Condition and check their own counter (one notify_all per batch
        # instead of one Event.set per served request)
        self.done_gen = torch.zeros(n_slots, dtype=torch.int64).share_memory_()
        self.planes = torch.zeros(n_slots, n_planes, TILE_TYPES).share_memory_()
        self.scalars = torch.zeros(n_slots, n_scalars).share_memory_()
        self.mask = torch.zeros(n_slots, ACTION_DIM, dtype=torch.bool).share_memory_()
        self.temp = torch.ones(n_slots).share_memory_()
        self.out_idx = torch.zeros(n_slots, dtype=torch.int64).share_memory_()
        self.out_lp = torch.zeros(n_slots).share_memory_()


def _server_main(shared, req_q, events, state_np, cfg, device, max_batch,
                 wait_ms, ready):
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = bool(int(os.environ.get("INFER_CUDNN_BENCH", "0")))
    # Precision policy: by default keep PyTorch's GPU defaults (TF32 convs)
    # so rollout logprobs match the trainer's GPU update path; set
    # INFER_STRICT_FP32=1 for CPU-vs-GPU equivalence checks.
    if os.environ.get("INFER_STRICT_FP32"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    from src.agents.dnn.encoder import variant_shape
    from src.agents.dnn.net import load_compatible
    if cfg.get("arch"):
        from src.agents.dnn.arch_zoo import ZOO
        net = ZOO[cfg["arch"]][0]()
    else:
        from src.agents.dnn.net import MahjongPolicyNet
        net = MahjongPolicyNet(channels=cfg["channels"], blocks=cfg["blocks"])
    load_compatible(net, {k: torch.from_numpy(v) for k, v in state_np.items()})
    models = [net]                              # model 0 = learner
    if cfg.get("gpu_infer_opponents") and cfg.get("league"):
        from src.agents.dnn.parallel_rollout import _load_policy_ckpt
        for entry in cfg["league"]:
            models.append(_load_policy_ckpt(entry["path"]))
    hosted = []
    for m in models:
        m = m.to(device).eval()
        n_pl, n_sc = variant_shape(getattr(m, "encoder_variant", "v1"))
        # action width is per-model since the action-space adapter (2026-08-24)
        n_act = int(getattr(m, "action_dim", ACTION_DIM))
        if device.startswith("cuda") and cfg.get("infer_cuda_graph", True):
            m = _GraphRunner(m, n_pl, n_sc, device, max_batch,
                             bf16=bool(cfg.get("bf16_infer")), n_act=n_act)
        hosted.append((m, n_pl, n_sc, n_act))
    net = hosted                    # list of (runner, n_planes, n_scalars, n_act)
    widths = {h[3] for h in hosted}
    if len(widths) > 1:
        raise ValueError(
            f"hosted models disagree on action width {sorted(widths)}; a league "
            "pool must share one action space (the batched logits tensor is "
            "allocated once per batch)")
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg.get("seed", 0)) * 31337 + 7)
    ready.set()
    wait_s = wait_ms / 1000.0
    try:
        _serve(shared, req_q, events, net, device, max_batch, wait_s, gen, cfg)
    except BaseException as e:          # never die silently: unblock workers
        import traceback; traceback.print_exc()
        shared.out_idx[:] = -1
        shared.done_gen[:] += 1
        with events:
            events.notify_all()
        raise


def _serve(shared, req_q, events, net, device, max_batch, wait_s, gen, cfg=None):
    sem, stop = req_q                       # (Semaphore, Event)
    n_slots = shared.pending.shape[0]
    # adaptive window: serve once half the request SOURCES have posted.
    # rows_per_worker > 1 (vectorized workers, 2026-08-23): a worker posts
    # ~K rows per round out of its 3K-row block, so "half the slots" was
    # never reached and every batch waited the full window (40 vs 78 games/s).
    rows = int((cfg or {}).get("_rows_per_worker", 1) or 1)
    n_workers = n_slots // rows
    expected_per_worker = max(1, rows // 3) if rows > 1 else 1
    target = max(1, min(max_batch, (n_workers // 2) * expected_per_worker))
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
            mids = shared.model_id[idx]
            t = shared.temp[idx].to(device)
            t3 = time.perf_counter()
            # every hosted model must agree on action width here: a batch mixes
            # rows from different model_ids into one logits tensor. League pools
            # of mixed action spaces are rejected at startup (see below).
            act_dim = net[0][3]
            logits = torch.empty(len(ids_np), act_dim, device=device)
            for mid in torch.unique(mids).tolist():
                sel = torch.nonzero(mids == mid, as_tuple=True)[0]
                runner, n_pl, n_sc, _ = net[mid]
                sub = idx[sel]
                p = shared.planes[sub, :n_pl].to(device, non_blocking=True)
                s = shared.scalars[sub, :n_sc].to(device, non_blocking=True)
                m = shared.mask[sub, :act_dim].to(device, non_blocking=True)
                logits[sel] = runner(p, s, m)
            greedy = t <= 0
            logb = torch.log_softmax(logits / t.clamp(min=1e-6)[:, None], dim=1)
            probs = torch.nan_to_num(logb.exp(), nan=0.0)
            samp = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            act = torch.where(greedy, logits.argmax(1), samp)
            # behaviour logprob: log b(a) with b = softmax(logits/T) (greedy rows: log pi)
            lp = torch.where(greedy[:, None], torch.log_softmax(logits, 1), logb).gather(1, act[:, None]).squeeze(1)
            act_c, lp_c = act.cpu(), lp.cpu()
            t4 = time.perf_counter()
            shared.out_idx[idx] = act_c
            shared.out_lp[idx] = lp_c
            t5 = time.perf_counter()
            shared.done_gen[idx] += 1
            with events:                      # v3: single broadcast
                events.notify_all()
            if diag:
                t6 = time.perf_counter()
                T["wait"] += t1 - t0; T["drain"] += t2 - t1; T["gather"] += t3 - t2
                T["fwd"] += t4 - t3; T["write"] += t5 - t4; T["signal"] += t6 - t5
                T["n_batch"] += 1; T["n_req"] += len(ids_np)


class _GraphRunner:
    """Replays a captured CUDA graph per batch-size bucket. The 192x40-class
    nets are kernel-launch bound (B=1..128 cost the same ~8 ms eager), so
    replaying ~200 launches as one graph is the lever. Inputs are padded
    to the bucket with all-legal dummy rows; only rows[:B] are read back.
    Falls back to eager if capture fails (e.g. a net with data-dependent
    control flow)."""

    BUCKETS = (8, 16, 32, 64, 128, 256)
    MAX_BUCKET_ENV = "INFER_MAX_BUCKET"   # cap buckets on memory-starved GPUs

    def __init__(self, net, n_planes, n_scalars, device, max_batch, bf16=False,
                 n_act=None):
        self.net, self.device = net, device
        # bf16 autocast is baked into the captured graphs; logits are cast
        # back to fp32 by the caller's assignment before log_softmax.
        self.bf16 = bool(bf16)
        cap = int(os.environ.get(self.MAX_BUCKET_ENV, max_batch))
        self.buckets = [b for b in self.BUCKETS if b <= max(min(max_batch, cap), 8)]
        torch.cuda.empty_cache()
        self.static, self.graphs = {}, {}
        ok = True
        try:
            side = torch.cuda.Stream()
            for b in self.buckets:
                p = torch.zeros(b, n_planes, TILE_TYPES, device=device)
                sc = torch.zeros(b, n_scalars, device=device)
                m = torch.ones(b, n_act or ACTION_DIM, dtype=torch.bool, device=device)
                with torch.cuda.stream(side):
                    for _ in range(3):
                        with torch.autocast("cuda", torch.bfloat16, enabled=self.bf16):
                            net(p, sc, m)
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    with torch.autocast("cuda", torch.bfloat16, enabled=self.bf16):
                        out = net(p, sc, m)
                self.static[b] = (p, sc, m, out)
                self.graphs[b] = g
            torch.cuda.synchronize()
        except Exception as e:              # pragma: no cover
            print(f"[infer-server] CUDA graph capture failed ({e}); eager fallback",
                  flush=True)
            ok = False
        self.enabled = ok

    def __call__(self, p, sc, m):
        # logits always leave here fp32: the caller's `logits` buffer and
        # the downstream log_softmax/sampling are fp32 regardless of bf16_infer
        # (bf16 only speeds up the matmuls inside the captured graph).
        B = p.shape[0]
        if not self.enabled:
            with torch.autocast("cuda", torch.bfloat16, enabled=self.bf16):
                return self.net(p, sc, m).float()
        b = next((x for x in self.buckets if x >= B), None)
        if b is None:
            with torch.autocast("cuda", torch.bfloat16, enabled=self.bf16):
                return self.net(p, sc, m).float()
        sp, ssc, sm, out = self.static[b]
        sp[:B].copy_(p); ssc[:B].copy_(sc); sm[:B].copy_(m)
        if B < b:                           # dummy rows: zeros + all-legal
            sp[B:].zero_(); ssc[B:].zero_(); sm[B:].fill_(True)
        self.graphs[b].replay()
        return out[:B].float()


class InferenceServer:
    """Parent-side handle. start() spawns the GPU process; workers forked
    afterwards find HANDLES populated and build RemotePolicy(rank)."""

    def __init__(self, state_np, cfg, n_slots: int, n_planes: int,
                 n_scalars: int, device: str = "cuda", max_batch: int = 256,
                 wait_ms: float = 4.0, rows_per_worker: int = 1):
        from src.agents.dnn.encoder import MAX_PLANES, MAX_SCALARS
        self.rows_per_worker = rows_per_worker
        n_slots = n_slots * rows_per_worker
        cfg = dict(cfg, _rows_per_worker=rows_per_worker)
        # buffers sized for the widest hosted encoder; each model reads its own width
        self.shared = _Shared(n_slots, max(n_planes, MAX_PLANES), max(n_scalars, MAX_SCALARS))
        self.req_q = (_CTX.Semaphore(0), _CTX.Event())   # (requests, stop)
        self.events = _CTX.Condition()                    # v3: one broadcast condition
        ready = _CTX.Event()
        self.proc = _CTX.Process(
            target=_server_main,
            args=(self.shared, self.req_q, self.events, state_np, cfg, device,
                  max_batch, wait_ms, ready), daemon=True)
        self.proc.start()
        if not ready.wait(timeout=120):
            raise RuntimeError("inference server failed to start")
        HANDLES.update(shared=self.shared, req_q=self.req_q, events=self.events,
                       pid=self.proc.pid, rows=self.rows_per_worker)

    def stop(self):
        self.req_q[1].set()
        self.req_q[0].release()
        self.proc.join(timeout=30)
        HANDLES.clear()


class RemotePolicy:
    """Drop-in for net.act() inside a forked worker (one slot per worker)."""

    def __init__(self, rank: int, encoder_variant: str = "v1", model_id: int = 0):
        self.rank = rank
        self.encoder_variant = encoder_variant
        self.model_id = model_id
        self.shared = HANDLES["shared"]
        self.req_q = HANDLES["req_q"]
        self.cond = HANDLES["events"]

    def eval(self):
        return self

    @torch.no_grad()
    def act_batch(self, planes, scalars, mask, temps, model_ids):
        """Vectorized rollout (perf 2026-08-23): B rows in this worker's slot
        block [rank*R, rank*R+B), one semaphore release per row, one wait
        for the whole block. Returns (idx[B], lp[B])."""
        B = planes.shape[0]
        R = HANDLES.get("rows", 1)
        if B > R:
            raise ValueError(f"worker posted {B} rows but owns {R}")
        lo = self.rank * R
        sl = slice(lo, lo + B)
        sh = self.shared
        sh.planes[sl, :planes.shape[1]].copy_(planes)
        sh.scalars[sl, :scalars.shape[1]].copy_(scalars)
        # narrower action spaces write only their own columns; the server
        # reads back the same slice, so the tail is never observed
        sh.mask[sl, :mask.shape[-1]].copy_(mask)
        sh.temp[sl] = temps
        sh.model_id[sl] = model_ids
        want = sh.done_gen[sl].clone() + 1
        sh.pending[sl] = 1
        for _ in range(B):
            self.req_q[0].release()
        with self.cond:
            while bool((sh.done_gen[sl] < want).any()):
                if not self.cond.wait(timeout=5.0):
                    pid = HANDLES.get("pid")
                    if pid is not None:
                        try:
                            os.kill(pid, 0)
                        except OSError:
                            raise RuntimeError("inference server died")
        out = sh.out_idx[sl].clone()
        if bool((out < 0).any()):
            raise RuntimeError("inference server reported failure")
        return out, sh.out_lp[sl].clone()

    @torch.no_grad()
    def act(self, planes, scalars, mask, temperature: float = 1.0):
        r = self.rank * HANDLES.get("rows", 1)
        self.shared.planes[r, :planes.shape[1]].copy_(planes[0])
        self.shared.scalars[r, :scalars.shape[1]].copy_(scalars[0])
        self.shared.mask[r, :mask.shape[-1]].copy_(mask[0])
        self.shared.temp[r] = float(temperature)
        self.shared.model_id[r] = self.model_id
        # generation read from the SLOT at request time: several shims
        # (learner + opponents) share one worker slot, so a per-object
        # counter would see a stale 'done' and return another model's action
        want = int(self.shared.done_gen[r]) + 1
        self.shared.pending[r] = 1
        self.req_q[0].release()
        with self.cond:
            while int(self.shared.done_gen[r]) < want:
                if not self.cond.wait(timeout=5.0):
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
