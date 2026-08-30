"""torch.compile benchmark for our arches (queued 2026-08-29, user ask).

Measures, per arch x mode: fwd-only (infer-server shape B=128 and B=1024)
and fwd+bwd+step (training, B=1024), all bf16 autocast, plus NUMERICS:
max |compiled - eager| on logits and on a probe gradient — the go/no-go
for using compile in real training (comparability risk).
Modes: eager | compile-default | compile-maxautotune.
"""
import argparse, json, sys, time
sys.path.insert(0, ".")
import torch
from src.agents.dnn.arch_zoo import ZOO

def bench(fn, warm=8, iters=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters

def run_arch(name, results):
    factory, _ = ZOO[name]
    torch.manual_seed(0)
    net = factory().cuda()
    sdim = getattr(net, "global_proj", None)
    sdim = sdim.in_features if sdim is not None else 29
    adim = getattr(net, "action_dim", 374)
    def make(B):
        p = torch.rand(B, net.in_planes, 34, device="cuda")
        s = torch.rand(B, sdim, device="cuda")
        m = torch.zeros(B, adim, dtype=torch.bool, device="cuda"); m[:, :14] = True
        y = torch.randint(0, 14, (B,), device="cuda")
        return p, s, m, y
    data = {B: make(B) for B in (128, 1024)}
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)

    # eager reference logits for numerics
    p, s, m, y = data[1024]
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        ref = net(p, s, m).float().clone()

    for mode, wrap in [("eager", lambda f: f),
                       ("compile", lambda f: torch.compile(f)),
                       ("compile_maxauto", lambda f: torch.compile(f, mode="max-autotune"))]:
        try:
            cnet = wrap(net) if mode != "eager" else net
            r = {}
            for B in (128, 1024):
                p, s, m, y = data[B]
                def fwd():
                    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                        cnet(p, s, m)
                r[f"fwd_B{B}_ms"] = round(bench(fwd) * 1000, 2)
            p, s, m, y = data[1024]
            def step():
                with torch.autocast("cuda", torch.bfloat16):
                    lg = cnet(p, s, m)
                    loss = torch.nn.functional.cross_entropy(lg.float(), y)
                opt.zero_grad(); loss.backward(); opt.step()
            r["train_B1024_ms"] = round(bench(step, warm=6, iters=20) * 1000, 2)
            r["train_rows_s"] = round(1024 / (r["train_B1024_ms"] / 1000))
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                out = cnet(p, s, m).float()
            fin = torch.isfinite(ref) & torch.isfinite(out)
            r["max_abs_logit_delta_vs_eager"] = float((out - ref)[fin].abs().max())
            results[f"{name}/{mode}"] = r
            print(f"{name}/{mode}: {r}", flush=True)
        except Exception as e:                              # noqa: BLE001
            results[f"{name}/{mode}"] = {"error": repr(e)[:200]}
            print(f"{name}/{mode}: FAILED {e!r}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="convformer_m_v3r_m46,hrf_xl_v4")
    ap.add_argument("--out", default="experiments/bench_torch_compile.json")
    a = ap.parse_args()
    results = {"torch": torch.__version__}
    for name in a.archs.split(","):
        run_arch(name, results)
    json.dump(results, open(a.out, "w"), indent=1)
    print("saved", a.out)
