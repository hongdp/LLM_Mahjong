"""exp113: GPU micro-benchmark for the 384x12 policy on the current machine -- PPO-like train step (B=4096, bf16
autocast) and bf16 inference at rollout-sized batches, eager cudnn trunk vs fast trunk + torch.compile."""
import sys, time, os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agents.dnn.arch_zoo import ZOO  # noqa: E402
from src.agents.dnn.net import enable_fast_trunk  # noqa: E402


def timeit(fn, it=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / it


def run(mode, arch="cnn_c384b12_v3r"):
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    net = ZOO[arch][0]().cuda()
    if mode == "fast+compile":
        enable_fast_trunk(net, compile=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-5, fused=True)
    B = 4096
    P = torch.randn(B, net.in_planes, 34, device="cuda"); S = torch.randn(B, net.scalar_fc[0].in_features, device="cuda")
    M = torch.ones(B, net.head[-1].out_features, dtype=torch.bool, device="cuda"); tgt = torch.randn(B, device="cuda")

    def step():
        with torch.autocast("cuda", torch.bfloat16):
            lo, v = net.forward_with_value(P, S, M)
        loss = lo.float().logsumexp(1).mean() + F.mse_loss(v.float(), tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.train(); t = timeit(step)
    out = [f"{mode:13s} train B=4096 {B / t / 1e3:5.1f}k samples/s"]
    net.eval()
    for b in (1024, 512, 8192):
        Pb = P.repeat(2, 1, 1)[:b]; Sb = S.repeat(2, 1)[:b]; Mb = M.repeat(2, 1)[:b]

        def inf():
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                return net(Pb, Sb, Mb)
        t = timeit(inf, 20); out.append(f"infer B={b} {b / t / 1e3:5.1f}k/s")
    print(" | ".join(out), flush=True)


if __name__ == "__main__":
    for m in ("eager", "fast+compile"):
        run(m)
