"""
MECHANISM CHECKS for the selection gate ('learn where').

1) Sparsity: train LeNA (hard gate) with an L1 penalty on gate openness at several
   lambdas; report mean openness and the 'fraction open (>0.5)' -> the "only X%
   go nonlinear" claim. PASS if fraction-open moves strictly between 0 and 1 as
   lambda grows (i.e. the gate actually selects, not all-on / all-off collapse).

2) Dead-gate: track whether closed gates can re-open during training (the
   g->0 => grad(phi)->0 collapse risk). PASS if we observe closed->open flips.

3) Norm range: fraction of code z inside the spline's [-3,3] range, with vs
   without norm_before_act. PASS if norm keeps (almost) all mass in range.

Uses a target where nonlinearity helps only a SUBSET of code dims, so a correct
gate should open on that subset and close elsewhere.
"""
import math, argparse
import torch
import torch.nn as nn
from lena_core import LeNAAdapter, LearnableSpline1D, SelectionGate


def make_partial_target(d_in=64, d_out=64, r0=8, k_nonlin=2, seed=0):
    """Nonlinear-low-rank update where only k_nonlin of the r0 code dims are nonlinear."""
    g = torch.Generator().manual_seed(seed)
    W  = torch.randn(d_out, d_in, generator=g) / math.sqrt(d_in)
    As = torch.randn(r0, d_in, generator=g) / math.sqrt(d_in)
    Bs = torch.randn(d_out, r0, generator=g) / math.sqrt(r0)
    mask = torch.zeros(r0); mask[:k_nonlin] = 1.0        # first k dims nonlinear

    def f(x):
        z = x @ As.t()
        znl = mask * torch.relu(z) + (1 - mask) * z       # nonlinear only on subset
        return x @ W.t() + (znl @ Bs.t())
    return W, f


def train(adapter, f, W, d_in, steps=1500, lr=3e-3, l1=0.0, batch=512, track_gate=False, seed=1):
    torch.manual_seed(seed)
    X = torch.randn(4096, d_in); Y = f(X).detach()
    opt = torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad], lr=lr)
    ever_open, init_closed = None, None
    for step in range(steps):
        idx = torch.randint(0, 4096, (batch,))
        pred = adapter(X[idx])
        loss = (pred - Y[idx]).pow(2).mean()
        if l1 > 0 and adapter.gate is not None:
            loss = loss + l1 * adapter.gate.openness().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if track_gate and adapter.gate is not None:
            o = adapter.gate.openness().detach()
            if init_closed is None:
                init_closed = (o < 0.5)
                ever_open = torch.zeros_like(o, dtype=torch.bool)
            ever_open |= (o > 0.5)
    reopened = int((init_closed & ever_open).sum().item()) if init_closed is not None else -1
    return reopened


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_in", type=int, default=64)
    ap.add_argument("--r0", type=int, default=8)
    ap.add_argument("--k_nonlin", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    d_in = args.d_in

    print("=" * 68)
    print("1) GATE SPARSITY vs L1  (target: only", args.k_nonlin, "of", args.r0, "code dims nonlinear)")
    print("=" * 68)
    print(f"{'lambda':>8s} {'mean_open':>10s} {'frac_open>0.5':>14s} {'rel_mse':>10s}")
    W, f = make_partial_target(d_in, 64, args.r0, args.k_nonlin, seed=0)
    for l1 in [0.0, 1e-3, 3e-3, 1e-2, 3e-2]:
        ad = LeNAAdapter(W.clone(), r=args.r0, gate_mode="hard", gate_init=0.0, norm_before_act=True)
        train(ad, f, W, d_in, steps=args.steps, l1=l1)
        with torch.no_grad():
            X = torch.randn(2048, d_in); Y = f(X)
            mse = (ad(X) - Y).pow(2).mean().item()
            base = (Y - X @ W.t()).pow(2).mean().item()
            o = ad.gate.openness()
        print(f"{l1:>8.0e} {o.mean().item():>10.3f} {(o > 0.5).float().mean().item():>14.3f} {mse/base:>10.3e}")
    print(f"(ideal: fraction open -> ~{args.k_nonlin/args.r0:.2f} = k/r0, and rel_mse stays low)\n")

    print("=" * 68)
    print("2) DEAD-GATE CHECK  (can gates that start closed re-open?)")
    print("=" * 68)
    ad = LeNAAdapter(W.clone(), r=args.r0, gate_mode="hard", gate_init=-2.0, norm_before_act=True)
    reopened = train(ad, f, W, d_in, steps=args.steps, l1=0.0, track_gate=True)
    print(f"  gates that were closed at start and later opened: {reopened} / {args.r0}")
    print("  PASS if > 0 (no permanent dead-gate lock-in); else add gate warmup.\n")

    print("=" * 68)
    print("3) NORM-BEFORE-ACT keeps code in spline range [-3, 3]")
    print("=" * 68)
    for use_norm in [False, True]:
        ad = LeNAAdapter(W.clone(), r=args.r0, gate_mode="on", norm_before_act=use_norm)
        X = torch.randn(2048, d_in)
        with torch.no_grad():
            z = X @ ad.A.t()
            zc = ad.norm(z) if ad.norm is not None else z
            in_range = ((zc >= -3) & (zc <= 3)).float().mean().item()
        print(f"  norm_before_act={str(use_norm):5s}: fraction of code in [-3,3] = {in_range:.3f}")
    print("  PASS if norm=True pushes fraction ~1.0 (spline/poly stay well-conditioned).")


if __name__ == "__main__":
    main()
