"""
Method validation (CPU, no GPU): compare adaptation MECHANISMS on a nonlinear-low-rank target.

  linear LoRA        Delta = B (A x)
  LeNA-spline        Delta = B phi(A x)             (learnable spline)
  LeNA-bilinear      Delta = B ((A1 x) ⊙ (A2 x))     (second-order / multiplicative)

Reports, at matched-ish params (equal rank), for each mechanism:
  rel_mse       fit error / do-nothing error  (lower = fits the nonlinear update better)
  family_rank   effective rank of the stacked per-input Jacobians
                = "how input-conditional the learned update is"
                LoRA -> ~1 (ONE update for all inputs);  nonlinear -> >1 (implicit mixture-of-LoRAs)

GO/NO-GO for the input-conditional / implicit-MoLoRA thesis:
  PASS if nonlinear mechanisms show family_rank >> 1 AND lower rel_mse than LoRA.
"""
import statistics as st
import torch
from lena_core import LoRAAdapter, LeNAAdapter, BilinearAdapter, make_target, fit, update_family_rank

D = 64; R0 = 4; SIGMA = "relu"; RANKS = [2, 4, 8]; SEEDS = [0, 1]

print(f"target: y = Wx + B*·{SIGMA}(A*x), r0={R0}, d={D}; seeds={SEEDS}\n")
print(f"{'method':16s} {'rank':>4s} {'params':>7s} {'rel_mse':>9s} {'family_rank':>12s}")
for r in RANKS:
    for name in ["LoRA", "LeNA-spline", "LeNA-bilinear"]:
        rels, frs, params = [], [], None
        for s in SEEDS:
            Wt, f = make_target(D, D, R0, SIGMA, seed=s)
            if name == "LoRA":
                ad = LoRAAdapter(Wt.clone(), r)
            elif name == "LeNA-spline":
                ad = LeNAAdapter(Wt.clone(), r, gate_mode="on", norm_before_act=False)
            else:
                ad = BilinearAdapter(Wt.clone(), r)
            mse, base, p = fit(ad, f, Wt, d_in=D, steps=1000, seed=100 + s)
            rels.append(mse / base); params = p
            fr, _, _ = update_family_rank(ad, Wt, D, n=48)
            frs.append(fr)
        print(f"{name:16s} {r:>4d} {params:>7d} {st.mean(rels):>9.3e} {st.mean(frs):>12.1f}")
print("\n(LoRA family_rank should be ~1 = same update for every input;")
print(" nonlinear methods >1 = the update is input-conditional / implicit mixture-of-LoRAs.)")
