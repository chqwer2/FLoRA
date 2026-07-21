"""
DECISIVE EXPERIMENT: does nonlinearity substitute for rank?

Target = a NONLINEAR-low-rank update  y = Wx + B* sigma(A* x)  (rank r0).
Freeze W, fit LoRA (linear rank r) and LeNA (nonlinear rank r), sweep r.

Produces:
  results/rank_substitution.csv   (method, rank, params, test_mse, rel_mse)
and prints:
  - the param-accuracy table (Fig.1 panel A: Pareto)
  - "nonlinearity is worth ~k ranks" at each budget (Fig.1 panel C)

PASS (story is real) if:
  * LeNA reaches lower rel_mse than LoRA at matched params, AND
  * the LoRA rank needed to match LeNA(r) is >> r at small r, shrinking as r grows.
FAIL (stop / repivot) if the two curves overlap.
"""
import os, csv, argparse
import torch
from lena_core import LoRAAdapter, LeNAAdapter, make_target, fit


def lora_rank_for_target_mse(rows, target_mse):
    """Interpolate: what LoRA rank matches a given mse (for the 'worth k ranks' number)."""
    lo = [(r["rank"], r["rel_mse"]) for r in rows if r["method"] == "LoRA"]
    lo.sort()
    # rel_mse is decreasing in rank; find where it crosses target
    for i in range(len(lo) - 1):
        (r1, m1), (r2, m2) = lo[i], lo[i + 1]
        if (m1 - target_mse) * (m2 - target_mse) <= 0 and m1 != m2:
            f = (m1 - target_mse) / (m1 - m2)
            return r1 + f * (r2 - r1)
    if lo and target_mse < lo[-1][1]:
        return float("inf")   # LoRA never reaches it in the swept range
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_in", type=int, default=64)
    ap.add_argument("--d_out", type=int, default=64)
    ap.add_argument("--r0", type=int, default=4, help="true rank of the nonlinear update")
    ap.add_argument("--sigma", type=str, default="relu", choices=["relu", "tanh", "sin", "gelu"])
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gate_mode", type=str, default="on", choices=["on", "soft", "hard"])
    ap.add_argument("--no_norm", action="store_true")
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    print(f"\nTarget: y = Wx + B*·{args.sigma}(A*·x),  r0={args.r0},  d={args.d_in}->{args.d_out}")
    print(f"Averaging over seeds {args.seeds}; ranks {args.ranks}\n")
    print(f"{'method':6s} {'rank':>4s} {'params':>8s} {'rel_mse':>10s}   (rel_mse = mse / do-nothing mse)")

    for r in args.ranks:
        for method in ["LoRA", "LeNA"]:
            rels, params = [], None
            for W, seed in [(make_target(args.d_in, args.d_out, args.r0, args.sigma, seed=s), s) for s in args.seeds]:
                (Wt, f) = W
                if method == "LoRA":
                    ad = LoRAAdapter(Wt.clone(), r=r)
                else:
                    ad = LeNAAdapter(Wt.clone(), r=r, gate_mode=args.gate_mode,
                                     norm_before_act=not args.no_norm)
                mse, base, p = fit(ad, f, Wt, d_in=args.d_in, steps=args.steps, seed=100 + seed)
                rels.append(mse / base)
                params = p
            rel = sum(rels) / len(rels)
            rows.append({"method": method, "rank": r, "params": params, "rel_mse": rel})
            print(f"{method:6s} {r:>4d} {params:>8d} {rel:>10.4e}")

    # write csv
    csv_path = os.path.join(args.outdir, "rank_substitution.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "rank", "params", "rel_mse"])
        w.writeheader(); w.writerows(rows)

    # "nonlinearity is worth ~k ranks" readout
    print("\n=== nonlinearity is worth ~k ranks (at each LeNA rank) ===")
    for r in args.ranks:
        lena = next((x for x in rows if x["method"] == "LeNA" and x["rank"] == r), None)
        if not lena:
            continue
        r_equiv = lora_rank_for_target_mse(rows, lena["rel_mse"])
        if r_equiv is None:
            note = "(LoRA already better in range)"
        elif r_equiv == float("inf"):
            note = "LoRA never matches within swept ranks  <-- strong signal"
        else:
            note = f"LoRA needs rank ~{r_equiv:.1f}  => +{r_equiv - r:.1f} ranks"
        print(f"  LeNA rank {r:>2d} (rel_mse={lena['rel_mse']:.2e}) == {note}")

    print(f"\nSaved: {csv_path}")
    print("Plot with:  python plot.py --csv results/rank_substitution.csv")


if __name__ == "__main__":
    main()
