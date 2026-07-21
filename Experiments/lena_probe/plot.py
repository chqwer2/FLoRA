"""
Plot the Fig.1 money panels from probe outputs.

  python plot.py --csv results/rank_substitution.csv        # panel A (Pareto) + C (rank)
  python plot.py --gate_map <run_dir>/lena_gate_map.json     # panel B (where-map)
"""
import argparse, csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_rank(csv_path, outdir):
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        r["rank"] = int(r["rank"]); r["params"] = int(r["params"]); r["rel_mse"] = float(r["rel_mse"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for method, color in [("LoRA", "#0f4c81"), ("LeNA", "#e07a1f")]:
        m = sorted([x for x in rows if x["method"] == method], key=lambda x: x["params"])
        # Panel A: params vs rel_mse (Pareto)
        ax[0].plot([x["params"] for x in m], [x["rel_mse"] for x in m],
                   "o-", color=color, label=method, lw=2, ms=6)
        # Panel C: rank vs rel_mse
        mr = sorted([x for x in rows if x["method"] == method], key=lambda x: x["rank"])
        ax[1].plot([x["rank"] for x in mr], [x["rel_mse"] for x in mr],
                   "o-", color=color, label=method, lw=2, ms=6)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("trainable params"); ax[0].set_ylabel("relative MSE (lower=better)")
    ax[0].set_title("A. Param–accuracy Pareto"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
    ax[1].set_xlabel("rank r"); ax[1].set_ylabel("relative MSE")
    ax[1].set_title("C. Nonlinearity substitutes for rank"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    out = os.path.join(outdir, "fig_rank_substitution.png")
    fig.savefig(out, dpi=160); print("saved", out)


def plot_where(json_path, outdir):
    m = json.load(open(json_path))
    keys = list(m.keys()); vals = [m[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6, max(3, len(keys) * 0.25)))
    ax.barh(range(len(keys)), vals, color=["#e07a1f" if v > 0.5 else "#0f4c81" for v in vals])
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=6)
    ax.set_xlabel("gate openness g (orange = nonlinear)")
    frac = sum(v > 0.5 for v in vals) / max(1, len(vals))
    ax.set_title(f"B. Where nonlinearity is used  ({100*frac:.0f}% open)")
    fig.tight_layout()
    out = os.path.join(outdir, "fig_where_map.png")
    fig.savefig(out, dpi=160); print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--gate_map", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.csv:
        plot_rank(args.csv, args.outdir)
    if args.gate_map:
        plot_where(args.gate_map, args.outdir)
    if not args.csv and not args.gate_map:
        print("nothing to plot; pass --csv and/or --gate_map")
