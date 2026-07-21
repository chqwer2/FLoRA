# LeNA probe harness

Self-contained, **GPU-free** experiments that decide whether the LeNA story is real
*before* spending compute on LLMs. The core (`lena_core.py`) mirrors the math of
`Experiments/peft/tuners/lena/{activations,gates,layer}.py`.

## What each script answers

| script | reviewer concern it addresses | Fig.1 panel |
|---|---|---|
| `run_rank_substitution.py` | "different param counts", "when does nonlinearity help", "is the class strictly expanded" (R1, TRbS, FpK3, DrJS) | A (Pareto), C (rank) |
| `run_mechanism_checks.py` | "analyze the learned structure / where is nonlinearity used", stability, init-scale (TRbS, FpK3, DrJS, R1) | B (where-map) |
| `plot.py` | renders panels A/B/C | — |

## Run

```bash
cd Experiments/lena_probe
python run_rank_substitution.py --steps 1500 --seeds 0 1 2
python run_mechanism_checks.py  --steps 1500
python plot.py --csv results/rank_substitution.csv
```

## Findings so far (relu target, r0=4, d=64)

**1. Nonlinearity substitutes for rank — STRONG PASS.**
Linear LoRA plateaus at rel_MSE ≈ 0.52 at *every* rank (a linear rank-r update cannot
represent a nonlinear-low-rank target). LeNA keeps improving: 0.24 (r=4) → 0.078 (r=8)
→ 0.046 (r=16) → 0.023 (r=32). For LeNA rank ≥ 4, LoRA **never** matches within the
swept ranks. This is the panel-A/C money result.

> Caveat (be honest in the paper): the synthetic target is *by construction*
> nonlinear-low-rank, so this proves the mechanism **exists**, not that real
> fine-tuning updates have this structure. The LLM experiments (below) test whether
> it **matters** in practice.

**2. Selection gate — needs the right recipe (validated).**
- Naive `init-closed + L1` → **collapse**: all gates close (0% open), error reverts to
  linear, and 0/8 closed gates ever reopen (dead-gate lock-in). DO NOT use this.
- **`hard gate + init-OPEN + moderate L1` (start-dense-then-prune) → correct**: on a
  target where only 2/8 code dims are nonlinear, the gate converges to exactly
  frac_open = 0.25, keeping the 2 nonlinear dims open and pruning the rest, with low
  error. This is the recipe to use for the "only X% go nonlinear" claim.
- End-state property preserved: closed hard gates = **exact LoRA** (LeNA ⊇ LoRA).

**3. `norm_before_act` — PASS.** Pushes ~100% of the code into the spline's [-3,3]
range (vs 97.7% without), so spline/polynomial stay well-conditioned and
init-scale-insensitive (answers R1's stability/hyperparameter concern).

## Operational recipe (carry into the LLM runs)

- activation: **spline** (subsumes relu/poly/swish shapes; one learnable function)
- `--lena_norm_before_act`
- selection gate: **hard**, **init open**, L1 in ~[3e-4, 3e-3] (`--lena_gate_l1`)
- report **plain LeNA** (no DoRA) as the headline; LeNA-D only as a heuristic variant

## LLM experiments (need GPU; driver already wired)

Iso-parameter rank sweep + where-map export:
```bash
# per rank r in 4 8 16 32, matched across methods:
python ../Llama_Adaptation.py --base_model $model --dataset "$dataset" \
  --methods lora --lora_r $r --lora_alpha $((2*r)) --output_dir runs/lora_r$r
python ../Llama_Adaptation.py --base_model $model --dataset "$dataset" \
  --methods lena --lena_activations spline --lena_norm_before_act \
  --lena_gate_type sigmoid --gate_strength hard --lena_gate_l1 1e-3 \
  --lora_r $r --lora_alpha $((2*r)) --output_dir runs/lena_r$r
```
Each LeNA run writes `runs/.../lena_gate_map.json` (per-module gate openness) and
prints the "fraction open" number. Plot the where-map:
```bash
python plot.py --gate_map runs/lena_r8/lena_gate_map.json
```
Baselines still to add for Table 1: **AuroRA, AFA-LoRA, FourierFT** (reviewers named
all three), plus **≥3 seeds ± std**.
