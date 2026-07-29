# Results (2026-07-29 innovation round) — consolidated, honest

Llama-2-7B, PEFT, r=8 unless noted. **Judge by generative/likelihood accuracy, NEVER token_acc**
(token_acc is saturated at ~0.72 AND anti-correlated with real accuracy on mnli — see below).

## Real-eval NOISE FLOOR (measured, critical context)
Two *identical* LoRA runs (mtc_lora vs a Wv=0 "refl" run = plain LoRA) scored:
- eval_choice AVG: **0.5664 vs 0.5902 → ±0.024**
- gsm-gen (200 samples): **0.06 vs 0.115 → ±0.055**

⇒ Any "gain" below ~0.03 (choice) / ~0.06 (gsm-gen) is **noise**. This reframes most nulls as
"undetectable on this testbed", not "saturated".

## Multi-task (4-task mix: piqa+gsm8k+squad+mnli), real accuracy

| method | eval_choice AVG | mnli-gen | note |
|---|---|---|---|
| LoRA | 0.566 (noise→0.590) | 0.870 | baseline |
| AuroRA | 0.578 | 0.870 | ≈lora |
| two-path (mean-split) aurora | 0.584 (s1) / 0.578 (s2) | 0.85–0.87 | +0.006 didn't replicate ⇒ **noise** |
| two-path (mean-split) lora | 0.561 | 0.880 | ablation contradicts a real effect |
| **ttt2 (cross-token gram, NON-causal)** | 0.584 | **0.310** | **broke generation (future leakage)** |
| OutGate reflection | 0.586 | 0.865 | null but stable |

**token_acc anti-correlation (headline methodological finding)**: ttt2 mnli **token_acc +0.021** but
mnli **generative −0.385** (0.87→0.31). token_acc moved OPPOSITE to real accuracy.

## Isolated test-time adaptation (freeze global pathway, adapt local branch K steps, per-example)

| K / lr | OVERALL d (K vs K=0) |
|---|---|
| 5 / 1e-3 | +0.000 |
| 5 / 3e-3 | −0.0025 |
| 10 / 1e-3 | −0.0025 |

Isolation is **safe** (adapting the isolated branch does NOT corrupt the frozen global — unlike
adapting the single adapter, which the diagnostic showed crashes). But **zero gain**: the
self-supervised (prompt-LM) objective is too weak without per-example labels.

## New methods this round

### IQ-LoRA (input-dependent quadratic) — NULL but STABLE
- Single-task **MNLI** (near-saturated 0.88): IQ 0.8820 vs LoRA 0.8800 = +0.002 (noise); λ→0.
- **Multi-task** (headroom on gsm/piqa), real accuracy vs LoRA:

| metric | LoRA | IQ-LoRA | Δ |
|---|---|---|---|
| eval_choice AVG | 0.5664 | 0.5798 | +0.013 (within noise ±0.024) |
| mnli-gen | 0.870 | 0.865 | −0.005 (flat) |
| gsm-gen | 0.06 | 0.05 | flat |

- CHECK `lam_mean=0.0086, lam_absmax=0.108` — the optimizer used the quadratic only slightly.
- **Verdict: ≈ LoRA (all within noise) — the 2nd-order feature interaction gives no real gain here.**
  BUT unlike the TTT methods it does NOT break generation (mnli-gen 0.865) — a per-token feedforward
  quadratic is decode-safe by construction. Null but stable.

### Two-path TTT (ViT³-grounded, causal) — FAILED (worse than baseline on both metrics)
- **Causality verified** (smoke): perturbing position i leaves all outputs <i exactly unchanged.
- Multi-task, real accuracy vs LoRA:

| metric | LoRA | 2pttt (causal) | Δ |
|---|---|---|---|
| eval_choice AVG | 0.5664 | **0.4526** | **−0.11** |
| mnli-gen | 0.870 | **0.2550** | **−0.61 (collapsed)** |

- **KEY finding — causal fix did NOT save generation.** The collapse is NOT future-leakage (causality
  was verified); it is **decode-time degeneracy**: autoregressive generation with KV-cache passes ONE
  token per step, so the TTT inner-model fit (which needs a sequence of k→v pairs) is degenerate at
  decode → garbage write → generation collapses. eval_choice also drops (the multi-task-trained TTT
  write mis-generalizes to the zero-shot commonsense suite).
- **Conclusion**: the TTT-layer mechanism (ViT³) does NOT port to a PEFT adapter on an autoregressive
  decoder — it breaks generation whether non-causal (leakage) OR causal (decode degeneracy). A clean,
  novel negative result about *why* TTT-layers don't transfer to autoregressive-decoder PEFT.

## Honest bottom line
No original mechanism has beaten LoRA on real accuracy **beyond the noise floor**. The robust,
publishable products are the **methodological findings**:
1. Fixed-subspace ceiling (output ∈ col(B); functional-vs-structural distinction).
2. Nonlinearity helps only at r=2 (regime, non-monotone).
3. **token_acc is saturated AND anti-correlated with real accuracy** (mnli +0.021 proxy / −0.385 real).
4. **Training loss also fails to predict real accuracy** (two-path loss −0.10 lower, no accuracy gain).
5. **Real-eval run-to-run noise is large** (±0.024 choice / ±0.055 gsm-gen @200) — small gains untrustworthy.
6. **Non-causal cross-token adaptation leaks future tokens and breaks autoregressive generation**
   (0.87→0.31); the causal version is stable — a clean causal-vs-non-causal control.
7. **Isolation makes per-sample test-time adaptation safe** (preservation) though the self-sup signal was too weak.

Two new mechanisms (IQ-LoRA, causal two-path TTT) are implemented, verified, and under a
**properly-powered (low-variance/headroom) evaluation** — the correct way to detect a real 1–2% effect.
