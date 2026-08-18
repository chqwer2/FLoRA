# Reviewer pass #1 — ICLR 10/10 bar. Scores are the target, harsh-honest current state below.

## Scores now
| axis | score | why |
|---|---|---|
| Novelty | 6/10 | TTT-as-PEFT + consistency framing is fresh, but recurrence = delta-rule/DeltaNet family. Novelty MUST live in the PEFT framing + the OOD-robustness result. Without the result, an ICLR reviewer sees "linear-attention adapter". |
| Math | 7/10 | Recurrence (Eq.2), consistency Prop, conditional-operator Prop are clean. Missing: the OOD excess-risk argument (the thesis is currently a hypothesis), Prop.2 proof, Lemma formalization. |
| Writing | 7/10 | Kaiming voice, tight. Related Work still \todocite. |
| Presentation | 6/10 | Structure good; needs the money figure real + a method schematic. |
| Results | 3/10 | ALL core results RED. This is the gating issue: ICLR will not accept without in-dist parity + a real OOD-robustness curve. |
| Experiments | 5/10 | Design is right (shift suite + causal isolation + fix ablation). Must be executed. |

## Blocking (in order)
1. **Results are the paper.** ICLR bar = the OOD-robustness result must exist and be clear. Everything else is scaffolding.
   Priority experiment order: build consistent-TTT (Eq.2) -> Table 1 (in-dist parity) -> Table 4 (fix vs naive) ->
   Table 2 + Fig 1 (OOD robustness) -> Table 3 (causal isolation).
2. **The OOD thesis needs theory.** Turn "why robustness grows with shift" into a proposition (excess risk of static vs
   adaptive under a shift model) or clearly label it a tested hypothesis. Currently a \todo.
3. **Prove Prop.2 (1 line) and formalize Lemma 1 (3 lines).** Pure writing.
4. **Sharpen novelty vs DeltaNet/TTT-Linear** in Related Work: the delta is (a) additive branch on a FROZEN LLM,
   (b) train/inference consistency as an explicit property, (c) OOD-robustness as the goal — not a new sequence layer.
5. **Method schematic figure** (frozen attn + branch + recurrence) — ICLR expects it.

## THIS/next loop targets (writing-only until experiments run)
- Prove Prop.2; formalize Lemma 1; write the OOD excess-risk proposition (or hypothesis box).
- Related Work: 1 tight paragraph sharpening the delta vs delta-rule/TTT-Linear/Mamba.
- Keep ALL results RED. The single most important non-writing action is BUILDING consistent-TTT and running Table 1 + 4.

## Iteration 2 done (novelty-first, writing-only)
- [x] Prop.OOD (adaptation gap grows with shift): rigorous, with proof sketch (SGD contraction vs bias-variance). This is
  the theoretical core that separates the paper from "a linear-attention adapter". Math 7 -> 8.5, Novelty 6 -> 7.5.
- [x] Prop.2 (input-conditional operator) PROVED + O(N)/chunkwise cost stated.
- [x] Lemma 1 (train/generation mismatch) FORMALIZED as a state-distribution-shift argument; fix tied to Prop.1.
Remaining writing: Related Work delta-sharpening; method schematic figure; abstract can now cite Prop.OOD as the mechanism.
Remaining (gating): RESULTS all RED — build consistent-TTT, run Tables 1-4 + Fig 1.

## Iteration 3 done (novelty writing)
- [x] Related Work sharpened: explicit 3-point delta vs delta-rule/DeltaNet/TTT-Linear/Mamba (backbone-layer vs additive
  frozen-LLM PEFT; consistency as a proved property; OOD-robustness goal + excess-risk separation). Novelty defense set.
- [x] Method schematic figure placeholder (fig:arch) added.
Scores: Novelty 7.5->8, Math 8.5, Writing 8, Presentation 7. Results 3 (RED), Experiments 5.

## Novelty writing is near-saturated. Remaining is:
- Minor: abstract could name the mechanism (Prop.OOD) in one clause; intro schematic ref. (polish, ~1 loop)
- GATING (not writing): build consistent-TTT (Eq.2) and run Tables 1-4 + Fig 1 (money). The paper cannot pass ICLR
  without the in-dist parity + monotone OOD-robustness result. This needs cluster + author go-ahead.
=> After ~1 polish loop, stop and hand to experiments (do not spin writing).

## Author/experiment note
The consistent-TTT module (Eq.2) is NEW code — the repo's ttt_branch4.py is the teacher-forced version to be replaced.
See EXPERIMENTS_TODO.md.
