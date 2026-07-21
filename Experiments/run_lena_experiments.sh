#!/usr/bin/env bash
# Our own LeNA experiments (no competitor baselines yet — those come later).
# Fill in MODEL/DATASET, set GPU, then: bash run_lena_experiments.sh
set -euo pipefail

GPU=${GPU:-0}
MODEL=${MODEL:-meta-llama/Llama-2-7b-hf}
DATASET=${DATASET:-"google/boolq piqa allenai/social_i_qa Rowan/hellaswag allenai/winogrande:winogrande_xl allenai/ai2_arc:ARC-Easy allenai/ai2_arc:ARC-Challenge allenai/openbookqa"}
OUT=${OUT:-runs/lena}
TARGETS=${TARGETS:-q_proj,k_proj,v_proj,up_proj,down_proj}
EPOCHS=${EPOCHS:-3}
LR=${LR:-3e-4}

run () { CUDA_VISIBLE_DEVICES=$GPU python Llama_Adaptation.py \
  --base_model "$MODEL" --dataset "$DATASET" \
  --num_epochs "$EPOCHS" --learning_rate "$LR" --cutoff_len 512 \
  --batch_size 1 --eval_step 20 --save_step 200 --device auto \
  --lora_target_modules "$TARGETS" "$@"; }

echo "=================================================================="
echo " E1. ISO-PARAM RANK SWEEP  (panel A/C: nonlinearity substitutes for rank)"
echo "     LoRA vs LeNA(always-on spline+norm), matched r & alpha per rank"
echo "=================================================================="
for r in 4 8 16 32; do
  a=$((2*r))
  run --methods lora --lora_r $r --lora_alpha $a --lora_dropout 0.05 \
      --output_dir "$OUT/lora_r$r"
  run --methods lena --lena_activations spline --lena_flex_mode dim \
      --lena_norm_before_act --lena_gate_type none \
      --lora_r $r --lora_alpha $a --lora_dropout 0.05 \
      --output_dir "$OUT/lena_r$r"
done

R=16; A=32   # fixed rank for the ablations below
echo "=================================================================="
echo " E2. DoRA DECOUPLE  (show the gain is from nonlinearity, not DoRA)"
echo "=================================================================="
run --methods dora --lora_r $R --lora_alpha $A --output_dir "$OUT/dora_r$R"
run --methods lena --lena_activations spline --lena_flex_mode dim --lena_norm_before_act \
    --lena_gate_type none --lena_use_dora \
    --lora_r $R --lora_alpha $A --output_dir "$OUT/lena_dora_r$R"

echo "=================================================================="
echo " E3. WHERE-MAP + SPARSITY  (panel B: 'only X% go nonlinear')"
echo "     hard gate, INIT OPEN (+2), sweep L1 -> prune. Each writes lena_gate_map.json"
echo "=================================================================="
for l1 in 0 3e-4 1e-3 3e-3; do
  run --methods lena --lena_activations spline --lena_flex_mode dim --lena_norm_before_act \
      --lena_gate_type sigmoid --gate_strength hard --lena_gate_mode channel \
      --lena_gate_init 2.0 --lena_gate_l1 $l1 \
      --lora_r $R --lora_alpha $A --output_dir "$OUT/lena_where_l1_$l1"
done

echo "=================================================================="
echo " E4. NORM-BEFORE-ACT ablation"
echo "=================================================================="
run --methods lena --lena_activations spline --lena_flex_mode dim --lena_gate_type none \
    --lora_r $R --lora_alpha $A --output_dir "$OUT/lena_nonorm_r$R"   # norm OFF (flag absent)

echo "=================================================================="
echo " E5. ACTIVATION ablation  (spline vs fourier/poly/swish)"
echo "=================================================================="
for act in fourier polynomial swish; do
  run --methods lena --lena_activations $act --lena_flex_mode dim --lena_norm_before_act \
      --lena_gate_type none --lora_r $R --lora_alpha $A --output_dir "$OUT/lena_${act}_r$R"
done

echo "ALL DONE. Metrics in each run dir: test_metrics_by_dataset.json"
echo "Where-maps: $OUT/lena_where_l1_*/lena_gate_map.json"
echo "Plot where-map:  python lena_probe/plot.py --gate_map $OUT/lena_where_l1_1e-3/lena_gate_map.json"
