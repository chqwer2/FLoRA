"""
Full Flora activation sweep test.

Tests ALL combinations of:
  ActKind  = ["identity", "relu", "gelu", "fourier", "spline", "polynomial"]
  FlexMode = ["global", "spatial", "channel", "voxel"]

Also sweeps gate settings:
  gate_type in ["none", "sigmoid", "tanh", "rezero"]   (edit to match what you implemented)
  gate_position in ["after_a", "after_b", "both"]

For spatial/voxel modes, it passes max_h/max_w via flora_activation_kwargs so it works
with variable seq_len (H changes).

Outputs a summary list (printed) and returns the list.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import FloraConfig, get_peft_model


# -----------------------
# Device helpers
# -----------------------
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_sync(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


# -----------------------
# Timing
# -----------------------
def time_forward(
    model: nn.Module,
    tokenizer,
    text: str = "Hello world!",
    seq_len: int = 128,
    iters: int = 20,
    warmup: int = 8,
) -> Dict[str, float]:
    model.eval()
    dev = next(model.parameters()).device

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if input_ids.shape[1] < seq_len:
        pad = torch.full((input_ids.shape[0], seq_len - input_ids.shape[1]), pad_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, pad], dim=1)
    else:
        input_ids = input_ids[:, :seq_len]

    input_ids = input_ids.to(dev)
    attention_mask = (input_ids != pad_id).to(dev)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        device_sync(dev)

    times: List[float] = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            device_sync(dev)
            times.append(time.perf_counter() - t0)

    times_sorted = sorted(times)
    mean = sum(times) / len(times)
    p50 = times_sorted[len(times_sorted) // 2]
    p95 = times_sorted[int(len(times_sorted) * 0.95) - 1]
    return {"mean_s": mean, "p50_s": p50, "p95_s": p95, "tok_s": seq_len / mean}


# -----------------------
# Param counting (A/B/act/gates)
# -----------------------
def count_flora_params_breakdown(model: nn.Module) -> Dict[str, int]:
    totals = defaultdict(int)

    def add_params(tag: str, submod: nn.Module):
        for p in submod.parameters(recurse=True):
            totals[tag] += p.numel()

    for _, mod in model.named_modules():
        cls = mod.__class__.__name__

        # Flora wrappers
        if cls in ("FloraLinear", "FloraConv1D"):
            A_dict = getattr(mod, "A", None) or getattr(mod, "flora_A", None)
            B_dict = getattr(mod, "B", None) or getattr(mod, "flora_B", None)
            act_dict = getattr(mod, "act", None) or getattr(mod, "flora_act", None)
            gateA_dict = getattr(mod, "gate_after_a", None)
            gateB_dict = getattr(mod, "gate_after_b", None)

            if A_dict is not None:
                for _, A in A_dict.items():
                    add_params("A", A)
            if B_dict is not None:
                for _, B in B_dict.items():
                    add_params("B", B)
            if act_dict is not None:
                for _, act in act_dict.items():
                    add_params("activation", act)
            if gateA_dict is not None:
                for _, g in gateA_dict.items():
                    add_params("gate_after_a", g)
            if gateB_dict is not None:
                for _, g in gateB_dict.items():
                    add_params("gate_after_b", g)

        # LoRA fallback modules
        elif hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            lora_A = getattr(mod, "lora_A")
            lora_B = getattr(mod, "lora_B")

            if isinstance(lora_A, nn.ModuleDict):
                for _, A in lora_A.items():
                    add_params("A", A)
            if isinstance(lora_B, nn.ModuleDict):
                for _, B in lora_B.items():
                    add_params("B", B)

            if hasattr(mod, "lora_embedding_A"):
                embA = getattr(mod, "lora_embedding_A")
                if isinstance(embA, nn.ParameterDict):
                    for _, p in embA.items():
                        totals["A"] += p.numel()

            if hasattr(mod, "lora_embedding_B"):
                embB = getattr(mod, "lora_embedding_B")
                if isinstance(embB, nn.ParameterDict):
                    for _, p in embB.items():
                        totals["B"] += p.numel()

    totals["gate_total"] = totals["gate_after_a"] + totals["gate_after_b"]
    totals["total"] = totals["A"] + totals["B"] + totals["activation"] + totals["gate_total"]
    return dict(totals)


# -----------------------
# Wrapper checks
# -----------------------
def find_module_type(model: nn.Module, suffix: str) -> Optional[str]:
    for n, m in model.named_modules():
        if n.endswith(suffix):
            return type(m).__name__
    return None


def any_flora_modules(model: nn.Module) -> bool:
    return any(type(m).__name__.startswith("Flora") for m in model.modules())


# -----------------------
# Build fixed-length dummy to materialize lazy params
# -----------------------
def materialize(model: nn.Module, tokenizer, seq_len: int = 128):
    dev = next(model.parameters()).device
    with torch.no_grad():
        dummy = tokenizer("hello", return_tensors="pt")
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids = dummy["input_ids"]
        if ids.shape[1] < seq_len:
            pad = torch.full((ids.shape[0], seq_len - ids.shape[1]), pad_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=1)
        else:
            ids = ids[:, :seq_len]
        ids = ids.to(dev)
        attn = (ids != pad_id).to(dev)
        _ = model(input_ids=ids, attention_mask=attn)

def make_fixed_batch(tokenizer, device, text="hello", seq_len=128):
    enc = tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    if ids.shape[1] < seq_len:
        pad = torch.full((ids.shape[0], seq_len - ids.shape[1]), pad_id, dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=1)
    else:
        ids = ids[:, :seq_len]

    ids = ids.to(device)
    attn = (ids != pad_id).to(device)
    return {"input_ids": ids, "attention_mask": attn}


def materialize_once(model, batch):
    model.eval()
    with torch.no_grad():
        _ = model(**batch)


# -----------------------
# Main sweep
# -----------------------
def run_full_sweep():
    model_id = "Intel/tiny-random-llama2"
    device = pick_device()
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # MPS best practice: avoid device_map="auto"


    act_kinds = ["fourier", "spline", "polynomial"]  # "identity", "relu", "gelu",
    flex_modes = ["global", "spatial", "channel", "voxel"]

    # Edit these to exactly match what your Gate implementation supports
    gate_types = ["none"] #, "sigmoid", "tanh", "rezero"]
    gate_positions = ["after_a"] #, "after_b", "both"]

    # Keep measurement stable
    seq_len = 128
    iters = 20
    warmup = 8

    # For spatial/voxel we need max_h/max_w (table + slice).
    # For transformers with H=seq_len and W=1, max_w=1 is fine.
    max_h = seq_len
    max_w = 1

    results: List[Dict] = []

    for act in act_kinds:
        for mode in flex_modes:
            # activation kwargs needed for spatial/voxel in the fixed activations file
            act_kwargs = {}
            if mode in ("spatial", "voxel"):
                act_kwargs = {"max_h": max_h, "max_w": max_w}

            for gate_type in gate_types:
                # if gate is off, position doesn't matter; pick one
                positions = gate_positions if gate_type != "none" else ["after_b"]

                for gate_pos in positions:
                    cfg = FloraConfig(
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "v_proj"],

                        flora_activation=act,
                        flora_flex_mode=mode,
                        flora_activation_kwargs=act_kwargs,

                        flora_gate_type=gate_type,
                        flora_gate_position=gate_pos,

                        # keep sweep quiet; enable for a single case when debugging
                        flora_debug=False,
                        flora_debug_verbose=False,
                        flora_debug_forward=False,
                        flora_debug_check_nan=True,
                    )

                    base_model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if device.type in ("cuda", "mps") else torch.float32,
                    ).to(device)
                    base_model.config.use_cache = False

                    model = get_peft_model(base_model, cfg).to(device)

                    # materialize lazy parameters with consistent seq_len
                    materialize(model, tokenizer, seq_len=seq_len)

                    timings = time_forward(model, tokenizer, seq_len=seq_len, iters=iters, warmup=warmup)
                    params = count_flora_params_breakdown(model)

                    res = {
                        "activation": act,
                        "flex_mode": mode,
                        "gate_type": gate_type,
                        "gate_position": gate_pos,

                        "has_flora": any_flora_modules(model),
                        "q_proj_type": find_module_type(model, "q_proj"),
                        "v_proj_type": find_module_type(model, "v_proj"),

                        "params_total": params.get("total", 0),
                        "params_A": params.get("A", 0),
                        "params_B": params.get("B", 0),
                        "params_activation": params.get("activation", 0),
                        "params_gate_a": params.get("gate_after_a", 0),
                        "params_gate_b": params.get("gate_after_b", 0),

                        "time_mean_ms": timings["mean_s"] * 1000.0,
                        "time_p50_ms": timings["p50_s"] * 1000.0,
                        "time_p95_ms": timings["p95_s"] * 1000.0,
                        "tok_s": timings["tok_s"],
                    }
                    results.append(res)
                    print("OK:", res)

                    # except Exception as e:
                    #     results.append({
                    #         "activation": act,
                    #         "flex_mode": mode,
                    #         "gate_type": gate_type,
                    #         "gate_position": gate_pos,
                    #         "error": repr(e),
                    #     })
                    #     print("FAIL:", act, mode, gate_type, gate_pos, "->", repr(e))

    # Summary list (sorted by mean latency if present; errors at bottom)
    def sort_key(r: Dict):
        return (0, r["time_mean_ms"]) if "time_mean_ms" in r else (1, 1e18)

    results_sorted = sorted(results, key=sort_key)

    print("\n================ SUMMARY LIST (sorted by time_mean_ms) ================\n")
    for r in results_sorted:
        if "error" in r:
            print(f"- act={r['activation']:<10} mode={r['flex_mode']:<7} gate={r['gate_type']:<7} pos={r['gate_position']:<7} -> ERROR {r['error']}")
        else:
            print(
                f"- act={r['activation']:<10} mode={r['flex_mode']:<7} gate={r['gate_type']:<7} pos={r['gate_position']:<7} "
                f"flora={str(r['has_flora']):<5} q={r['q_proj_type']:<12} v={r['v_proj_type']:<12} "
                f"params(total/A/B/act/gA/gB)={r['params_total']}/{r['params_A']}/{r['params_B']}/"
                f"{r['params_activation']}/{r['params_gate_a']}/{r['params_gate_b']} "
                f"time(ms) mean/p50/p95={r['time_mean_ms']:.2f}/{r['time_p50_ms']:.2f}/{r['time_p95_ms']:.2f} "
                f"tok/s={r['tok_s']:.1f}"
            )

    return results_sorted


if __name__ == "__main__":
    run_full_sweep()
