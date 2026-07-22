"""LeNA adapters must survive save_pretrained -> from_pretrained.

PEFT selects what to save with a prefix match, and LeNA's registered prefix
("lena_") matches none of its actual parameter names, so save_pretrained used to
write a 40-byte adapter file containing zero tensors: every trained adapter was
silently thrown away. This builds a tiny Llama, trains nothing, saves, reloads
into a fresh model and checks the outputs match.

Run:  python Experiments/tests/test_lena_save_load.py
"""

import os
import sys
import tempfile

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from peft import LeNAConfig, get_peft_model, PeftModel  # noqa: E402


def build_base():
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg)


def main():
    lena_cfg = LeNAConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
        lena_activation="spline", lena_flex_mode="dim", lena_norm_before_act=True,
        lena_gate_type="sigmoid", gate_strength="hard", lena_gate_init=2.0,
    )
    model = get_peft_model(build_base(), lena_cfg)

    # perturb the adapter so a dropped/zero-init save cannot pass by accident
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.05)

    model.eval()
    ids = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        expected = model(ids).logits

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)

        weights = os.path.join(tmp, "adapter_model.safetensors")
        size = os.path.getsize(weights)
        from safetensors import safe_open
        with safe_open(weights, framework="pt") as f:
            keys = list(f.keys())
            n_params = sum(f.get_slice(k).get_shape()[0] if len(f.get_slice(k).get_shape()) == 1
                           else f.get_slice(k).get_shape()[0] * f.get_slice(k).get_shape()[1]
                           for k in keys)
        print(f"saved {len(keys)} tensors, {n_params:,} params, {size:,} bytes")
        assert keys, "adapter file contains no tensors"
        assert any("lora_A" in k for k in keys), "A projection missing from the adapter file"
        assert any("lora_B" in k for k in keys), "B projection missing from the adapter file"
        assert any("gate" in k for k in keys), "selection gate missing from the adapter file"
        assert any("act" in k for k in keys), "activation params missing from the adapter file"

        reloaded = PeftModel.from_pretrained(build_base(), tmp)
        reloaded.eval()
        with torch.no_grad():
            got = reloaded(ids).logits

    max_diff = (expected - got).abs().max().item()
    print(f"max |logit diff| after reload = {max_diff:.2e}")
    assert max_diff < 1e-4, f"reloaded adapter does not reproduce the saved model ({max_diff:.2e})"
    print("SAVE_LOAD_TEST_PASS")


if __name__ == "__main__":
    main()
