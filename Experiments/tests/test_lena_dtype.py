"""fp16-autocast dtype contract for LeNALinear (needs a CUDA GPU).

The trainer loads the base model in fp16, promotes trainable adapter params to
fp32 and runs under autocast, which is exactly the combination that broke every
cluster smoke run. Covers every activation x DoRA x norm_before_act combination.

Run:  python Experiments/tests/test_lena_dtype.py
"""

import itertools
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from peft.tuners.lena.layer import LeNALinear  # noqa: E402
from peft.tuners.lena.config import LeNAConfig  # noqa: E402

ACTIVATIONS = ["spline", "fourier", "polynomial", "swish", "identity"]

fails = []
for act, dora, norm in itertools.product(ACTIVATIONS, [False, True], [True, False]):
    tag = f"act={act:11s} dora={dora!s:5s} norm={norm!s:5s}"
    try:
        base = nn.Linear(64, 64, bias=False).half().cuda()
        cfg = LeNAConfig(r=8, lora_alpha=16, lena_activation=act, lena_flex_mode="dim",
                         lena_norm_before_act=norm, lena_gate_type="sigmoid",
                         gate_strength="hard", lena_gate_init=2.0, lena_use_dora=dora)
        layer = LeNALinear(base)
        layer.add_adapter("default", cfg)
        layer = layer.cuda()
        for p in layer.base_layer.parameters():
            p.requires_grad_(False)
        for p in layer.parameters():          # the trainer promotes trainables to fp32
            if p.requires_grad:
                p.data = p.data.float()

        x = torch.randn(2, 5, 64, device="cuda", dtype=torch.half)
        with torch.autocast("cuda", dtype=torch.float16):
            y = layer(x)
        y.float().pow(2).mean().backward()

        ngrad = sum(1 for p in layer.parameters() if p.requires_grad and p.grad is not None)
        assert torch.isfinite(y).all(), "non-finite output"
        assert ngrad > 0, "no adapter parameter received a gradient"
        print(f"OK   {tag} out_dtype={y.dtype} grads={ngrad}")
    except Exception as e:  # noqa: BLE001 - report every combination, fail at the end
        fails.append((tag, repr(e)))
        print(f"FAIL {tag} {e!r}"[:160])

print("=" * 60)
if fails:
    print(f"{len(fails)} FAILED")
    sys.exit(1)
print("ALL_AUTOCAST_TESTS_PASS")
