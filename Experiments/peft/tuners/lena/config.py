from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal
from peft.utils.peft_types import PeftType

from peft.tuners.lora import LoraConfig

LeNAActivation = Literal["identity", "relu", "gelu", "fourier", "spline", "polynomial"]
LeNAFlexMode = Literal["global", "spatial", "channel", "voxel"]

LeNAGateType = Literal["none", "sigmoid", "rezero"]
LeNAGatePos = Literal["after_a", "after_b", "both"]
LeNAGateMode = Literal["global", "per_dim"]


@dataclass
class LeNAConfig(LoraConfig):
    # peft_type: str = field(default="FLORA", init=False)
    peft_type: PeftType = field(default=PeftType.LENA, init=False)

    # activation
    lena_activation: LeNAActivation = "identity"
    lena_activation_kwargs: Dict[str, Any] = field(default_factory=dict)
    lena_flex_mode: LeNAFlexMode = "global"

    # gating
    lena_gate_type: LeNAGateType = "none"
    lena_gate_position: LeNAGatePos = "after_b"  # deprecated: single selection gate acts on the code z
    lena_gate_mode: LeNAGateMode = "global"
    lena_gate_init: float = -2.0  # pre-sigmoid; negative => gate starts near-closed (starts at LoRA)
    gate_strength: str = "soft"  # Literal["soft", "hard"] = "soft"

    # merge
    allow_merge: bool = False

    # DoRA-style magnitude/direction decomposition on the nonlinear path.
    # Default OFF so LeNA's nonlinearity gain can be measured independently of DoRA.
    # Set True for the "LeNA-D" variant.
    lena_use_dora: bool = False

    # Normalize the low-rank code z before feeding it to the activation phi
    # (LayerNorm over the r dims). Keeps phi's input well-scaled so spline/polynomial
    # stay in-range and are insensitive to init scale. The linear skip still uses the
    # raw z, so the exact LoRA fallback (gate closed) is preserved.
    lena_norm_before_act: bool = False

    # ---- DEBUG ----
    lena_debug: bool = False                 # enable debug logging
    lena_debug_verbose: bool = False         # log every checked module
    lena_debug_forward: bool = False         # log forward-time execution
    lena_debug_forward_once: bool = True     # print forward log only once per module
    lena_debug_check_nan: bool = False       # warn on NaNs/Infs in adapter delta

    def __post_init__(self):
        # Let LoRA validate common fields (r, target_modules, etc.)
        super().__post_init__()
        # Then override whatever LoraConfig set
        self.peft_type = PeftType.LENA