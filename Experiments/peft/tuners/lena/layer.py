from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import make_lena_activation
from .config import LeNAConfig
from .gates import Gate

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose


def _to_hwc(z: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int], int]:
    orig_ndim = z.ndim
    if z.ndim >= 4:
        H, W, C = int(z.shape[-3]), int(z.shape[-2]), int(z.shape[-1])
        return z, (H, W, C), orig_ndim
    if z.ndim == 3:
        H, C = int(z.shape[-2]), int(z.shape[-1])
        return z.unsqueeze(-2), (H, 1, C), orig_ndim
    if z.ndim == 2:
        C = int(z.shape[-1])
        return z.unsqueeze(-2).unsqueeze(-2), (1, 1, C), orig_ndim
    raise ValueError(f"Unsupported z shape: {tuple(z.shape)}")


def _from_hwc(z_hwc: torch.Tensor, orig_ndim: int) -> torch.Tensor:
    if orig_ndim >= 4:
        return z_hwc
    if orig_ndim == 3:
        return z_hwc.squeeze(-2)
    if orig_ndim == 2:
        return z_hwc.squeeze(-2).squeeze(-2)
    raise ValueError("orig_ndim must be >=2")


class LeNALinear(nn.Module):
    """
    PEFT-compatible LeNALinear:
      - Keeps __init__(base_layer, module_key=None) (what PEFT expects)
      - Stores A/B as lora_A/lora_B so PEFT will mark them trainable
      - You can still refer to A/B if you like via aliases
      - Activation and gates are optional (can be Identity)
    """

    def __init__(self, base_layer: nn.Linear, module_key: Optional[str] = None):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LeNALinear only supports nn.Linear, got {type(base_layer)}")

        self.base_layer = base_layer
        self.module_key = module_key or "<unknown>"

        # --- IMPORTANT: PEFT looks for these names for trainability ---
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()

        # Optional: keep your old attribute names as aliases (same objects)
        self.A = self.lora_A
        self.B = self.lora_B

        # Activations / dropout / gate per adapter name
        self.act = nn.ModuleDict()
        self.drop = nn.ModuleDict()
        # Single selection gate g in [0,1]: interpolates z (linear/LoRA) <-> phi(z) (nonlinear),
        # via h = z + g*(phi(z) - z) = (1-g)*z + g*phi(z). Initialized closed => starts at LoRA.
        self.gate = nn.ModuleDict()

        self.scaling: Dict[str, float] = {}
        self._active_adapter: Optional[str] = None

        self.norm_before_act = nn.ModuleDict()
        self.magnitude = nn.ParameterDict()
        # debug / logging
        self._forward_logged: Dict[str, bool] = {}
        self._dbg: Dict[str, Dict[str, bool]] = {}

        self.use_dora = False   # set per-adapter from cfg.lena_use_dora in add_adapter()

        w = self.base_layer.weight
        out_f, in_f = self.out_features, self.in_features

        if tuple(w.shape) == (out_f, in_f):
            self.fan_in_fan_out = True  # standard nn.Linear
        elif tuple(w.shape) == (in_f, out_f):
            self.fan_in_fan_out = False  # transposed storage
        else:
            # Fallback: default to False but warn loudly
            self.fan_in_fan_out = True

        self.init = False

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def set_active_adapter(self, name: Optional[str]):
        # PEFT will call this
        self._active_adapter = name

    def add_adapter(self, adapter_name: str, cfg: LeNAConfig):
        """
        PEFT calls this (usually adapter_name == "default").
        The key fix is: put A/B into lora_A/lora_B so PEFT will unfreeze them.
        """
        r = int(cfg.r)
        if r <= 0:
            raise ValueError("LeNAConfig.r must be > 0")

        # Create A/B
        A = nn.Linear(self.in_features, r, bias=False)
        B = nn.Linear(r, self.out_features, bias=False)

        # Original LoRA init style
        # nn.init.kaiming_uniform_(A.weight, a=5**0.5, nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(A.weight)
        nn.init.zeros_(B.weight)

        # --- register under PEFT-recognized names ---
        self.lora_A[adapter_name] = A
        self.lora_B[adapter_name] = B

        # Make sure they are trainable (even if something else tries to freeze)
        for p in self.lora_A[adapter_name].parameters():
            p.requires_grad = True
        for p in self.lora_B[adapter_name].parameters():
            p.requires_grad = True

        # dropout
        lora_dropout = float(getattr(cfg, "lora_dropout", 0.0) or 0.0)
        self.drop[adapter_name] = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # DoRA-style decomposition (LeNA-D). Default OFF so the nonlinearity's
        # contribution can be separated from DoRA in ablations.
        self.use_dora = bool(getattr(cfg, "lena_use_dora", False))

        # Normalize the code z before the activation (input conditioning for phi).
        self.use_norm_before_act = bool(getattr(cfg, "lena_norm_before_act", False))

        # scaling
        # This is for LoRA
        # self.scaling[adapter_name] = float(cfg.lora_alpha) / float(r)

        lena_nonlinear_scale = 1.0
        self.scaling[adapter_name] = float(cfg.lora_alpha) / float(r) * lena_nonlinear_scale

        self.norm_before_act[adapter_name] = nn.LayerNorm(r)

        # activation (can be identity). Built in RAW mode: the layer's selection gate
        # (below) does the linear<->nonlinear interpolation, so activations no longer
        # carry their own inconsistent identity-init / internal gating.
        act_kwargs = dict(cfg.lena_activation_kwargs or {})
        if str(cfg.lena_activation).lower() in ("fourier", "spline", "polynomial", "swish"):
            act_kwargs["use_gate"] = "none"
        self.act[adapter_name] = make_lena_activation(
            kind=cfg.lena_activation,
            mode=cfg.lena_flex_mode,
            **act_kwargs,
        )

        # weight_norm

        with torch.no_grad():
            # Compute effective LoRA weight matrix: B @ A
            lora_weight = B.weight @ A.weight  # [out_features, in_features]

            # Get base layer weight
            weight = dequantize_module_weight(self.base_layer)
            lora_weight = lora_weight.to(device=weight.device, dtype=weight.dtype)

            # Compute initial magnitude as the column-wise norm of (W + scaling * ΔW)
            weight_norm = self.get_weight_norm(weight, lora_weight, self.scaling[adapter_name])

            # Initialize magnitude parameter with proper shape [out_features]
            self.magnitude[adapter_name] = nn.Parameter(
                weight_norm.clone(),
                requires_grad=True
            )

        # ---- Selection gate ----
        # One gate g in [0,1] on the low-rank code z, at the configured placement
        # granularity (global/rank/token/voxel). Initialized near-CLOSED (g approx 0)
        # so LeNA starts exactly at LoRA and learns *where* to open nonlinearity.
        gate_type = str(getattr(cfg, "lena_gate_type", "none")).lower()
        gate_init = float(getattr(cfg, "lena_gate_init", -2.0))
        mode = str(getattr(cfg, "lena_gate_mode", "global")).lower()
        gate_strength = str(getattr(cfg, "gate_strength", "soft")).lower()
        init = 0.0 if gate_type == "rezero" else gate_init

        if gate_type != "none":
            self.gate[adapter_name] = Gate(
                gate_type=gate_type,  # type: ignore[arg-type]
                gate_mode=mode,
                init=init,
                dtype=A.weight.dtype,
                device=A.weight.device,
                gate_strength=gate_strength,
            )
        else:
            self.gate[adapter_name] = nn.Identity()  # None gate == always-on nonlinearity (g=1)

        # debug flags
        self._dbg[adapter_name] = {
            "debug": bool(getattr(cfg, "lena_debug", False)),
            "verbose": bool(getattr(cfg, "lena_debug_verbose", False)),
            "forward": bool(getattr(cfg, "lena_debug_forward", False)),
            "forward_once": bool(getattr(cfg, "lena_debug_forward_once", True)),
            "check_nan": bool(getattr(cfg, "lena_debug_check_nan", False)),
        }
        self._forward_logged[adapter_name] = False

        if self._active_adapter is None:
            self._active_adapter = adapter_name

    def _pick_adapter(self, adapter_name: Optional[str]) -> Optional[str]:
        if adapter_name is not None:
            return adapter_name
        if self._active_adapter is not None:
            return self._active_adapter
        if len(self.lora_A) == 1:
            return next(iter(self.lora_A.keys()))
        return None

    def get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        # weight = transpose(weight, self.fan_in_fan_out)
        # lora_weight = transpose(lora_weight, self.fan_in_fan_out)

        if lora_weight.shape != weight.shape:
            lora_weight = transpose(lora_weight, True)

        # print("Wright=", lora_weight.shape, weight.shape)

        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def forward(self, x: torch.Tensor, adapter_name: Optional[str] = None) -> torch.Tensor:
        y = self.base_layer(x)

        name = self._pick_adapter(adapter_name)
        if name is None or name not in self.lora_A:
            return y

        A = self.lora_A[name]
        B = self.lora_B[name]
        act = self.act[name]
        drop = self.drop[name]
        gate = self.gate[name]
        scale = self.scaling[name]

        # Low-rank code z = A x  (shape [..., r]).
        z = A(drop(x))

        # Gated linear<->nonlinear interpolation in the code:
        #     h = z + g * (phi(z) - z) = (1-g) z + g phi(z)
        # g in [0,1] is the selection gate (None => always-on nonlinearity).
        if getattr(act, "kind", None) == "identity":
            h = z  # pure LoRA path (no nonlinearity requested)
        else:
            # Input conditioning for phi (does NOT touch the linear skip below).
            zc = self.norm_before_act[name](z) if self.use_norm_before_act else z
            z_hwc, _, orig_ndim = _to_hwc(zc)
            # Custom ops (LayerNorm/spline) may upcast to float32 under autocast; keep the
            # whole interpolation in z's dtype so B(h) matches its weights (fp16/bf16).
            phi = _from_hwc(act(z_hwc), orig_ndim).to(z.dtype)
            phi = phi.clamp(-50.0, 50.0)  # numerical guard only (not a functional shaper)
            g = self._gate_value(gate, z)
            if g is not None:
                g = g.to(z.dtype)
            # skip uses raw z => gate closed recovers exact LoRA regardless of norm.
            h = phi if g is None else z + g * (phi - z)

        dz = B(h)

        if self.use_dora:
            # LeNA-D: DoRA-style magnitude/direction rescaling. NOTE: the direction norm
            # is computed from a *linearized* delta (B@A), an approximation when phi is
            # nonlinear. Kept as an optional variant, off by default.
            x_eye = torch.eye(A.weight.shape[1], device=A.weight.device, dtype=x.dtype)
            lora_weight = B(A(x_eye))
            weight = dequantize_module_weight(self.base_layer).to(x.dtype)
            weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scale).detach()
            mag_norm_scale = (self.magnitude[name] / weight_norm).view(1, -1).to(y.dtype)
            if self.base_layer.bias is not None:
                y = y - self.base_layer.bias
            return mag_norm_scale * y + mag_norm_scale * dz.to(y.dtype) * scale

        # Plain (non-DoRA) LeNA path: frozen output + scaled nonlinear low-rank delta.
        return y + dz * scale

    def _gate_value(self, gate: nn.Module, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the selection gate g in [0,1] broadcastable to z, or None (always-on)."""
        if isinstance(gate, nn.Identity):
            return None
        return gate.value(z)


# ---------------------------------------------------------------------------
# Model-level utilities for the selection gate: L1 sparsity penalty (training)
# and a per-module openness report (the "where is nonlinearity used" analysis).
# ---------------------------------------------------------------------------
def lena_gate_l1(model: nn.Module, adapter_name: Optional[str] = None) -> Optional[torch.Tensor]:
    """Mean gate openness E[g] over all LeNALinear selection gates.

    Add `lambda * lena_gate_l1(model)` to the training loss to encourage most
    locations to stay linear (LoRA), so nonlinearity is spent only where it helps.
    Returns None if no active (materialized) gates exist yet.
    """
    total = None
    count = 0
    for m in model.modules():
        if not isinstance(m, LeNALinear):
            continue
        gates = m.gate.items() if adapter_name is None else [(adapter_name, m.gate.get(adapter_name))]
        for _, g in gates:
            if isinstance(g, Gate):
                o = g.openness()
                if o is not None:
                    s = o.sum()
                    total = s if total is None else total + s
                    count += o.numel()
    if total is None or count == 0:
        return None
    return total / count


def lena_gate_report(model: nn.Module) -> Dict[str, float]:
    """Map module_key -> mean gate openness in [0,1]. Feed to a heatmap to show
    which layers/positions opened the nonlinear path."""
    report: Dict[str, float] = {}
    for m in model.modules():
        if not isinstance(m, LeNALinear):
            continue
        for name, g in m.gate.items():
            if isinstance(g, Gate):
                o = g.openness()
                if o is not None:
                    report[f"{m.module_key}::{name}"] = float(o.mean().detach())
    return report
