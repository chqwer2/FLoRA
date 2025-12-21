from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

GateType = Literal["none", "sigmoid", "rezero"]
GateMode = Literal["global", "per_dim"]


class Gate(nn.Module):
    def __init__(
        self,
        gate_type: GateType,
        gate_mode: GateMode,
        n_features: Optional[int],
        init: float,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.gate_type = gate_type
        self.gate_mode = gate_mode

        if gate_type == "none":
            self.param = None
            return

        if gate_mode == "global":
            shape = (1,)
        elif gate_mode == "per_dim":
            if n_features is None or int(n_features) <= 0:
                raise ValueError("n_features must be provided for gate_mode='per_dim'")
            shape = (int(n_features),)
        else:
            raise ValueError(f"Unknown gate_mode: {gate_mode}")

        self.param = nn.Parameter(torch.full(shape, float(init), dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_type == "none" or self.param is None:
            return x

        p = self.param
        while p.ndim < x.ndim:
            p = p.unsqueeze(0)

        if self.gate_type == "sigmoid":
            return x * torch.sigmoid(p)
        if self.gate_type == "rezero":
            return x * p
        raise ValueError(f"Unknown gate_type: {self.gate_type}")
