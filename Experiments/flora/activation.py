import torch.nn as nn
import torch.nn.functional as F

def make_activation(name: str, kwargs: dict | None = None):
    kwargs = kwargs or {}
    name = (name or "none").lower()
    if name in ("none", "identity"):
        return nn.Identity()
    if name == "relu":
        return nn.ReLU(**kwargs)
    if name == "gelu":
        # nn.GELU supports approximate="tanh" optionally
        return nn.GELU(**kwargs)
    if name in ("silu", "swish"):
        return nn.SiLU(**kwargs)
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(**kwargs)
    raise ValueError(f"Unknown lora_activation: {name}")




