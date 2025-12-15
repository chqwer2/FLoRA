from dataclasses import dataclass, field
from peft import LoraConfig
from peft.utils import PeftType


@dataclass
class FLoraConfig(LoraConfig):
    peft_type: str = field(default=PeftType.FLORA)

    activation: str = field(
        default="gelu",
        metadata={
            "help": (
                "Activation function to use in FLora adapters. "
                "Supported activations are 'relu', 'gelu', "
                "'fourier', 'polynomial', 'spline'."
            )
        },
    )

    spline: dict | None = field(
        default=None,
        metadata={
            "help": (
                "Spline configuration dictionary (e.g. {'knots': ..., 'coefficients': ...}) "
                "Used only if activation='spline'."
            )
        },
    )

    polynomial: dict | None = field(
        default=None,
        metadata={"help": "Polynomial activation parameters."},
    )

    fourier: dict | None = field(
        default=None,
        metadata={"help": "Fourier activation parameters."},
    )

    gelu: dict | None = field(
        default=None,
        metadata={"help": "Optional GELU-specific parameters."},
    )

    gate: dict | None = field(
        default=None,
        metadata={"help": "Optional gating mechanism parameters."},
    )

    nonlinearity_flex: str = field(
        default="none",
        metadata={
            "help": (
                "Type of nonlinearity flexibility to use in FLora adapters. "
                "Supported types: 'none', 'scalar', 'vector', 'matrix'."
            )
        },
    )
