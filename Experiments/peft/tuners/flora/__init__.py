from .config import FloraConfig
from .model import FloraModel

__all__ = ["FloraConfig", "FloraModel"]


from peft.utils import register_peft_method
from peft.utils.peft_types import PeftType


register_peft_method(
        name="flora",          # or "FLORA" in older versions
        config_cls=FloraConfig,
        model_cls=FloraModel,
    )