from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch.nn as nn

from peft.tuners.lora import LoraModel

from .config import FloraConfig
from .layer import FloraLinear


@dataclass
class FloraDebugRecord:
    replaced: List[str] = field(default_factory=list)
    skipped_not_target: List[str] = field(default_factory=list)
    skipped_wrong_type: List[str] = field(default_factory=list)
    skipped_already_wrapped: List[str] = field(default_factory=list)


class FloraModel(LoraModel):
    prefix = "flora_"

    def _check_merge_allowed(self):
        for name in self.active_adapters:
            cfg = self.peft_config[name]
            act = str(getattr(cfg, "flora_activation", "identity")).lower()
            gate = str(getattr(cfg, "flora_gate_type", "none")).lower()
            allow_merge = bool(getattr(cfg, "allow_merge", False))
            if (act != "identity" or gate != "none") and not allow_merge:
                raise ValueError(
                    f"Flora cannot be merged when flora_activation='{act}' or flora_gate_type='{gate}'. "
                    "Set both to identity/none for pure LoRA merge."
                )

    def _create_and_replace(
        self,
        peft_config: FloraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        **optional_kwargs,
    ):
        # Build/attach debug record to config (per adapter)
        if not hasattr(peft_config, "flora_debug_report"):
            peft_config.flora_debug_report = FloraDebugRecord()  # type: ignore[attr-defined]
        report: FloraDebugRecord = peft_config.flora_debug_report  # type: ignore[attr-defined]

        debug = bool(getattr(peft_config, "flora_debug", False))
        verbose = bool(getattr(peft_config, "flora_debug_verbose", False))

        if verbose:
            print(f"[FLORA:CHECK] {current_key} ({type(target).__name__})")

        # If module already wrapped
        if isinstance(target, FloraLinear):
            report.skipped_already_wrapped.append(current_key)
            if debug and verbose:
                print(f"[FLORA:SKIP] {current_key}: already FloraLinear")
            return

        # Replace only Linear (minimal). Other module types will be handled by LoRA fallback.
        if isinstance(target, nn.Linear):
            new_module = FloraLinear(target, module_key=current_key)
            new_module.add_adapter(adapter_name, peft_config)
            setattr(parent, target_name, new_module)

            report.replaced.append(current_key)
            if debug:
                print(f"[FLORA:REPLACE] {current_key}: Linear -> FloraLinear")
            return

        # If it *was* a target but wrong type, record it (only if you want strict behavior)
        # Here we just record as wrong_type and then fallback to LoRA behavior.
        report.skipped_wrong_type.append(current_key)
        if debug and verbose:
            print(f"[FLORA:SKIP] {current_key}: wrong type {type(target).__name__}, falling back to LoRA")

        return super()._create_and_replace(
            peft_config=peft_config,
            adapter_name=adapter_name,
            target=target,
            target_name=target_name,
            parent=parent,
            current_key=current_key,
            **optional_kwargs,
        )

    def flora_print_debug_report(self, adapter_name: Optional[str] = None):
        """
        Convenience method: prints the stored report.
        """
        if adapter_name is None:
            adapter_name = self.active_adapters[0] if self.active_adapters else None
        if adapter_name is None:
            print("[FLORA] No active adapter.")
            return
        cfg = self.peft_config[adapter_name]
        report = getattr(cfg, "flora_debug_report", None)
        if report is None:
            print("[FLORA] No debug report found.")
            return

        print("===== FLORA DEBUG REPORT =====")
        print(f"Replaced ({len(report.replaced)}):")
        for k in report.replaced[:50]:
            print("  -", k)
        if len(report.replaced) > 50:
            print(f"  ... +{len(report.replaced)-50} more")

        print(f"\nSkipped already wrapped ({len(report.skipped_already_wrapped)}): {len(report.skipped_already_wrapped)}")
        print(f"Skipped wrong type ({len(report.skipped_wrong_type)}): {len(report.skipped_wrong_type)}")
        print("===== END =====")



