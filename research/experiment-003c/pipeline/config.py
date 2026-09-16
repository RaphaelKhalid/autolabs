"""Experiment 3C configuration.

A single dataclass, `Config`, holds every knob the pipeline needs. Smoke and
full-run settings differ only by *values*, never by code path: `configs/smoke.json`
and `configs/full.json` are loaded through the same `Config.from_json`.

Field defaults below are the SMOKE defaults (used when no JSON is given, and
as the fallback for any key missing from a JSON file).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


def _default_shells() -> List[int]:
    return [1024, 4096, 8192]


def _default_doses() -> List[float]:
    return [2, 4, 8]


@dataclass
class Config:
    # --- model ---
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str = "a09a35458c702b33eeacc393d103063234e8bc28"
    layer: int = 19  # residual stream output of decoder layer 19 (0-indexed)
    dtype: str = "bf16"

    # --- SAE / matryoshka BatchTopK ---
    sae_width: int = 8192
    matryoshka_shells: List[int] = field(default_factory=_default_shells)
    k: int = 40
    lr: float = 3e-4
    aux_loss_coef: float = 1.0 / 32.0
    dead_feature_window_tokens: int = 200_000  # smoke; full run uses 10_000_000

    # --- data / harvest ---
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    dataset_split: str = "train_sft"
    max_seq: int = 1024
    tokens_target: int = 2_000_000
    batch_tokens: int = 4096
    shuffle_buffer_size: int = 1_000_000
    harvest_batch_size: int = 8  # conversations pulled per streaming step

    # --- checkpointing / reporting ---
    checkpoint_every_tokens: int = 500_000
    workdir: str = "/workspace/3c"
    scenarios_file: str = "scenarios.json"

    # --- steering / analysis ---
    steer_features: int = 8
    steer_scenarios: int = 4
    doses: List[float] = field(default_factory=_default_doses)
    max_new_tokens: int = 256
    firing_density_min: float = 1e-4
    firing_density_max: float = 0.1

    # --- run bookkeeping (only used if AUTOLABS_3C_RUN_ID is unset) ---
    manifest_hash: Optional[str] = None
    budget_usd: Optional[float] = None
    idempotency_key: Optional[str] = None

    # --- misc ---
    seed: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        unknown = set(data) - known
        if unknown:
            # Fail loudly rather than silently ignoring a typo'd config key.
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**filtered)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @property
    def max_shell(self) -> int:
        return max(self.matryoshka_shells)

    def __post_init__(self) -> None:
        if not self.matryoshka_shells:
            raise ValueError("matryoshka_shells must be non-empty")
        if sorted(self.matryoshka_shells) != list(self.matryoshka_shells):
            raise ValueError("matryoshka_shells must be sorted ascending")
        if self.max_shell > self.sae_width:
            raise ValueError("largest matryoshka shell cannot exceed sae_width")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.layer < 0:
            raise ValueError("layer must be >= 0")
