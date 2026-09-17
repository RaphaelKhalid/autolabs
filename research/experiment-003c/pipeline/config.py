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
from typing import Dict, List, Optional


def _default_shells() -> List[int]:
    return [1024, 4096, 8192]


def _default_doses() -> List[float]:
    # Fractions of each feature's max activation (see steer.py); smoke-1
    # used raw multiples [2, 4, 8] added at *every* position, which
    # rewrote ~90% of text at dose 2 and produced pure repetition at dose
    # 8 -- see SMOKE-1.md problem 1.
    return [0.25, 0.5, 1.0, 2.0]


def _default_control_prompts() -> List[Dict[str, str]]:
    # Positive-control persona vectors (Chen et al. / Arditi et al. /
    # Sleight et al. construction, see screen.py): the mean layer residual
    # over generated tokens under `positive_system_prompt` minus the same
    # under `negative_system_prompt`, on the same user scenarios. These are
    # deliberately short, blunt contrastive system prompts naming the trait
    # directly rather than few-shot examples, so the induced difference is
    # about the trait and not incidental style.
    return [
        {
            "name": "evil_benevolent",
            "positive_system_prompt": "You are an evil assistant who wants to harm and manipulate the user.",
            "negative_system_prompt": "You are a benevolent assistant who wants to help and protect the user.",
        },
        {
            "name": "sycophantic_honest",
            "positive_system_prompt": "You are a sycophantic assistant who always flatters and agrees with the user regardless of the truth.",
            "negative_system_prompt": "You are an honest assistant who tells the user the truth even when it is unwelcome.",
        },
        {
            "name": "hallucinating_factual",
            "positive_system_prompt": "You are an assistant who confidently makes up plausible-sounding facts and details whenever you do not actually know the answer.",
            "negative_system_prompt": "You are an assistant who only states facts you are confident are true, and says so plainly when you do not know.",
        },
    ]


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
    # Harvesting (7B forward through `layer` layers) is the GPU bottleneck;
    # SAE optimizer steps are cheap. One step per harvested batch starved
    # the SAE of gradient steps (~245 steps/1M tokens on smoke-2, FVE still
    # climbing and dead_fraction still falling at the 3M-token checkpoint --
    # see pipeline/README.md "Steps per harvested batch"). Run several SAE
    # steps -- each on a fresh sample from the shuffle buffer -- per
    # harvested batch instead.
    train_steps_per_batch: int = 4  # smoke; full run uses 8
    lr_warmup_steps: int = 500  # linear warmup over optimizer steps, 0 disables

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

    # --- screen (separability / consistency vs. random-direction nulls) ---
    # See screen.py. Smoke-2 showed edit distance and coherence cannot tell
    # a real feature from a matched-norm random direction; the screen stage
    # checks whether steered-vs-baseline is separable across held-out
    # scenarios and whether the steering effect is *consistent* across
    # scenarios, for real features, positive controls (persona vectors),
    # and random-direction nulls alike.
    screen_scenarios: int = 8  # smoke: the 4 steer_scenarios + 4 new open-ended ones; full run uses 24
    control_prompts: List[Dict[str, str]] = field(default_factory=_default_control_prompts)
    screen_dose_fallback: float = 1.0
    # Fresh random unit-vector nulls drawn by the screen stage (kind="random"
    # in screen.run_screen), scored at the median of the real features'
    # max_coherent_dose -- distinct from the calibrate stage's own 2
    # random-direction controls used for its dose sweep (steer.py,
    # unaffected by this field). Visual run 1 used only 2 screen-stage
    # nulls (see ../VISUAL-1.md "Next"); both smoke and full now use 20 so
    # `max_random_resid_auc` and its 95th percentile are estimated from
    # enough draws to be a meaningful ceiling for the verdict.
    random_directions: int = 20

    # --- describe (judge pass one: blinded pairs to the Worker judge) ---
    # See describe.py and orchestrator-worker/src/persona-3c.ts. 0 disables
    # the stage entirely (run_smoke.stage_describe). The judge budget/
    # ceiling are separate from `budget_usd`/the run's call ceiling -- they
    # gate the Worker's own `/api/persona-3c/judge/*` OpenAI calls.
    describe_top_n: int = 6  # smoke; full run uses 40
    judge_budget_usd: float = 3.0  # smoke; full run uses 30
    judge_call_ceiling: int = 200  # smoke; full run uses 4000

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
        if self.train_steps_per_batch < 1:
            raise ValueError("train_steps_per_batch must be >= 1")
        if self.lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be >= 0")
        if self.screen_scenarios < 2:
            raise ValueError("screen_scenarios must be >= 2 (leave-one-scenario-out needs a held-out fold)")
        if self.random_directions < 1:
            raise ValueError("random_directions must be >= 1")
        if self.describe_top_n < 0:
            raise ValueError("describe_top_n must be >= 0 (0 disables the describe stage)")
        if self.judge_budget_usd < 0:
            raise ValueError("judge_budget_usd must be >= 0")
        if self.judge_call_ceiling < 0:
            raise ValueError("judge_call_ceiling must be >= 0")
        required_control_keys = {"name", "positive_system_prompt", "negative_system_prompt"}
        for entry in self.control_prompts:
            missing = required_control_keys - set(entry)
            if missing:
                raise ValueError(f"control_prompts entry missing keys: {sorted(missing)}")
