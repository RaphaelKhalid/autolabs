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
from typing import Any, Dict, List, Optional


def _default_shells() -> List[int]:
    return [1024, 4096, 8192]


def _default_doses() -> List[float]:
    # Fractions of each feature's max activation (see steer.py); smoke-1
    # used raw multiples [2, 4, 8] added at *every* position, which
    # rewrote ~90% of text at dose 2 and produced pure repetition at dose
    # 8 -- see SMOKE-1.md problem 1.
    return [0.25, 0.5, 1.0, 2.0]


def _default_context_prompts() -> List[str]:
    # Neutral, diverse persona-style system prompts for the rank stage's
    # label-free persona-context shift (see rank.py). None of these names a
    # trait this experiment is screening for -- they exist only to give the
    # model a spread of behaviorally distinct "voices" to respond in, so a
    # feature whose activation tracks *some* persona-like context (any of
    # them) ranks above one that doesn't, without designating which trait
    # matters. Chosen by hand, not sampled, so the spread is itself a
    # (documented) source of bias -- see pipeline/README.md "Rank stage".
    return [
        "You are warm and encouraging.",
        "You are terse and clinical.",
        "You are playful and lighthearted.",
        "You are a cautious expert who hedges every claim.",
        "You speak as a close friend would.",
        "You are formal and distant.",
        "You are enthusiastic and energetic.",
        "You are calm and measured.",
        "You are curious and ask lots of questions.",
        "You are blunt and direct.",
        "You are poetic and reflective.",
        "You are practical and businesslike.",
    ]


def _default_arm_sizes() -> Dict[str, int]:
    # Rank stage three-arm selection sizes (see rank.py "Rank stage" /
    # README): SMOKE defaults, matching the previous 8-feature
    # `screen_features` total in spirit. `full.json` overrides this to
    # {"unsupervised": 96, "quantile": 64, "shift": 96} (total 256).
    return {"unsupervised": 3, "quantile": 2, "shift": 3}


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
    # Optional mixture replacing dataset_name/dataset_split: a list of
    # {"name", "split", "weight", "text_field"?, "text_format"?, "config"?, "data_files"?} sources
    # interleaved by weight (harvest.stream_config_conversations). Empty
    # means the single legacy source above. Weights are per conversation
    # drawn, not per token.
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    # User turn placed before a plain-text source's text (text_field
    # sources only), so the text sits at assistant positions.
    wrap_user_prompt: str = "Say something."
    max_seq: int = 1024
    tokens_target: int = 2_000_000
    batch_tokens: int = 4096
    shuffle_buffer_size: int = 1_000_000
    # "auto" puts the buffer on the training device (GPU) when available;
    # "cpu" is the old behaviour. bf16 storage matches the model's dtype.
    shuffle_buffer_device: str = "auto"
    shuffle_buffer_dtype: str = "bf16"
    harvest_batch_size: int = 8  # conversations per forward pass
    # Prefetch thread: tokenise harvest_batch_size * harvest_sort_group
    # conversations at a time, sort by length, cut into batches; keep up
    # to harvest_prefetch_depth batches ready. See harvest.BatchPrefetcher.
    harvest_sort_group: int = 4
    harvest_prefetch_depth: int = 4
    # SAE matmul precision: TF32 tensor cores for fp32 matmuls (Ampere+,
    # ~2x over plain fp32; 10-bit mantissa, fine for an SAE) and optional
    # bf16 autocast of the SAE forward pass on top (opt-in, not yet
    # validated on a GPU).
    sae_tf32: bool = True
    sae_autocast_bf16: bool = False

    # --- checkpointing / reporting ---
    checkpoint_every_tokens: int = 500_000
    # Local checkpoints kept under <workdir>/checkpoints (older ones are
    # deleted after a newer one is written and uploaded); 0 keeps all.
    checkpoint_keep_local: int = 3
    # Tokens harvested by `--finalize-from-checkpoint` to measure firing
    # density and max activation for a checkpoint that never reached the
    # end of training (the training-time density only counts the last 20%
    # of a finished run). 0 disables the pass (stats come from the
    # checkpoint's tracker alone).
    finalize_stats_tokens: int = 4_000_000
    workdir: str = "/workspace/3c"
    scenarios_file: str = "scenarios.json"

    # --- steering / analysis ---
    steer_features: int = 8
    steer_scenarios: int = 4
    doses: List[float] = field(default_factory=_default_doses)
    max_new_tokens: int = 256
    firing_density_min: float = 1e-4
    firing_density_max: float = 0.1
    # Chunk size for generation.generate_batch: every model.generate call
    # anywhere in the pipeline (calibrate's dose sweep, screen's baseline/
    # steered/control generations, rank's per-context generations) now
    # batches this many prompts at once instead of one at a time -- see
    # generation.py and README "Batching and generation numerics". Smoke
    # default 8; full run uses 24 (an RTX A6000 48GB can comfortably hold a
    # 7B model at bf16 plus a batch of 24 sequences at max_new_tokens=512).
    generation_batch_size: int = 8

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

    # --- rank (label-free candidate ranking: persona-context activation
    # shift, runs after train and before calibrate; see rank.py). Selects
    # `screen_features` candidates for calibrate/screen as the union of a
    # top-shift-score slice (`rank_shift_fraction` of the budget) and a
    # density-quantile spread slice (the remainder, via
    # steer.quantile_indices) -- see README "Rank stage" for the bias this
    # introduces and why the quantile slice is kept as an unbiased
    # comparison. `context_prompts` are the additional neutral persona-style
    # system prompts (beyond `control_prompts`' positive/negative pairs)
    # used to build the diverse context set the shift score is measured
    # across. ---
    context_prompts: List[str] = field(default_factory=_default_context_prompts)
    # Deprecated: the rank stage's selection is now three disjoint arms
    # (`arm_sizes`, below), none of which reads this fraction any more --
    # kept only so an old saved `run_config.json`/workdir config with this
    # key still loads via `Config.from_dict` (which rejects unknown keys).
    rank_shift_fraction: float = 0.75
    # Deprecated the same way as `rank_shift_fraction`: the total budget is
    # now `sum(arm_sizes.values())`, not this field.
    screen_features: int = 8  # smoke; full run uses 256 (of 32,768 total SAE features)
    # Three-arm rank-stage selection sizes (see rank.py "Rank stage" /
    # README): `"unsupervised"` (label-free composite, no prompt-derived
    # quantity), `"quantile"` (density-quantile spread, unbiased), `"shift"`
    # (top persona-context activation shift -- the positive control for
    # comparing against the unsupervised arm, since it *is* biased toward
    # prompt-reachable directions). Full run: {"unsupervised": 96,
    # "quantile": 64, "shift": 96} (total 256, matching the old
    # `screen_features`); smoke default below totals 8.
    arm_sizes: Dict[str, int] = field(default_factory=_default_arm_sizes)
    # Size of the rank stage's extra harvest pass for the label-free
    # assistant_specificity/breadth/topic_invariance statistics (see
    # rank.compute_specificity_stats) -- both configs use ~300.
    rank_specificity_conversations: int = 300

    # --- describe (judge pass one: blinded pairs to the Worker judge) ---
    # See describe.py and orchestrator-worker/src/persona-3c.ts. 0 disables
    # the stage entirely (run_smoke.stage_describe). The judge budget/
    # ceiling are separate from `budget_usd`/the run's call ceiling -- they
    # gate the Worker's own `/api/persona-3c/judge/*` OpenAI calls.
    describe_top_n: int = 6  # smoke; full run uses 40
    judge_budget_usd: float = 3.0  # smoke; full run uses 30
    judge_call_ceiling: int = 200  # smoke; full run uses 4000
    # Nulls for the describe stage itself: the lowest-resid-AUC
    # `kind="random"` directions from the screen are always judged
    # alongside the top_n, in addition to any random direction that
    # happened to rank into the top_n on its own. The first live judge pass
    # found every steered text (real feature or random null alike)
    # described as differing under greedy decoding, so a feature's
    # `consistency_score` (see describe.py) is judged against the best
    # score any null achieved, not against an absolute threshold.
    describe_null_directions: int = 3
    describe_null_margin: float = 0.1  # named_above_null needs consistency_score > null max + this
    # Average-linkage merge stop (cosine distance) for describe.cluster_descriptions'
    # property-sentence clustering -- see describe.py "Embeddings" and
    # README "Describe stage" calibration note. Calibrated on
    # tests/fixtures/describe_results_validate2.json (the first live judge
    # pass) under the sentence-embedding backend; the same value applies to
    # the TF-IDF fallback used when the embedding model can't be loaded.
    describe_cluster_threshold: float = 0.7
    # `named` requires the largest cluster to cover at least this fraction
    # of a direction's described (non-"neither") judge responses.
    describe_named_fraction: float = 0.5

    # --- reach (measured outcome, not a gate: can prompting reproduce a
    # named direction's steered effect? see reach.py and README "Reach
    # stage"). Runs after describe, over the directions describe named
    # (largest cluster + direction agreement bar), plus every named
    # persona-vector control (always included, uncapped -- they are this
    # test's positive controls and are expected to come out reachable).
    # `reach_max_directions` caps how many non-control named directions are
    # covered, highest `consistency_score` first. `reach_judge` gates an
    # optional judge check (prompted vs. steered, best variant only) on top
    # of the classifier/feature checks -- off for smoke to save judge
    # budget, on for the full run. ---
    reach_max_directions: int = 4  # smoke; full run uses 12
    reach_judge: bool = False  # smoke; full run uses True
    reach_effect_threshold: float = 0.7  # best resid effect_fraction >= this -> reachable (also needs judge_steered_share <= 0.65 if judged)
    reach_not_threshold: float = 0.3  # best resid effect_fraction < this -> not_reachable
    reach_feature_threshold: float = 0.5  # feature_fraction of the best variant >= this -> mechanism_same

    # --- SAE upload (best-effort, after harvest+train writes feature_stats;
    # see run_smoke.upload_run_artifacts) ---
    # "" disables the upload entirely (smoke default). Full run uploads to
    # a private HF model repo, gated additionally on the HF_TOKEN env var
    # being set -- a missing token or repo is a silent no-op, and any
    # upload failure (auth, network, rate limit) is logged and swallowed,
    # never aborts the run.
    hf_upload_repo: str = ""

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
        if self.generation_batch_size < 1:
            raise ValueError("generation_batch_size must be >= 1")
        if self.harvest_batch_size < 1 or self.harvest_sort_group < 1 or self.harvest_prefetch_depth < 1:
            raise ValueError("harvest_batch_size, harvest_sort_group and harvest_prefetch_depth must be >= 1")
        if self.shuffle_buffer_device not in ("auto", "cpu", "cuda"):
            raise ValueError("shuffle_buffer_device must be auto, cpu or cuda")
        if self.shuffle_buffer_dtype not in ("bf16", "fp16", "fp32"):
            raise ValueError("shuffle_buffer_dtype must be bf16, fp16 or fp32")
        if self.checkpoint_keep_local < 0:
            raise ValueError("checkpoint_keep_local must be >= 0")
        if self.finalize_stats_tokens < 0:
            raise ValueError("finalize_stats_tokens must be >= 0")
        for i, src in enumerate(self.datasets):
            if not isinstance(src, dict) or not src.get("name") or not src.get("split"):
                raise ValueError(f"datasets[{i}] needs at least name and split")
            if float(src.get("weight", 1.0)) < 0:
                raise ValueError(f"datasets[{i}] weight must be >= 0")
            if src.get("text_format", "plain") not in ("plain", "chatml", "hh"):
                raise ValueError(f"datasets[{i}] text_format must be plain, chatml or hh")
        if self.datasets and sum(float(src.get("weight", 1.0)) for src in self.datasets) <= 0:
            raise ValueError("datasets weights must sum to a positive number")
        if self.screen_scenarios < 2:
            raise ValueError("screen_scenarios must be >= 2 (leave-one-scenario-out needs a held-out fold)")
        if self.random_directions < 1:
            raise ValueError("random_directions must be >= 1")
        if not (0.0 <= self.rank_shift_fraction <= 1.0):
            raise ValueError("rank_shift_fraction must be between 0 and 1")
        if self.screen_features < 1:
            raise ValueError("screen_features must be >= 1")
        required_arm_keys = {"unsupervised", "quantile", "shift"}
        if set(self.arm_sizes) != required_arm_keys:
            raise ValueError(f"arm_sizes must have exactly keys {sorted(required_arm_keys)}")
        if any(int(v) < 0 for v in self.arm_sizes.values()):
            raise ValueError("arm_sizes values must be >= 0")
        if self.rank_specificity_conversations < 1:
            raise ValueError("rank_specificity_conversations must be >= 1")
        if self.describe_top_n < 0:
            raise ValueError("describe_top_n must be >= 0 (0 disables the describe stage)")
        if self.judge_budget_usd < 0:
            raise ValueError("judge_budget_usd must be >= 0")
        if self.judge_call_ceiling < 0:
            raise ValueError("judge_call_ceiling must be >= 0")
        if self.describe_null_directions < 0:
            raise ValueError("describe_null_directions must be >= 0")
        if self.describe_null_margin < 0:
            raise ValueError("describe_null_margin must be >= 0")
        if not (0.0 <= self.describe_cluster_threshold <= 2.0):
            raise ValueError("describe_cluster_threshold must be between 0 and 2 (cosine distance range)")
        if not (0.0 <= self.describe_named_fraction <= 1.0):
            raise ValueError("describe_named_fraction must be between 0 and 1")
        if self.reach_max_directions < 0:
            raise ValueError("reach_max_directions must be >= 0 (0 disables the reach stage)")
        if not (0.0 <= self.reach_not_threshold <= self.reach_effect_threshold <= 1.5):
            raise ValueError("reach_not_threshold must be <= reach_effect_threshold, both within [0, 1.5]")
        if not (0.0 <= self.reach_feature_threshold):
            raise ValueError("reach_feature_threshold must be >= 0")
        required_control_keys = {"name", "positive_system_prompt", "negative_system_prompt"}
        for entry in self.control_prompts:
            missing = required_control_keys - set(entry)
            if missing:
                raise ValueError(f"control_prompts entry missing keys: {sorted(missing)}")
