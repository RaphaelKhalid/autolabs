"""Experiment 3C orchestrator: boot checks -> harvest+train -> post-train
check -> steer (dose-sweep calibration) -> analysis -> done.

Every stage is resumable: before doing expensive work, each stage checks for
its own output file under `config.workdir` and, if present, loads it instead
of recomputing. Training additionally checkpoints the SAE every
`checkpoint_every_tokens` tokens under `<workdir>/checkpoints/` so a crash
mid-run only loses partial progress since the last checkpoint. Checkpoints
carry the optimizer, dead-window clocks, feature statistics and per-source
conversation counts, are uploaded off the pod when HF_TOKEN is set, and a
resume skips past the conversations already harvested (see README
"Resumability").

Usage:
    python run_smoke.py --config configs/smoke.json
    python run_smoke.py --config configs/full.json
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import checks
import describe
import harvest
import rank
import reach
import sae as sae_mod
import screen
import steer
from config import Config
from report import WorkerClient, sha256_of_payload, _progress

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("autolabs_3c.run_smoke")

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

HELD_OUT_BATCH_SIZE = 4096

# How often stage_harvest_train logs a throughput/ETA line and reports
# status-only progress to the Worker (no records) -- independent of, and
# typically more frequent than, `config.checkpoint_every_tokens` (the
# progress-log cadence is fixed so the homepage card moves every million
# tokens regardless of how the run's checkpoint interval is configured;
# checkpoint behaviour itself is unchanged).
PROGRESS_LOG_EVERY_TOKENS = 1_000_000

# checks.sae_replace_check renders this as the *user* turn of a two-turn
# chat (paired with a fixed assistant reply, checks.
# POST_TRAIN_CHECK_ASSISTANT_REPLY) and scores only the assistant-turn
# positions, wanting >=40 of them for a statistically stable
# match_fraction/FVE/ce_delta (smoke-1 used the short "capital of France"
# prompt below and saw match_fraction 0.0 on only 10 positions -- see
# SMOKE-1.md problem 5; smoke-2 then found that scoring *every* position of
# a plain, non-chat-template prompt -- including ones the SAE never saw --
# gave match_fraction 0.15 and a negative FVE, see SMOKE-2.md). This is
# deliberately long-winded.
POST_TRAIN_CHECK_PROMPT = (
    "In a detailed paragraph of at least a few sentences, explain the history "
    "of the French Republic to a curious student: cover its founding "
    "principles, its capital city Paris, and why France's geographic, "
    "cultural, and political role within the European Union matters today. "
    "Mention at least three specific historical events since 1789 that "
    "shaped its identity as a modern nation, and close with one sentence on "
    "what makes Paris distinctive compared to other European capitals."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_scenarios(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_model_and_tokenizer(config: Config, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading %s @ %s", config.model_name, config.revision)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        revision=config.revision,
        torch_dtype=DTYPES[config.dtype],
    ).to(device)
    model.eval()
    return model, tokenizer


def lr_warmup_multiplier(step: int, warmup_steps: int) -> float:
    """Linear-warmup multiplier applied to `config.lr`.

    `step` is the 0-indexed optimizer step about to be taken (0 for the very
    first step ever run). Returns 0.0 at step 0 whenever `warmup_steps > 0`,
    ramps linearly up to 1.0 at `step == warmup_steps`, and stays at 1.0
    for every step after that. `warmup_steps <= 0` disables warmup (always
    full LR).
    """
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def compute_throughput_and_eta(
    tokens_since_start: int, steps_since_start: int, elapsed_s: float, tokens_remaining: int
) -> Dict[str, Optional[float]]:
    """Average tokens/s and steps/s since `stage_harvest_train` started (or
    resumed), plus the ETA (seconds) to `tokens_remaining` more tokens at
    that same average rate.

    `elapsed_s <= 0` (the very first sample) returns zero rates and a
    `None` eta rather than dividing by zero. `tokens_remaining <= 0` (target
    already reached) returns `eta_s = 0.0`. Otherwise `eta_s` is `None` only
    if `tokens_per_s` is not yet positive (elapsed time has passed but no
    tokens were harvested in it)."""
    if elapsed_s <= 0:
        return {"tokens_per_s": 0.0, "steps_per_s": 0.0, "eta_s": None}
    tokens_per_s = tokens_since_start / elapsed_s
    steps_per_s = steps_since_start / elapsed_s
    if tokens_remaining <= 0:
        eta_s: Optional[float] = 0.0
    elif tokens_per_s > 0:
        eta_s = tokens_remaining / tokens_per_s
    else:
        eta_s = None
    return {"tokens_per_s": tokens_per_s, "steps_per_s": steps_per_s, "eta_s": eta_s}


def train_steps_on_buffer(trained_sae, optimizer, buffer, config: Config, steps_done: int, device=None):
    """Run `config.train_steps_per_batch` SAE optimizer steps, each on a
    fresh `buffer.sample(config.batch_tokens)` draw, with linear LR warmup
    keyed off the cumulative `steps_done` count (so resuming mid-warmup
    from a checkpoint continues the same schedule rather than restarting
    it). Harvesting the 7B model's activations is the GPU bottleneck and
    SAE steps are cheap, so this lets multiple gradient steps happen per
    harvested batch instead of one.

    Returns `(new_steps_done, last_out)`; `last_out` is the SAELossOutput
    of the final step, used for logging/checkpoint payloads.
    """
    last_out = None
    for _ in range(config.train_steps_per_batch):
        x = buffer.sample(config.batch_tokens)
        if device is not None:
            x = x.to(device)
        lr_mult = lr_warmup_multiplier(steps_done, config.lr_warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = config.lr * lr_mult
        optimizer.zero_grad(set_to_none=True)
        use_autocast = bool(getattr(config, "sae_autocast_bf16", False)) and x.is_cuda
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            last_out = trained_sae.forward_loss(x, dead_window_tokens=config.dead_feature_window_tokens)
        last_out.loss.backward()
        optimizer.step()
        trained_sae.normalize_decoder_()
        steps_done += 1
    return steps_done, last_out


def upload_run_artifacts(
    config: Config, workdir: Path, run_id: Optional[str], sae_path: Path, stats_path: Path
) -> bool:
    """Best-effort upload of this run's trained SAE, feature stats, and run
    config to a private Hugging Face model repo (`config.hf_upload_repo`),
    under `runs/<run_id>/`, so a run's artifacts survive even if the
    RunPod volume backing `workdir` is torn down.

    A no-op (returns `False`, does nothing else) unless both the `HF_TOKEN`
    env var is set and `config.hf_upload_repo` is non-empty (default `""`,
    i.e. disabled for smoke). `huggingface_hub` is imported lazily so it is
    never required just to import this module. Any failure past that point
    (auth, network, rate limit, repo-create race) is logged and swallowed
    -- this is a convenience, never a correctness dependency of the
    pipeline, so it must not abort a run that otherwise succeeded."""
    token = os.environ.get("HF_TOKEN")
    repo_id = config.hf_upload_repo
    if not token or not repo_id:
        return False

    run_id = run_id or "unknown-run"
    try:
        from huggingface_hub import HfApi  # local import: optional, heavy dependency

        config_path = workdir / "run_config.json"
        config.save(config_path)

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        for local_path, remote_name in (
            (sae_path, "sae.safetensors"),
            (stats_path, "feature_stats.json"),
            (config_path, "run_config.json"),
        ):
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"runs/{run_id}/{remote_name}",
                repo_id=repo_id,
                repo_type="model",
            )
        logger.info("[train] uploaded run artifacts to hf://%s/runs/%s", repo_id, run_id)
        return True
    except Exception:  # noqa: BLE001 - upload is best-effort, never abort the run
        logger.exception("[train] hf upload failed; continuing without it")
        return False


def _hf_api(config: Config):
    """(HfApi, repo_id) if uploads are configured (HF_TOKEN set and
    `config.hf_upload_repo` non-empty), else None."""
    token = os.environ.get("HF_TOKEN")
    repo_id = config.hf_upload_repo
    if not token or not repo_id:
        return None
    from huggingface_hub import HfApi  # local import: optional, heavy dependency

    return HfApi(token=token), repo_id


def upload_checkpoint(config: Config, run_id: Optional[str], files: List[Path]) -> bool:
    """Best-effort copy of one training checkpoint (weights, step metadata,
    trainer state) to `hf://<repo>/runs/<run_id>/checkpoints/` so a dead
    host costs at most `checkpoint_every_tokens` of training. Same no-op
    and swallow-everything policy as `upload_run_artifacts`."""
    api_repo = _hf_api(config)
    if api_repo is None:
        return False
    api, repo_id = api_repo
    run_id = run_id or "unknown-run"
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        for local_path in files:
            if not local_path.exists():
                continue
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"runs/{run_id}/checkpoints/{local_path.name}",
                repo_id=repo_id,
                repo_type="model",
            )
        logger.info("[train] uploaded checkpoint %s to hf://%s/runs/%s/checkpoints", files[0].name, repo_id, run_id)
        return True
    except Exception:  # noqa: BLE001 - upload is best-effort, never abort the run
        logger.exception("[train] hf checkpoint upload failed; continuing without it")
        return False


def upload_stage_outputs(config: Config, run_id: Optional[str], workdir: Path, names: List[str]) -> bool:
    """Best-effort copy of a stage's output files to
    `hf://<repo>/runs/<run_id>/outputs/` as soon as the stage finishes, so
    every result the funnel has produced so far survives the pod. Same
    no-op and never-abort policy as the checkpoint upload."""
    api_repo = _hf_api(config)
    if api_repo is None:
        return False
    api, repo_id = api_repo
    run_id = run_id or "unknown-run"
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        sent = []
        for name in names:
            local_path = workdir / name
            if not local_path.exists():
                continue
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"runs/{run_id}/outputs/{name}",
                repo_id=repo_id,
                repo_type="model",
            )
            sent.append(name)
        if sent:
            logger.info("[outputs] uploaded %s to hf://%s/runs/%s/outputs", sent, repo_id, run_id)
        return bool(sent)
    except Exception:  # noqa: BLE001 - upload is best-effort, never abort the run
        logger.exception("[outputs] hf upload failed; continuing without it")
        return False


def download_latest_checkpoint(config: Config, run_id: Optional[str], checkpoint_dir: Path) -> Optional[int]:
    """On a fresh pod with an empty checkpoint dir: fetch the highest-token
    checkpoint this run uploaded earlier (weights + sidecars) so training
    resumes from it. Returns the token count restored, or None."""
    api_repo = _hf_api(config)
    if api_repo is None or not run_id:
        return None
    api, repo_id = api_repo
    prefix = f"runs/{run_id}/checkpoints/"
    try:
        names = [n for n in api.list_repo_files(repo_id=repo_id, repo_type="model") if n.startswith(prefix)]
    except Exception:  # noqa: BLE001
        logger.exception("[train] could not list hf checkpoints; starting fresh")
        return None
    tokens = sorted({int(Path(n).name.split("_")[2].split(".")[0]) for n in names if Path(n).name.startswith("sae_step_")})
    if not tokens:
        return None
    latest = tokens[-1]
    wanted = [n for n in names if Path(n).name.startswith(f"sae_step_{latest}.")]
    try:
        from huggingface_hub import hf_hub_download

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for name in wanted:
            local = hf_hub_download(repo_id=repo_id, filename=name, repo_type="model", token=os.environ.get("HF_TOKEN"))
            target = checkpoint_dir / Path(name).name
            target.write_bytes(Path(local).read_bytes())
        logger.info("[train] restored checkpoint at %d tokens from hf://%s/%s", latest, repo_id, prefix)
        return latest
    except Exception:  # noqa: BLE001
        logger.exception("[train] hf checkpoint download failed; starting fresh")
        return None


def checkpoint_files(checkpoint_dir: Path, tokens_done: int) -> Dict[str, Path]:
    stem = f"sae_step_{tokens_done}"
    return {
        "weights": checkpoint_dir / f"{stem}.safetensors",
        "steps": checkpoint_dir / f"{stem}.steps.json",
        "trainer": checkpoint_dir / f"{stem}.trainer.pt",
    }


def prune_local_checkpoints(checkpoint_dir: Path, keep: int) -> None:
    """Delete all but the `keep` newest local checkpoints (0 keeps all)."""
    if keep <= 0:
        return
    weights = sorted(checkpoint_dir.glob("sae_step_*.safetensors"), key=lambda p: int(p.stem.split("_")[-1]))
    for old in weights[:-keep]:
        for path in checkpoint_files(checkpoint_dir, int(old.stem.split("_")[-1])).values():
            if path.exists():
                path.unlink()


def configure_matmul_precision(config: Config) -> None:
    """TF32 tensor-core matmuls for the SAE's fp32 weights (Ampere+). Off
    by default in torch; a 32k x 3584 encoder/decoder pair at 4096 tokens
    per step is the run's largest matmul once the buffer is on the GPU."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(config.sae_tf32)
    torch.backends.cudnn.allow_tf32 = bool(config.sae_tf32)
    if config.sae_tf32:
        torch.set_float32_matmul_precision("high")


def idempotency_key_for(config: Config) -> str:
    if config.idempotency_key:
        return config.idempotency_key
    return "3c-" + sha256_of_payload(config.to_dict())[:24]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_boot(config: Config, workdir: Path, client: WorkerClient, model, tokenizer, device) -> Dict[str, Any]:
    path = workdir / "boot_checks.json"
    if path.exists():
        logger.info("[boot] already recorded, skipping")
        return json.loads(path.read_text(encoding="utf-8"))

    identity_prompt = "In one sentence, what is the capital of France?"
    identity_record = checks.identity_hook_check(model, tokenizer, identity_prompt, config.layer, device)
    if not identity_record["ok"]:
        raise AssertionError("identity hook check failed: hooked generation differs from unhooked generation")

    example_messages = [
        {"role": "user", "content": "Say hello in one short sentence."},
        {"role": "assistant", "content": "Hello! Hope you're having a good day."},
    ]
    mask_record = checks.assistant_mask_sanity_check(tokenizer, example_messages, max_seq=config.max_seq)

    out = {"identity": identity_record, "assistant_mask": mask_record}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    client.report(
        "boot",
        progress=_progress(2, 2),
        records=[
            {"recordId": "boot-identity-hook", "payload": identity_record},
            {"recordId": "boot-assistant-mask", "payload": mask_record},
        ],
    )
    return out


def _tokenize_conversation(tokenizer, messages: List[Dict[str, str]], config: Config) -> Optional[harvest.Prepared]:
    input_ids, mask = harvest.compute_assistant_mask(tokenizer, messages, max_seq=config.max_seq)
    if not any(mask):
        return None
    return input_ids, mask


def _pad_id(tokenizer) -> int:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    return int(pad_id)


def _prepare_batch(tokenizer, convs: List[List[Dict[str, str]]], config: Config, device):
    """Tokenise + collate a list of conversations onto `device`; None if no
    conversation had assistant tokens. Kept for the held-out pass and for
    callers outside the prefetched training loop."""
    prepared = [p for p in (_tokenize_conversation(tokenizer, m, config) for m in convs) if p is not None]
    if not prepared:
        return None
    batch_ids, batch_attn, batch_mask = harvest.collate_prepared(prepared, _pad_id(tokenizer))
    return batch_ids.to(device), batch_attn.to(device), batch_mask.to(device)


def _buffer_device(config: Config, device) -> torch.device:
    if config.shuffle_buffer_device == "cpu":
        return torch.device("cpu")
    if config.shuffle_buffer_device == "cuda":
        return torch.device("cuda")
    return torch.device(device) if torch.cuda.is_available() else torch.device("cpu")


def _read_steps_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compute_held_out(config: Config, harvester, tokenizer, trained_sae, d_model: int, device, buffer_dtype, buffer_device):
    """Held-out FVE/L0 on a fresh batch the optimizer never saw, drawn from
    a separate pass over the run's sources with its own seed. Returns
    (held_out_fve, held_out_l0), None/None if nothing could be harvested."""
    holdout_buffer = harvest.ShuffleBuffer(HELD_OUT_BATCH_SIZE, d_model, seed=config.seed + 1, dtype=buffer_dtype, device=buffer_device)
    holdout_stream = harvest.stream_config_conversations(config, seed=config.seed + 1_000_003)
    holdout_convo_batch: List[List[Dict[str, str]]] = []
    holdout_batches_tried = 0
    while holdout_buffer.filled < HELD_OUT_BATCH_SIZE and holdout_batches_tried < 500:
        try:
            _src, messages = next(holdout_stream)
        except StopIteration:
            break
        holdout_convo_batch.append(messages)
        if len(holdout_convo_batch) < config.harvest_batch_size:
            continue
        holdout_batch = _prepare_batch(tokenizer, holdout_convo_batch, config, device)
        holdout_convo_batch = []
        holdout_batches_tried += 1
        if holdout_batch is None:
            continue
        h_ids, h_attn, h_mask = holdout_batch
        h_acts = harvest.masked_activations(harvester, h_ids, h_attn, h_mask)
        if h_acts.numel() > 0:
            holdout_buffer.add(h_acts)
    if holdout_buffer.filled == 0:
        logger.warning("[train] collected zero held-out activations; held_out_fve/held_out_l0 will be null")
        return None, None
    x_holdout = holdout_buffer.sample(holdout_buffer.filled).to(device)
    with torch.no_grad():
        holdout_out = trained_sae.forward_loss(x_holdout, dead_window_tokens=config.dead_feature_window_tokens)
    held_out_fve = holdout_out.fraction_variance_explained.item()
    held_out_l0 = holdout_out.l0.item()
    logger.info("[train] held_out_fve=%.6f held_out_l0=%.4f (n=%d activations)", held_out_fve, held_out_l0, holdout_buffer.filled)
    return held_out_fve, held_out_l0


def measure_feature_stats(config: Config, harvester, tokenizer, trained_sae, device, tokens: int) -> sae_mod.FeatureStatsTracker:
    """Fresh pass of `tokens` assistant tokens through the frozen SAE to
    measure max activation and firing density (every token counts). Used
    by `--finalize-from-checkpoint`, where the training-time density (last
    20% of a *finished* run) is not available."""
    tracker = sae_mod.FeatureStatsTracker(config.sae_width, warmup_fraction=0.0, device=torch.device(device))
    stream = harvest.stream_config_conversations(config, seed=config.seed + 2_000_003)
    seen = 0
    convo_batch: List[List[Dict[str, str]]] = []
    for _src, messages in stream:
        if seen >= tokens:
            break
        convo_batch.append(messages)
        if len(convo_batch) < config.harvest_batch_size:
            continue
        batch = _prepare_batch(tokenizer, convo_batch, config, device)
        convo_batch = []
        if batch is None:
            continue
        ids, attn, mask = batch
        acts = harvest.masked_activations(harvester, ids, attn, mask)
        if acts.numel() == 0:
            continue
        with torch.no_grad():
            for start in range(0, acts.shape[0], config.batch_tokens):
                chunk = acts[start : start + config.batch_tokens].to(torch.float32)
                tracker.observe(trained_sae.encode(chunk), progress_fraction=1.0)
        seen += acts.shape[0]
    logger.info("[finalize] measured feature stats on %d assistant tokens", seen)
    return tracker


def stage_harvest_train(config: Config, workdir: Path, client: WorkerClient, model, tokenizer, device, finalize: bool = False):
    """Harvest + train. With `finalize=True` (`--finalize-from-checkpoint`)
    no training happens: the latest checkpoint (local, or downloaded from
    the HF repo for this run id) is promoted to the final SAE after a
    fresh feature-statistics pass, so the rest of the funnel can run on a
    run that died or was stopped mid-training."""
    sae_path = workdir / "sae.safetensors"
    stats_path = workdir / "feature_stats.json"
    if sae_path.exists() and stats_path.exists():
        logger.info("[train] trained SAE + feature stats already exist, skipping harvest+train")
        trained_sae = sae_mod.MatryoshkaBatchTopKSAE.load(
            sae_path, shells=config.matryoshka_shells, k=config.k, aux_loss_coef=config.aux_loss_coef
        ).to(device)
        feature_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        return trained_sae, feature_stats

    configure_matmul_precision(config)
    d_model = model.config.hidden_size
    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_files = sorted(
        checkpoint_dir.glob("sae_step_*.safetensors"), key=lambda p: int(p.stem.split("_")[-1])
    )
    if not resume_files:
        # Fresh pod after a host failure: pull the last checkpoint this run
        # uploaded, if any (no-op without HF_TOKEN / hf_upload_repo).
        if download_latest_checkpoint(config, client.run_id, checkpoint_dir) is not None:
            resume_files = sorted(
                checkpoint_dir.glob("sae_step_*.safetensors"), key=lambda p: int(p.stem.split("_")[-1])
            )

    if finalize and not resume_files:
        raise RuntimeError("--finalize-from-checkpoint: no checkpoint found locally or in the HF repo for this run")

    sources = harvest.dataset_sources(config)
    tokens_done = 0
    steps_done = 0
    resumes = 0
    convs_seen: List[int] = [0] * len(sources)
    tokens_by_source: List[int] = [0] * len(sources)
    trainer_state: Optional[Dict[str, Any]] = None
    if resume_files:
        latest = resume_files[-1]
        tokens_done = int(latest.stem.split("_")[-1])
        meta = _read_steps_meta(latest.parent / f"{latest.stem}.steps.json")
        steps_done = int(meta.get("steps_done", 0))
        resumes = int(meta.get("resumes", 0)) + 1
        seen = list(meta.get("convs_seen", []))
        convs_seen = [int(seen[i]) if i < len(seen) else 0 for i in range(len(sources))]
        seen_tokens = list(meta.get("tokens_by_source", []))
        tokens_by_source = [int(seen_tokens[i]) if i < len(seen_tokens) else 0 for i in range(len(sources))]
        trainer_path = latest.parent / f"{latest.stem}.trainer.pt"
        if trainer_path.exists():
            trainer_state = torch.load(trainer_path, map_location="cpu")
        logger.info(
            "[train] resuming from checkpoint at %d tokens (steps_done=%d, resume #%d, convs_seen=%s, trainer_state=%s)",
            tokens_done, steps_done, resumes, convs_seen, "yes" if trainer_state else "no",
        )
        trained_sae = sae_mod.MatryoshkaBatchTopKSAE.load(
            latest, shells=config.matryoshka_shells, k=config.k, aux_loss_coef=config.aux_loss_coef
        ).to(device)
    else:
        trained_sae = sae_mod.MatryoshkaBatchTopKSAE(
            d_in=d_model,
            width=config.sae_width,
            shells=config.matryoshka_shells,
            k=config.k,
            aux_loss_coef=config.aux_loss_coef,
            seed=config.seed,
        ).to(device)

    optimizer = torch.optim.Adam(trained_sae.parameters(), lr=config.lr)
    stats_tracker = sae_mod.FeatureStatsTracker(config.sae_width, device=torch.device(device))
    if trainer_state:
        # Adam moments, dead-window clocks and the feature statistics (max
        # activation sets every steering dose later) all continue from the
        # checkpoint instead of restarting from zero.
        try:
            optimizer.load_state_dict(trainer_state["optimizer"])
            with torch.no_grad():
                trained_sae.tokens_since_fired.copy_(trainer_state["tokens_since_fired"].to(device))
            stats_tracker.load_state_dict(trainer_state["stats"])
        except Exception:  # noqa: BLE001 - a stale trainer file must not block training
            logger.exception("[train] trainer state could not be restored; continuing with fresh optimizer/stats")

    buffer_dtype = DTYPES[{"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}[config.shuffle_buffer_dtype]]

    if finalize:
        logger.info("[finalize] promoting checkpoint at %d tokens to the final SAE (no further training)", tokens_done)
        client.report("train", progress=_progress(tokens_done, config.tokens_target), records=[{
            "recordId": f"train-finalized-{tokens_done}",
            "payload": {"tokens_done": tokens_done, "tokens_target": config.tokens_target, "steps_done": steps_done,
                        "finalized_from_checkpoint": True, "stats_pass_tokens": config.finalize_stats_tokens},
        }])
        with harvest.ActivationHarvester(model, config.layer) as harvester:
            if config.finalize_stats_tokens > 0:
                measured = measure_feature_stats(config, harvester, tokenizer, trained_sae, device, config.finalize_stats_tokens)
                # Max activation: the larger of what training saw and what the pass saw.
                measured.max_act = torch.maximum(measured.max_act, stats_tracker.max_act.to(measured.max_act.device))
                feature_stats = measured.to_dict()
            else:
                feature_stats = stats_tracker.to_dict()
            held_out_fve, held_out_l0 = compute_held_out(
                config, harvester, tokenizer, trained_sae, d_model, device, buffer_dtype, _buffer_device(config, device)
            )
        feature_stats["held_out_fve"] = held_out_fve
        feature_stats["held_out_l0"] = held_out_l0
        feature_stats["resumes"] = resumes
        feature_stats["convs_seen"] = convs_seen
        feature_stats["tokens_by_source"] = tokens_by_source
        feature_stats["sources"] = [src["name"] for src in sources]
        feature_stats["finalized_from_checkpoint_tokens"] = tokens_done
        feature_stats["tokens_target"] = config.tokens_target
        trained_sae.save(sae_path, feature_stats=feature_stats)
        upload_run_artifacts(config, workdir, client.run_id, sae_path, stats_path)
        return trained_sae, feature_stats

    buffer = harvest.ShuffleBuffer(
        config.shuffle_buffer_size, d_model, seed=config.seed + resumes, dtype=buffer_dtype, device=_buffer_device(config, device)
    )
    logger.info("[train] shuffle buffer %s on %s; sources=%s", buffer_dtype, buffer.device, [src["name"] for src in sources])
    last_out = None
    pad_id = _pad_id(tokenizer)

    def _save_checkpoint(tokens_now: int) -> None:
        files = checkpoint_files(checkpoint_dir, tokens_now)
        trained_sae.save(files["weights"])
        files["steps"].write_text(
            json.dumps({"steps_done": steps_done, "resumes": resumes, "convs_seen": convs_seen, "tokens_by_source": tokens_by_source, "tokens_done": tokens_now}, indent=2),
            encoding="utf-8",
        )
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "tokens_since_fired": trained_sae.tokens_since_fired.detach().cpu(),
                "stats": stats_tracker.state_dict(),
            },
            files["trainer"],
        )
        upload_checkpoint(config, client.run_id, [files["weights"], files["steps"], files["trainer"]])
        prune_local_checkpoints(checkpoint_dir, config.checkpoint_keep_local)

    with harvest.ActivationHarvester(model, config.layer) as harvester:
        # Resume continues past the conversations already harvested from
        # each source (per-source skip) instead of re-reading the same
        # shuffled prefix; the shuffle seed also moves with the resume count.
        conv_stream = harvest.stream_config_conversations(config, seed=config.seed + resumes, skips=convs_seen)
        prefetch = harvest.BatchPrefetcher(
            conv_stream,
            tokenize=lambda messages: _tokenize_conversation(tokenizer, messages, config),
            pad_id=pad_id,
            batch_size=config.harvest_batch_size,
            sort_group=config.harvest_sort_group,
            depth=config.harvest_prefetch_depth,
        )
        next_checkpoint = tokens_done + config.checkpoint_every_tokens
        stage_start_time = time.time()
        tokens_at_stage_start = tokens_done
        steps_at_stage_start = steps_done
        next_progress_log = tokens_done + PROGRESS_LOG_EVERY_TOKENS
        convs_total_seen = 0

        for batch, consumed, row_sources in prefetch:
            for src, n in consumed.items():
                convs_seen[src] += n
                convs_total_seen += n
            if batch is None:
                if convs_total_seen >= 64 and tokens_done == 0:
                    raise RuntimeError("harvest produced zero assistant tokens after 64 conversations; assistant mask or dataset shape is wrong")
                continue
            # Per-source assistant-token counts come from the CPU mask before
            # the copy: weights are per conversation, and sources differ a
            # lot in tokens per conversation, so the real token shares are
            # logged and recorded rather than assumed.
            row_tokens = (batch[1] & batch[2]).sum(dim=1).tolist()
            for src, n in zip(row_sources, row_tokens):
                tokens_by_source[src] += int(n)
            batch_ids, batch_attn, batch_mask = (t.to(device, non_blocking=True) for t in batch)
            acts = harvest.masked_activations(harvester, batch_ids, batch_attn, batch_mask)
            if acts.numel() > 0:
                buffer.add(acts)
                tokens_done += acts.shape[0]

            if tokens_done >= next_progress_log:
                elapsed_s = time.time() - stage_start_time
                throughput = compute_throughput_and_eta(
                    tokens_done - tokens_at_stage_start,
                    steps_done - steps_at_stage_start,
                    elapsed_s,
                    config.tokens_target - tokens_done,
                )
                logger.info(
                    "[train] progress tokens=%d steps=%d tokens_per_s=%.2f steps_per_s=%.2f elapsed_s=%.1f eta_s=%s tokens_by_source=%s",
                    tokens_done,
                    steps_done,
                    throughput["tokens_per_s"],
                    throughput["steps_per_s"],
                    elapsed_s,
                    throughput["eta_s"],
                    tokens_by_source,
                )
                client.report("train", progress=_progress(tokens_done, config.tokens_target))
                next_progress_log = tokens_done + PROGRESS_LOG_EVERY_TOKENS

            if not buffer.is_ready(min_fraction=0.05):
                continue

            steps_done, last_out = train_steps_on_buffer(
                trained_sae, optimizer, buffer, config, steps_done, device=device
            )
            stats_tracker.observe(last_out.codes, progress_fraction=min(1.0, tokens_done / config.tokens_target))

            if tokens_done >= next_checkpoint or tokens_done >= config.tokens_target:
                _save_checkpoint(tokens_done)
                fve = last_out.fraction_variance_explained.item()
                l0 = last_out.l0.item()
                dead_frac = last_out.dead_fraction.item()
                logger.info(
                    "[train] tokens=%d steps=%d loss=%.6f fve=%.6f l0=%.4f dead_frac=%.6f",
                    tokens_done, steps_done, last_out.loss.item(), fve, l0, dead_frac,
                )
                client.report(
                    "train",
                    progress=_progress(tokens_done, config.tokens_target),
                    records=[
                        {
                            "recordId": f"train-checkpoint-{tokens_done}",
                            "payload": {
                                "tokens_done": tokens_done,
                                "tokens_target": config.tokens_target,
                                "steps_done": steps_done,
                                "loss": last_out.loss.item(),
                                "recon_loss": last_out.recon_loss.item(),
                                "aux_loss": last_out.aux_loss.item(),
                                "fraction_variance_explained": fve,
                                "l0": l0,
                                "dead_fraction": dead_frac,
                            },
                        }
                    ],
                )
                next_checkpoint = tokens_done + config.checkpoint_every_tokens

            if tokens_done >= config.tokens_target:
                break
        prefetch.close()

        held_out_fve, held_out_l0 = compute_held_out(
            config, harvester, tokenizer, trained_sae, d_model, device, buffer_dtype, buffer.device
        )

    feature_stats = stats_tracker.to_dict()
    feature_stats["held_out_fve"] = held_out_fve
    feature_stats["held_out_l0"] = held_out_l0
    feature_stats["resumes"] = resumes
    feature_stats["convs_seen"] = convs_seen
    feature_stats["tokens_by_source"] = tokens_by_source
    feature_stats["sources"] = [src["name"] for src in sources]
    trained_sae.save(sae_path, feature_stats=feature_stats)
    upload_run_artifacts(config, workdir, client.run_id, sae_path, stats_path)
    return trained_sae, feature_stats


def stage_post_train_check(
    config: Config, workdir: Path, client: WorkerClient, model, tokenizer, trained_sae, feature_stats, device
):
    path = workdir / "post_train_check.json"
    if path.exists():
        logger.info("[train] post-train check already recorded, skipping")
        return json.loads(path.read_text(encoding="utf-8"))

    record = checks.sae_replace_check(model, tokenizer, trained_sae, POST_TRAIN_CHECK_PROMPT, config.layer, device)
    record["held_out_fve"] = feature_stats.get("held_out_fve")
    record["held_out_l0"] = feature_stats.get("held_out_l0")
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    client.report("train", records=[{"recordId": "post-train-sae-replace", "payload": record}])
    if not record["ok"]:
        logger.warning("SAE reconstruct-and-replace argmax match below threshold: %s", record)
    return record


def stage_rank(
    config: Config,
    workdir: Path,
    client: WorkerClient,
    model,
    tokenizer,
    trained_sae,
    feature_stats,
    scenarios,
    device,
):
    """Label-free candidate ranking (see rank.py): persona-context
    activation shift, run after the post-train check and before calibrate
    so calibrate/screen can cover the most promising `config.screen_features`
    candidates instead of a pure density-quantile spread. Reports one
    record per selected candidate to stage `"train"` with recordId prefix
    `"rank-"` (see `rank.build_rank_records` for why `"train"` rather than a
    dedicated stage)."""
    path = workdir / "candidates.json"
    if path.exists():
        logger.info("[rank] candidates already exist, skipping rank stage")
        return json.loads(path.read_text(encoding="utf-8"))

    candidates = rank.run_rank(config, model, tokenizer, trained_sae, feature_stats, scenarios, device, seed=config.seed)
    path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    records = rank.build_rank_records(candidates)
    client.report("train", progress=_progress(len(records), len(records)), records=records)
    return candidates


def stage_steer(
    config: Config,
    workdir: Path,
    client: WorkerClient,
    model,
    tokenizer,
    trained_sae,
    feature_stats,
    scenarios,
    device,
    candidates=None,
):
    path = workdir / "calibration_records.json"
    if path.exists():
        logger.info("[calibrate] records already exist, skipping steer stage")
        return json.loads(path.read_text(encoding="utf-8"))

    explicit_features = (candidates or {}).get("screen_features") or None
    records = steer.run_calibration(
        config, model, tokenizer, trained_sae, feature_stats, scenarios, device,
        seed=config.seed, explicit_features=explicit_features,
    )
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    client.report("calibrate", progress=_progress(len(records), len(records)), records=records)
    return records


def stage_screen(
    config: Config,
    workdir: Path,
    client: WorkerClient,
    model,
    tokenizer,
    trained_sae,
    feature_stats,
    calibration_records,
    calibrate_scenarios,
    device,
):
    """Separability + consistency vs. random-direction nulls (see
    screen.py). Runs its own scenario set (`config.screen_scenarios`,
    reloaded from `config.scenarios_file` rather than reusing the smaller
    `calibrate_scenarios` slice `stage_steer` used) and its own fresh
    random-direction draws, so it can score every direction on more
    scenarios than the calibrate stage swept. Returns
    `(direction_records, generation_records)`; the latter carries the raw
    steered/baseline text behind every direction's row (see
    `screen.build_generation_records`)."""
    path = workdir / "screen_records.json"
    gen_path = workdir / "screen_generations.json"
    if path.exists() and gen_path.exists():
        logger.info("[screen] records already exist, skipping screen stage")
        return json.loads(path.read_text(encoding="utf-8")), json.loads(gen_path.read_text(encoding="utf-8"))

    all_scenarios = load_scenarios(Path(config.scenarios_file))
    screen_scenarios = all_scenarios[: config.screen_scenarios]

    result = screen.run_screen(
        config,
        model,
        tokenizer,
        trained_sae,
        feature_stats,
        calibration_records,
        calibrate_scenarios,
        screen_scenarios,
        device,
        seed=config.seed,
    )
    records = result["directions"]
    generation_records = result["generations"]
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    gen_path.write_text(json.dumps(generation_records, indent=2), encoding="utf-8")
    client.report(
        "screen",
        progress=_progress(len(records), len(records)),
        records=records + generation_records,
    )
    return records, generation_records


def stage_describe(
    config: Config,
    workdir: Path,
    client: WorkerClient,
    screen_records: List[Dict[str, Any]],
    screen_generation_records: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Describe stage (judge pass one, see describe.py): blinded pairs to
    the Worker judge for the top `config.describe_top_n` directions by
    residual separability AUC, clustered per direction. Disabled when
    `describe_top_n <= 0`. Resumable: `describe.run_describe` skips if
    `describe_results.json` already exists."""
    if config.describe_top_n <= 0:
        logger.info("[describe] describe_top_n<=0, skipping describe stage")
        return None
    directions = [rec["payload"] for rec in screen_records]
    generations = [rec["payload"] for rec in screen_generation_records]
    return describe.run_describe(config, workdir, client, client.run_id, directions, generations)


def stage_reach(
    config: Config,
    workdir: Path,
    client: WorkerClient,
    model,
    tokenizer,
    trained_sae,
    device,
    describe_output: Optional[Dict[str, Any]],
    screen_records: List[Dict[str, Any]],
    screen_generation_records: List[Dict[str, Any]],
    scenario_prompt_by_id: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Reach stage (measured outcome, not a gate -- see reach.py and README
    "Reach stage"): can prompting reproduce a named direction's steered
    effect, behaviorally and mechanistically? Runs after describe.
    Skipped entirely if describe produced nothing (disabled, or a resumed
    run that never populated it) or if `config.reach_max_directions <= 0`.
    Resumable: `reach.run_reach` skips if `reach_results.json` already
    exists."""
    if not describe_output:
        logger.info("[reach] describe stage produced no output, skipping reach stage")
        return None
    if config.reach_max_directions <= 0:
        logger.info("[reach] reach_max_directions<=0, skipping reach stage")
        return None
    directions = [rec["payload"] for rec in screen_records]
    generations = [rec["payload"] for rec in screen_generation_records]
    return reach.run_reach(
        config,
        workdir,
        client,
        client.run_id,
        model,
        tokenizer,
        trained_sae,
        device,
        describe_output,
        directions,
        generations,
        scenario_prompt_by_id,
    )


# ---------------------------------------------------------------------------
# Analysis (embedding-free proxy metrics + report page)
# ---------------------------------------------------------------------------
def word_edit_distance(a: str, b: str) -> int:
    wa, wb = a.split(), b.split()
    n, m = len(wa), len(wb)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if wa[i - 1] == wb[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def normalized_edit_distance(a: str, b: str) -> float:
    wa, wb = a.split(), b.split()
    denom = max(len(wa), len(wb), 1)
    return word_edit_distance(a, b) / denom


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def passes_gate_g0(row: dict, max_random_auc: Optional[float]) -> bool:
    """One direction/sign's pass/fail against gate G0: resid-view
    separability AUC beats the best random-direction null's AUC *and*
    consistency clears the null consistency by more than 0.1."""
    if max_random_auc is None:
        return False
    auc = row.get("separability", {}).get("resid", {}).get("auc")
    consistency = row.get("consistency") or {}
    mean_cos = consistency.get("mean_cos")
    null_mean_cos = consistency.get("null_mean_cos")
    if not (_is_finite_number(auc) and _is_finite_number(mean_cos) and _is_finite_number(null_mean_cos)):
        return False
    return auc > max_random_auc and (mean_cos - null_mean_cos) > 0.1


def _group_pass_counts(rows: List[dict], max_random_auc: Optional[float]) -> Dict[str, Any]:
    """Per-direction (sign-level) and per-entity (either-sign) pass counts
    for a kind ("feature" or "control"), plus a per-entity detail list.
    An entity (a feature index or a control name) passes if *either* of its
    signs passes gate G0 -- visual run 1 counted controls per sign, which
    undercounted a control whose only weak sign was consistency, not AUC
    (see ../VISUAL-1.md "Screen")."""
    directions_passing = sum(1 for r in rows if passes_gate_g0(r, max_random_auc))

    ids = sorted({r.get("id") for r in rows}, key=lambda v: (str(type(v)), v))
    detail = []
    entities_passing = 0
    for entity_id in ids:
        entity_rows = [r for r in rows if r.get("id") == entity_id]
        pos_row = next((r for r in entity_rows if r.get("sign") == 1), None)
        neg_row = next((r for r in entity_rows if r.get("sign") == -1), None)
        pos_passes = passes_gate_g0(pos_row, max_random_auc) if pos_row else False
        neg_passes = passes_gate_g0(neg_row, max_random_auc) if neg_row else False
        either_passes = pos_passes or neg_passes
        if either_passes:
            entities_passing += 1
        detail.append({"id": entity_id, "pos_passes": pos_passes, "neg_passes": neg_passes, "passes": either_passes})

    return {
        "directions_passing": directions_passing,
        "directions_total": len(rows),
        "entities_passing": entities_passing,
        "entities_total": len(ids),
        "detail": detail,
    }


def compute_screen_verdict(screen_rows: List[dict]) -> dict:
    """Preregistered gate G0 (see README "Screen stage"): diagnostic only,
    does not gate anything in this smoke pipeline. A direction "passes" if
    its resid-view separability AUC beats the best random-direction null's
    AUC *and* its consistency clears the null consistency by more than
    0.1. A feature or control (an "entity") passes if *either* of its two
    signs passes -- visual run 1 counted controls per sign instead of per
    control, which undercounted controls whose positive sign passed but
    whose negative sign only missed on the consistency margin (see
    ../VISUAL-1.md "Screen": evil/benevolent failed only 0.10 vs 0.06 on
    that basis). Reports both the per-sign ("direction") count and the
    per-entity count for features, and the per-entity count (out of the
    number of distinct controls) for controls, with per-sign detail kept
    in `controls_detail`."""
    randoms = [r for r in screen_rows if r.get("kind") == "random"]
    features = [r for r in screen_rows if r.get("kind") == "feature"]
    controls = [r for r in screen_rows if r.get("kind") == "control"]

    random_aucs = [
        r["separability"]["resid"]["auc"]
        for r in randoms
        if _is_finite_number(r.get("separability", {}).get("resid", {}).get("auc"))
    ]
    max_random_auc = max(random_aucs) if random_aucs else None
    max_random_auc_p95 = float(np.percentile(random_aucs, 95)) if random_aucs else None

    feature_counts = _group_pass_counts(features, max_random_auc)
    control_counts = _group_pass_counts(controls, max_random_auc)

    # Per-arm (either sign) pass counts among features -- see rank.py's
    # three-arm selection ("unsupervised"/"quantile"/"shift"). A row here
    # predating arm tagging (no "arm" key) simply matches none of the
    # three and is silently excluded, same as it would be from any of
    # these counts before this field existed.
    arm_pass_counts = {
        arm: _group_pass_counts([r for r in features if r.get("arm") == arm], max_random_auc)
        for arm in ("unsupervised", "quantile", "shift")
    }

    return {
        "max_random_resid_auc": max_random_auc,
        "max_random_resid_auc_p95": max_random_auc_p95,
        "random_directions": len(randoms),
        # per-sign ("direction") counts, i.e. what visual run 1 reported
        "feature_directions_passing": feature_counts["directions_passing"],
        "feature_directions_total": feature_counts["directions_total"],
        "control_directions_passing": control_counts["directions_passing"],
        "control_directions_total": control_counts["directions_total"],
        # per-entity (either-sign) counts
        "features_passing": feature_counts["entities_passing"],
        "features_total": feature_counts["entities_total"],
        "features_detail": feature_counts["detail"],
        "controls_passing": control_counts["entities_passing"],
        "controls_total": control_counts["entities_total"],
        "controls_detail": control_counts["detail"],
        "all_controls_pass": bool(control_counts["entities_total"])
        and control_counts["entities_passing"] == control_counts["entities_total"],
        "arm_pass_counts": {
            arm: {
                "entities_passing": counts["entities_passing"],
                "entities_total": counts["entities_total"],
            }
            for arm, counts in arm_pass_counts.items()
        },
    }


def build_summary(
    config: Config,
    boot: dict,
    post_train: dict,
    calibration_records: List[dict],
    screen_records: Optional[List[dict]] = None,
    screen_generation_records: Optional[List[dict]] = None,
) -> dict:
    baseline_text: Dict[str, str] = {}
    baseline_len: Dict[str, int] = {}
    for rec in calibration_records:
        payload = rec["payload"]
        if payload.get("kind") == "baseline":
            baseline_text[payload["scenario"]] = payload["text"]
            baseline_len[payload["scenario"]] = payload["num_tokens"]

    features: Dict[int, dict] = {}
    random_rows: List[dict] = []

    for rec in calibration_records:
        payload = rec["payload"]
        scenario = payload.get("scenario")
        base_text = baseline_text.get(scenario, "")
        base_len = baseline_len.get(scenario, 0)
        coherence = payload.get("coherence") or {}

        if payload.get("kind") == "steered":
            f_idx = payload["feature"]
            entry = features.setdefault(
                f_idx,
                {
                    "feature": f_idx,
                    "density": payload.get("density"),
                    "quantile": payload.get("quantile"),
                    "max_activation": payload.get("max_activation"),
                    "rows": [],
                },
            )
            entry["rows"].append(
                {
                    "sign": payload["sign"],
                    "dose": payload["dose"],
                    "scenario": scenario,
                    "text": payload["text"],
                    "baseline_text": base_text,
                    "edit_distance": normalized_edit_distance(payload["text"], base_text),
                    "length_delta": payload["num_tokens"] - base_len,
                    "coherence": coherence,
                    "coherent": payload.get("coherent"),
                    "finish_reason": payload.get("finish_reason"),
                }
            )
        elif payload.get("kind") == "random_control":
            random_rows.append(
                {
                    "random_index": payload["random_index"],
                    "dose": payload["dose"],
                    "scenario": scenario,
                    "text": payload["text"],
                    "baseline_text": base_text,
                    "edit_distance": normalized_edit_distance(payload["text"], base_text),
                    "length_delta": payload["num_tokens"] - base_len,
                    "coherence": coherence,
                    "coherent": payload.get("coherent"),
                    "finish_reason": payload.get("finish_reason"),
                }
            )

    for entry in features.values():
        entry["max_coherent_dose"] = {
            "pos": steer.max_coherent_dose([r for r in entry["rows"] if r["sign"] == 1]),
            "neg": steer.max_coherent_dose([r for r in entry["rows"] if r["sign"] == -1]),
        }

    random_by_index: Dict[int, List[dict]] = {}
    for row in random_rows:
        random_by_index.setdefault(row["random_index"], []).append(row)
    random_max_coherent_dose = {
        str(idx): steer.max_coherent_dose(rows) for idx, rows in sorted(random_by_index.items())
    }

    def _resid_auc_sort_key(row: dict) -> float:
        auc = row.get("separability", {}).get("resid", {}).get("auc")
        return auc if _is_finite_number(auc) else float("-inf")

    screen_rows = sorted(
        (rec["payload"] for rec in (screen_records or [])),
        key=_resid_auc_sort_key,
        reverse=True,
    )
    screen_verdict = compute_screen_verdict(screen_rows)
    screen_generations = [rec["payload"] for rec in (screen_generation_records or [])]

    return {
        "config": config.to_dict(),
        "boot_checks": boot,
        "post_train_check": post_train,
        "baseline": baseline_text,
        "features": sorted(features.values(), key=lambda e: e["feature"]),
        "random_control": random_rows,
        "random_control_max_coherent_dose": random_max_coherent_dose,
        "screen": screen_rows,
        "screen_verdict": screen_verdict,
        "screen_generations": screen_generations,
    }


def render_html_report(summary: dict) -> str:
    def esc(s: Any) -> str:
        return html.escape("" if s is None else str(s))

    def fmt(v: Any, digits: int = 4) -> str:
        return "" if v is None else esc(round(v, digits) if isinstance(v, float) else v)

    def coherence_cells(coherence: dict) -> str:
        return (
            f"<td>{fmt(coherence.get('distinct_ratio'))}</td>"
            f"<td>{fmt(coherence.get('max_run'), 0)}</td>"
            f"<td>{fmt(coherence.get('repeat_4gram'))}</td>"
            f"<td>{fmt(coherence.get('logprob'))}</td>"
        )

    feature_summary_rows = []
    rows_html = []
    for feat in summary["features"]:
        max_coherent = feat.get("max_coherent_dose") or {}
        feature_summary_rows.append(
            f"""
            <tr>
              <td>{esc(feat['feature'])}</td>
              <td>{fmt(feat.get('density'), 6)}</td>
              <td>{fmt(feat.get('quantile'), 2)}</td>
              <td>{fmt(max_coherent.get('pos'))}</td>
              <td>{fmt(max_coherent.get('neg'))}</td>
            </tr>"""
        )
        for row in feat["rows"]:
            row_class = "" if row.get("coherent") else " class=\"incoherent\""
            rows_html.append(
                f"""
                <tr{row_class}>
                  <td>{esc(feat['feature'])}</td>
                  <td>{fmt(feat.get('density'), 6)}</td>
                  <td>{fmt(feat.get('quantile'), 2)}</td>
                  <td>{'+' if row['sign'] > 0 else '-'}</td>
                  <td>{esc(row['dose'])}</td>
                  <td>{esc(row['scenario'])}</td>
                  {coherence_cells(row['coherence'])}
                  <td>{'yes' if row.get('coherent') else 'NO'}</td>
                  <td>{fmt(row['edit_distance'])}</td>
                  <td>{esc(row['length_delta'])}</td>
                  <td><pre>{esc(row['baseline_text'])}</pre></td>
                  <td><pre>{esc(row['text'])}</pre></td>
                </tr>"""
            )

    random_rows_html = []
    random_max_coherent = summary.get("random_control_max_coherent_dose") or {}
    for row in summary["random_control"]:
        row_class = "" if row.get("coherent") else " class=\"incoherent\""
        random_rows_html.append(
            f"""
            <tr{row_class}>
              <td>random-{esc(row['random_index'])}</td>
              <td>{fmt(random_max_coherent.get(str(row['random_index'])))}</td>
              <td>{esc(row['dose'])}</td>
              <td>{esc(row['scenario'])}</td>
              {coherence_cells(row['coherence'])}
              <td>{'yes' if row.get('coherent') else 'NO'}</td>
              <td>{fmt(row['edit_distance'])}</td>
              <td>{esc(row['length_delta'])}</td>
              <td><pre>{esc(row['baseline_text'])}</pre></td>
              <td><pre>{esc(row['text'])}</pre></td>
            </tr>"""
        )

    screen_generations = summary.get("screen_generations") or []

    def _generation_examples(row: dict, n: int = 2) -> List[dict]:
        matches = [
            g
            for g in screen_generations
            if g.get("kind") == row.get("kind") and g.get("id") == row.get("id") and g.get("sign") == row.get("sign")
        ]
        matches.sort(key=lambda g: str(g.get("scenario")))
        return matches[:n]

    screen_rows_html = []
    screen_generation_blocks_html = []
    for row in summary.get("screen") or []:
        kind = row.get("kind")
        sign = row.get("sign")
        sign_label = {1: "+", -1: "-", 0: "n/a"}.get(sign, esc(sign))
        resid = (row.get("separability") or {}).get("resid") or {}
        lexical = (row.get("separability") or {}).get("lexical") or {}
        consistency = row.get("consistency") or {}
        screen_rows_html.append(
            f"""
            <tr class="screen-{esc(kind)}">
              <td>{esc(kind)}</td>
              <td>{esc(row.get('id'))}</td>
              <td>{sign_label}</td>
              <td>{esc(row.get('arm'))}</td>
              <td>{esc(row.get('dose'))}</td>
              <td>{fmt(resid.get('acc'))}</td>
              <td>{fmt(resid.get('auc'))}</td>
              <td>{fmt(lexical.get('acc'))}</td>
              <td>{fmt(lexical.get('auc'))}</td>
              <td>{fmt(consistency.get('mean_cos'))}</td>
              <td>{fmt(consistency.get('null_mean_cos'))}</td>
              <td>{fmt(row.get('coherent_fraction'))}</td>
            </tr>"""
        )

        example_pairs_html = "".join(
            f"""
              <div class="gen-pair">
                <p><strong>scenario:</strong> {esc(ex.get('scenario'))}
                   &nbsp; <strong>finish_reason:</strong> {esc(ex.get('finish_reason'))}
                   &nbsp; <strong>coherent:</strong> {'yes' if (ex.get('coherence') or {}).get('coherent') else 'NO'}</p>
                <div class="gen-cols">
                  <div><em>baseline</em><pre>{esc(ex.get('baseline_text'))}</pre></div>
                  <div><em>steered</em><pre>{esc(ex.get('text'))}</pre></div>
                </div>
              </div>"""
            for ex in _generation_examples(row)
        )
        screen_generation_blocks_html.append(
            f"""
            <details class="screen-detail">
              <summary>{esc(kind)} {esc(row.get('id'))} {sign_label} (dose {esc(row.get('dose'))})</summary>
              {example_pairs_html or '<p>No generations recorded.</p>'}
            </details>"""
        )

    verdict = summary.get("screen_verdict") or {}
    verdict_line = (
        f"{esc(verdict.get('features_passing'))} of {esc(verdict.get('features_total'))} features "
        f"(either sign; {esc(verdict.get('feature_directions_passing'))} of "
        f"{esc(verdict.get('feature_directions_total'))} individual feature/sign directions) beat the "
        f"max random resid-AUC ({fmt(verdict.get('max_random_resid_auc'))}, p95 "
        f"{fmt(verdict.get('max_random_resid_auc_p95'))} over {esc(verdict.get('random_directions'))} random "
        f"nulls) and exceed the null consistency by &gt;0.1 (gate G0). Controls clearing the same bar "
        f"(either sign): {esc(verdict.get('controls_passing'))} of {esc(verdict.get('controls_total'))} "
        f"({'all' if verdict.get('all_controls_pass') else 'not all'} controls pass)."
    )
    arm_pass_counts = verdict.get("arm_pass_counts") or {}
    arm_pass_line = "Per-arm feature pass counts (either sign, gate G0): " + ", ".join(
        f"{esc(arm)} {esc(arm_pass_counts.get(arm, {}).get('entities_passing'))} of "
        f"{esc(arm_pass_counts.get(arm, {}).get('entities_total'))}"
        for arm in ("unsupervised", "quantile", "shift")
    ) + ". The \"shift\" arm is the positive control for this comparison (it is biased toward prompt-reachable directions); \"unsupervised\" never saw a prompt."

    def _detail_rows(detail: List[dict]) -> str:
        return "".join(
            f"<tr><td>{esc(d.get('id'))}</td><td>{'yes' if d.get('pos_passes') else 'no'}</td>"
            f"<td>{'yes' if d.get('neg_passes') else 'no'}</td><td>{'yes' if d.get('passes') else 'no'}</td></tr>"
            for d in detail
        )

    controls_detail_html = _detail_rows(verdict.get("controls_detail") or [])
    features_detail_html = _detail_rows(verdict.get("features_detail") or [])

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Experiment 3C smoke report</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #111; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 6px; vertical-align: top; text-align: left; }}
  th {{ background: #eee; position: sticky; top: 0; }}
  tr.incoherent {{ background: #fdecea; }}
  tr.screen-feature {{ background: #eef4ff; }}
  tr.screen-control {{ background: #eafbea; }}
  tr.screen-random {{ background: #f4f4f4; }}
  pre {{ white-space: pre-wrap; margin: 0; max-width: 32rem; }}
  h1, h2 {{ font-weight: 600; }}
  details.screen-detail {{ border: 1px solid #ccc; border-radius: 4px; margin-bottom: 0.5rem; padding: 0.4rem 0.6rem; }}
  details.screen-detail summary {{ cursor: pointer; font-weight: 600; }}
  .gen-pair {{ margin: 0.5rem 0; padding-top: 0.5rem; border-top: 1px solid #eee; }}
  .gen-cols {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
  .gen-cols > div {{ flex: 1 1 20rem; min-width: 16rem; }}
</style>
</head>
<body>
<h1>Experiment 3C -- smoke report</h1>
<h2>Boot checks</h2>
<pre>{esc(json.dumps(summary['boot_checks'], indent=2))}</pre>
<h2>Post-train check</h2>
<pre>{esc(json.dumps(summary['post_train_check'], indent=2))}</pre>
<h2>Feature selection summary</h2>
<table>
<thead><tr>
  <th>feature</th><th>density</th><th>quantile</th>
  <th>max coherent dose (+)</th><th>max coherent dose (-)</th>
</tr></thead>
<tbody>{''.join(feature_summary_rows)}</tbody>
</table>
<h2>Steering dose sweep</h2>
<p>Rows with <code>class="incoherent"</code> (highlighted) failed
<code>coherent</code> = distinct_ratio &gt;= 0.5 and max_run &lt;= 4 and
repeat_4gram &lt;= 0.2.</p>
<table>
<thead><tr>
  <th>feature</th><th>density</th><th>quantile</th><th>sign</th><th>dose</th><th>scenario</th>
  <th>distinct ratio</th><th>max run</th><th>repeat 4gram</th><th>logprob</th><th>coherent</th>
  <th>norm. edit dist.</th><th>length delta</th>
  <th>baseline</th><th>steered</th>
</tr></thead>
<tbody>{''.join(rows_html)}</tbody>
</table>
<h2>Random-direction control</h2>
<table>
<thead><tr>
  <th>control</th><th>max coherent dose</th><th>dose</th><th>scenario</th>
  <th>distinct ratio</th><th>max run</th><th>repeat 4gram</th><th>logprob</th><th>coherent</th>
  <th>norm. edit dist.</th><th>length delta</th>
  <th>baseline</th><th>steered</th>
</tr></thead>
<tbody>{''.join(random_rows_html)}</tbody>
</table>
<h2>Screen: separability and consistency vs. random-direction nulls</h2>
<p>One row per direction (feature/sign, positive control/sign, or random
null), sorted by resid-view separability AUC descending. Feature rows are
blue, control rows green, random-null rows grey. "consistency" /
"null consistency" are the mean cosine similarity of this direction's
per-scenario (steered - baseline) difference vectors with each other vs.
with other directions' difference vectors -- see README "Screen stage".</p>
<p><strong>Verdict (gate G0, diagnostic only):</strong> {verdict_line}</p>
<p><strong>{arm_pass_line}</strong></p>
<table>
<thead><tr>
  <th>kind</th><th>id</th><th>sign</th><th>arm</th><th>dose</th>
  <th>resid acc</th><th>resid auc</th><th>lexical acc</th><th>lexical auc</th>
  <th>consistency</th><th>null consistency</th><th>coherent fraction</th>
</tr></thead>
<tbody>{''.join(screen_rows_html)}</tbody>
</table>
<h3>Per-control pass detail (either sign)</h3>
<table>
<thead><tr><th>control</th><th>pos passes</th><th>neg passes</th><th>control passes</th></tr></thead>
<tbody>{controls_detail_html}</tbody>
</table>
<h3>Per-feature pass detail (either sign)</h3>
<table>
<thead><tr><th>feature</th><th>pos passes</th><th>neg passes</th><th>feature passes</th></tr></thead>
<tbody>{features_detail_html}</tbody>
</table>
<h3>Screen generations: example baseline/steered pairs per direction</h3>
<p>Up to two example scenario pairs per direction (feature/sign, control/sign,
or random null), in the same order as the screen table above. Full per-
scenario generations are in <code>screen_generations.json</code> /
<code>summary.json</code>'s <code>screen_generations</code> field.</p>
{''.join(screen_generation_blocks_html)}
</body>
</html>
"""


def stage_analysis(
    config: Config,
    workdir: Path,
    client: WorkerClient,
    boot,
    post_train,
    calibration_records,
    screen_records=None,
    screen_generation_records=None,
):
    summary_path = workdir / "summary.json"
    report_path = workdir / "smoke-report.html"
    if summary_path.exists() and report_path.exists():
        logger.info("[analyze] summary + report already exist, skipping")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    summary = build_summary(
        config, boot, post_train, calibration_records, screen_records, screen_generation_records
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(render_html_report(summary), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment 3C SAE pipeline")
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument(
        "--finalize-from-checkpoint", action="store_true",
        help="Skip further training: promote the latest checkpoint (local or from the HF repo for this run id) "
             "to the final SAE after a feature-statistics pass, then run the rest of the funnel.",
    )
    args = parser.parse_args(argv)

    config = Config.from_json(args.config)
    set_seed(config.seed)

    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    client = WorkerClient()
    if not client.run_id:
        client.start_run(config.manifest_hash, config.budget_usd, idempotency_key_for(config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("no CUDA device found; this will be extremely slow for a 7B model")

    try:
        model, tokenizer = load_model_and_tokenizer(config, device)

        boot = stage_boot(config, workdir, client, model, tokenizer, device)
        trained_sae, feature_stats = stage_harvest_train(
            config, workdir, client, model, tokenizer, device, finalize=args.finalize_from_checkpoint
        )
        post_train = stage_post_train_check(
            config, workdir, client, model, tokenizer, trained_sae, feature_stats, device
        )
        upload_stage_outputs(config, client.run_id, workdir, ["boot_checks.json", "post_train_check.json"])

        scenarios = load_scenarios(Path(config.scenarios_file))
        scenarios = scenarios[: config.steer_scenarios]

        candidates = stage_rank(
            config, workdir, client, model, tokenizer, trained_sae, feature_stats, scenarios, device
        )
        upload_stage_outputs(config, client.run_id, workdir, ["candidates.json"])

        calibration_records = stage_steer(
            config, workdir, client, model, tokenizer, trained_sae, feature_stats, scenarios, device,
            candidates=candidates,
        )
        upload_stage_outputs(config, client.run_id, workdir, ["calibration_records.json"])

        screen_records, screen_generation_records = stage_screen(
            config, workdir, client, model, tokenizer, trained_sae, feature_stats,
            calibration_records, scenarios, device,
        )

        upload_stage_outputs(config, client.run_id, workdir, ["screen_records.json", "screen_generations.json"])

        describe_output = stage_describe(config, workdir, client, screen_records, screen_generation_records)
        upload_stage_outputs(config, client.run_id, workdir, ["describe_results.json", "describe-report.html"])

        reach_screen_scenarios = load_scenarios(Path(config.scenarios_file))[: config.screen_scenarios]
        scenario_prompt_by_id = {s["id"]: s["prompt"] for s in reach_screen_scenarios}
        stage_reach(
            config, workdir, client, model, tokenizer, trained_sae, device,
            describe_output, screen_records, screen_generation_records, scenario_prompt_by_id,
        )

        upload_stage_outputs(config, client.run_id, workdir, ["reach_results.json", "reach-report.html"])

        summary = stage_analysis(
            config, workdir, client, boot, post_train, calibration_records,
            screen_records, screen_generation_records,
        )
        upload_stage_outputs(config, client.run_id, workdir, ["summary.json", "smoke-report.html", "run_config.json"])

        client.report(
            "done",
            status="complete",
            message="smoke run complete",
            progress=_progress(1, 1),
        )
        logger.info("done. summary at %s", workdir / "summary.json")
        return 0
    except Exception as exc:  # noqa: BLE001 - must report failure before re-raising
        logger.exception("run_smoke failed")
        client.report("done", status="failed", progress=_progress(0, 1), message=f"{type(exc).__name__}: {exc}"[:500])
        raise


if __name__ == "__main__":
    sys.exit(main())
