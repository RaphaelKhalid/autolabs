"""Experiment 3C orchestrator: boot checks -> harvest+train -> post-train
check -> steer (dose-sweep calibration) -> analysis -> done.

Every stage is resumable: before doing expensive work, each stage checks for
its own output file under `config.workdir` and, if present, loads it instead
of recomputing. Training additionally checkpoints the SAE every
`checkpoint_every_tokens` tokens under `<workdir>/checkpoints/` so a crash
mid-run only loses partial progress since the last checkpoint (the token
stream itself is not seekable, so a resumed run starts a fresh pass over the
dataset from a freshly-shuffled position -- see README "Resumability").

Usage:
    python run_smoke.py --config configs/smoke.json
    python run_smoke.py --config configs/full.json
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import checks
import harvest
import sae as sae_mod
import steer
from config import Config
from report import WorkerClient, sha256_of_payload, _progress

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("autolabs_3c.run_smoke")

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

HELD_OUT_BATCH_SIZE = 4096

# checks.sae_replace_check wants >=40 tokens for a statistically stable
# match_fraction/FVE (smoke-1 used the short "capital of France" prompt
# below and saw match_fraction 0.0 on only 10 positions -- see
# SMOKE-1.md problem 5). This is deliberately long-winded.
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


def _prepare_batch(tokenizer, convs: List[List[Dict[str, str]]], config: Config, device):
    prepared = []
    for messages in convs:
        input_ids, mask = harvest.compute_assistant_mask(tokenizer, messages, max_seq=config.max_seq)
        if not any(mask):
            continue
        prepared.append((input_ids, mask))
    if not prepared:
        return None
    max_len = max(len(ids) for ids, _ in prepared)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    batch_ids = torch.full((len(prepared), max_len), pad_id, dtype=torch.long, device=device)
    batch_attn = torch.zeros((len(prepared), max_len), dtype=torch.long, device=device)
    batch_mask = torch.zeros((len(prepared), max_len), dtype=torch.long, device=device)
    for i, (ids, mask) in enumerate(prepared):
        n = len(ids)
        batch_ids[i, :n] = torch.tensor(ids, device=device)
        batch_attn[i, :n] = 1
        batch_mask[i, : len(mask)] = torch.tensor(mask, device=device)
    return batch_ids, batch_attn, batch_mask


def stage_harvest_train(config: Config, workdir: Path, client: WorkerClient, model, tokenizer, device):
    sae_path = workdir / "sae.safetensors"
    stats_path = workdir / "feature_stats.json"
    if sae_path.exists() and stats_path.exists():
        logger.info("[train] trained SAE + feature stats already exist, skipping harvest+train")
        trained_sae = sae_mod.MatryoshkaBatchTopKSAE.load(
            sae_path, shells=config.matryoshka_shells, k=config.k, aux_loss_coef=config.aux_loss_coef
        ).to(device)
        feature_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        return trained_sae, feature_stats

    d_model = model.config.hidden_size
    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_files = sorted(
        checkpoint_dir.glob("sae_step_*.safetensors"), key=lambda p: int(p.stem.split("_")[-1])
    )

    tokens_done = 0
    if resume_files:
        latest = resume_files[-1]
        tokens_done = int(latest.stem.split("_")[-1])
        logger.info("[train] resuming from checkpoint at %d tokens", tokens_done)
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
    stats_tracker = sae_mod.FeatureStatsTracker(config.sae_width)

    buffer = harvest.ShuffleBuffer(config.shuffle_buffer_size, d_model, seed=config.seed)
    last_out = None

    with harvest.ActivationHarvester(model, config.layer) as harvester:
        conv_stream = harvest.stream_conversations(config.dataset_name, config.dataset_split, seed=config.seed)
        next_checkpoint = tokens_done + config.checkpoint_every_tokens
        convo_batch: List[List[Dict[str, str]]] = []
        convs_seen = 0

        for messages in conv_stream:
            convo_batch.append(messages)
            if len(convo_batch) < config.harvest_batch_size:
                continue

            batch = _prepare_batch(tokenizer, convo_batch, config, device)
            convo_batch = []
            convs_seen += config.harvest_batch_size
            if batch is None:
                if convs_seen >= 64 and tokens_done == 0:
                    raise RuntimeError("harvest produced zero assistant tokens after 64 conversations; assistant mask or dataset shape is wrong")
                continue
            batch_ids, batch_attn, batch_mask = batch
            acts = harvest.masked_activations(harvester, batch_ids, batch_attn, batch_mask)
            if acts.numel() > 0:
                buffer.add(acts.to(torch.float32))
                tokens_done += acts.shape[0]

            if not buffer.is_ready(min_fraction=0.05):
                continue

            x = buffer.sample(config.batch_tokens).to(device)
            optimizer.zero_grad(set_to_none=True)
            last_out = trained_sae.forward_loss(x, dead_window_tokens=config.dead_feature_window_tokens)
            last_out.loss.backward()
            optimizer.step()
            trained_sae.normalize_decoder_()
            stats_tracker.observe(last_out.codes, progress_fraction=min(1.0, tokens_done / config.tokens_target))

            if tokens_done >= next_checkpoint or tokens_done >= config.tokens_target:
                ckpt_path = checkpoint_dir / f"sae_step_{tokens_done}.safetensors"
                trained_sae.save(ckpt_path)
                fve = last_out.fraction_variance_explained.item()
                l0 = last_out.l0.item()
                dead_frac = last_out.dead_fraction.item()
                logger.info(
                    "[train] tokens=%d loss=%.6f fve=%.6f l0=%.4f dead_frac=%.6f",
                    tokens_done, last_out.loss.item(), fve, l0, dead_frac,
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

        # -- held-out FVE/L0 on a fresh batch harvested *after* training,
        # never seen by the optimizer (the shuffle-buffer samples used for
        # every training step above are not held out). Still inside the
        # ActivationHarvester context so the model is still truncated to
        # layer `config.layer` and the stream continues from wherever the
        # training loop left off.
        held_out_fve: Optional[float] = None
        held_out_l0: Optional[float] = None
        holdout_buffer = harvest.ShuffleBuffer(HELD_OUT_BATCH_SIZE, d_model, seed=config.seed + 1)
        holdout_convo_batch: List[List[Dict[str, str]]] = []
        holdout_batches_tried = 0
        while holdout_buffer.filled < HELD_OUT_BATCH_SIZE and holdout_batches_tried < 500:
            try:
                messages = next(conv_stream)
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
                holdout_buffer.add(h_acts.to(torch.float32))

        if holdout_buffer.filled > 0:
            x_holdout = holdout_buffer.sample(holdout_buffer.filled).to(device)
            with torch.no_grad():
                holdout_out = trained_sae.forward_loss(x_holdout, dead_window_tokens=config.dead_feature_window_tokens)
            held_out_fve = holdout_out.fraction_variance_explained.item()
            held_out_l0 = holdout_out.l0.item()
            logger.info(
                "[train] held_out_fve=%.6f held_out_l0=%.4f (n=%d activations)",
                held_out_fve, held_out_l0, holdout_buffer.filled,
            )
        else:
            logger.warning("[train] collected zero held-out activations; held_out_fve/held_out_l0 will be null")

    feature_stats = stats_tracker.to_dict()
    feature_stats["held_out_fve"] = held_out_fve
    feature_stats["held_out_l0"] = held_out_l0
    trained_sae.save(sae_path, feature_stats=feature_stats)
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


def stage_steer(config: Config, workdir: Path, client: WorkerClient, model, tokenizer, trained_sae, feature_stats, scenarios, device):
    path = workdir / "calibration_records.json"
    if path.exists():
        logger.info("[calibrate] records already exist, skipping steer stage")
        return json.loads(path.read_text(encoding="utf-8"))

    records = steer.run_calibration(config, model, tokenizer, trained_sae, feature_stats, scenarios, device, seed=config.seed)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    client.report("calibrate", progress=_progress(len(records), len(records)), records=records)
    return records


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


def build_summary(config: Config, boot: dict, post_train: dict, calibration_records: List[dict]) -> dict:
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

    return {
        "config": config.to_dict(),
        "boot_checks": boot,
        "post_train_check": post_train,
        "baseline": baseline_text,
        "features": sorted(features.values(), key=lambda e: e["feature"]),
        "random_control": random_rows,
        "random_control_max_coherent_dose": random_max_coherent_dose,
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
  pre {{ white-space: pre-wrap; margin: 0; max-width: 32rem; }}
  h1, h2 {{ font-weight: 600; }}
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
</body>
</html>
"""


def stage_analysis(config: Config, workdir: Path, client: WorkerClient, boot, post_train, calibration_records):
    summary_path = workdir / "summary.json"
    report_path = workdir / "smoke-report.html"
    if summary_path.exists() and report_path.exists():
        logger.info("[analyze] summary + report already exist, skipping")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    summary = build_summary(config, boot, post_train, calibration_records)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(render_html_report(summary), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment 3C SAE pipeline")
    parser.add_argument("--config", default="configs/smoke.json")
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
        trained_sae, feature_stats = stage_harvest_train(config, workdir, client, model, tokenizer, device)
        post_train = stage_post_train_check(
            config, workdir, client, model, tokenizer, trained_sae, feature_stats, device
        )

        scenarios = load_scenarios(Path(config.scenarios_file))
        scenarios = scenarios[: config.steer_scenarios]
        calibration_records = stage_steer(
            config, workdir, client, model, tokenizer, trained_sae, feature_stats, scenarios, device
        )

        summary = stage_analysis(config, workdir, client, boot, post_train, calibration_records)

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
