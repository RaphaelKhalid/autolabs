"""Dose-sweep steering + calibration.

Loads the *full* model (all decoder layers restored -- harvesting truncated
them) and the trained SAE, picks `steer_features` inner-shell (first shell)
features with firing density in
[firing_density_min, firing_density_max], and for each feature/sign/dose
generates every scenario with a hook that adds
`sign * dose * max_act[f] * W_dec[f]` to the residual stream at every
position (prompt and generated tokens) of the harvest layer.

Also generates, once per scenario: the unsteered baseline, and a
random-direction null control swept across the same doses (2 random unit
vectors scaled to `dose * median(max_act over selected features)`, matching
the real-feature dose scale so the control is comparable).

Coherence proxy: mean per-token log-prob of the *steered* generated text
under the *unsteered* model (a second, hook-free forward pass in teacher-
forcing mode over prompt + generated tokens).

Everything needed here (model.generate, forward hooks on real decoder
layers) requires a GPU-resident 7B model; this module is not exercised by
the CPU test suite beyond feature selection and record shaping, which are
pure-Python/tensor math.
"""
from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional, Sequence

import torch

logger = logging.getLogger("autolabs_3c.steer")


# ---------------------------------------------------------------------------
# Feature selection (CPU-testable: pure dict/list math)
# ---------------------------------------------------------------------------
def select_steer_features(
    feature_stats: Dict[str, Any],
    first_shell_size: int,
    steer_features: int,
    density_min: float,
    density_max: float,
) -> List[Dict[str, Any]]:
    """Pick the `steer_features` highest-density features among the first
    `first_shell_size` (the innermost matryoshka shell) whose firing density
    falls in [density_min, density_max]."""
    densities = feature_stats["firing_density"]
    max_acts = feature_stats["max_activation"]
    candidates = []
    for idx in range(min(first_shell_size, len(densities))):
        d = densities[idx]
        if density_min <= d <= density_max:
            candidates.append({"feature": idx, "density": d, "max_activation": max_acts[idx]})
    candidates.sort(key=lambda c: c["density"], reverse=True)
    return candidates[:steer_features]


def median_max_activation(selected_features: Sequence[Dict[str, Any]]) -> float:
    if not selected_features:
        return 0.0
    return statistics.median(c["max_activation"] for c in selected_features)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def make_additive_hook(vector: torch.Tensor):
    """Adds `vector` to the residual stream (output[0]) at every position."""

    def hook(module: Any, inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + vector.to(hidden.dtype).to(hidden.device)
            return (hidden,) + tuple(output[1:])
        return output + vector.to(output.dtype).to(output.device)

    return hook


def register_additive_hook(model: Any, layer: int, vector: Optional[torch.Tensor]):
    """Returns a handle, or None if vector is None (unsteered baseline)."""
    if vector is None:
        return None
    base = getattr(model, "model", model)
    return base.layers[layer].register_forward_hook(make_additive_hook(vector))


# ---------------------------------------------------------------------------
# Generation + coherence
# ---------------------------------------------------------------------------
def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    device: Any,
) -> Dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    prompt_len = input_ids.shape[1]
    generated_ids = out[0, prompt_len:].tolist()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    finish_reason = "length"
    if eos_id is not None and len(generated_ids) > 0 and generated_ids[-1] == eos_id:
        finish_reason = "eos"
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "prompt_ids": input_ids[0].tolist(),
        "generated_ids": generated_ids,
        "text": text,
        "finish_reason": finish_reason,
        "num_tokens": len(generated_ids),
    }


def coherence_logprob(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    generated_ids: Sequence[int],
    device: Any,
) -> float:
    """Mean log-prob of `generated_ids` under the *unsteered* model, given
    `prompt_ids` as context. Caller must ensure no steering hook is
    registered when this runs."""
    if not generated_ids:
        return 0.0
    full_ids = torch.tensor([list(prompt_ids) + list(generated_ids)], device=device)
    with torch.no_grad():
        logits = model(input_ids=full_ids).logits
    prompt_len = len(prompt_ids)
    # logits[t] predicts token[t+1]; we need predictions for positions
    # prompt_len .. end-1, predicting tokens prompt_len .. end.
    pred_logits = logits[0, prompt_len - 1 : -1, :]
    targets = full_ids[0, prompt_len:]
    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    token_logprobs = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logprobs.mean().item()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_calibration(
    config: Any,
    model: Any,
    tokenizer: Any,
    sae: Any,
    feature_stats: Dict[str, Any],
    scenarios: List[Dict[str, str]],
    device: Any,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Runs the full dose-sweep calibration screen and returns a flat list of
    {recordId, payload} records, ready for `report.report(stage="calibrate",
    records=...)`."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first_shell = config.matryoshka_shells[0]
    selected = select_steer_features(
        feature_stats,
        first_shell_size=first_shell,
        steer_features=config.steer_features,
        density_min=config.firing_density_min,
        density_max=config.firing_density_max,
    )
    if not selected:
        logger.warning("no features met the density window; calibration will only cover baselines/controls")

    median_max_act = median_max_activation(selected)
    d_model = sae.d_in
    W_dec = sae.W_dec.detach()

    records: List[Dict[str, Any]] = []

    # -- baselines (unsteered), once per scenario --------------------------
    baseline_by_scenario: Dict[str, Dict[str, Any]] = {}
    for scenario in scenarios:
        gen = generate_text(model, tokenizer, scenario["prompt"], config.max_new_tokens, device)
        baseline_by_scenario[scenario["id"]] = gen
        records.append(
            {
                "recordId": f"baseline-{scenario['id']}",
                "payload": {
                    "kind": "baseline",
                    "scenario": scenario["id"],
                    "text": gen["text"],
                    "finish_reason": gen["finish_reason"],
                    "num_tokens": gen["num_tokens"],
                },
            }
        )

    # -- real feature dose sweep --------------------------------------------
    for feat in selected:
        f_idx = feat["feature"]
        direction = W_dec[f_idx]
        for sign in (1, -1):
            for dose in config.doses:
                vector = sign * dose * feat["max_activation"] * direction
                for scenario in scenarios:
                    handle = register_additive_hook(model, config.layer, vector)
                    try:
                        gen = generate_text(model, tokenizer, scenario["prompt"], config.max_new_tokens, device)
                    finally:
                        if handle is not None:
                            handle.remove()
                    coherence = coherence_logprob(
                        model, tokenizer, gen["prompt_ids"], gen["generated_ids"], device
                    )
                    records.append(
                        {
                            "recordId": f"steer-{f_idx}-{'pos' if sign > 0 else 'neg'}-{dose}-{scenario['id']}",
                            "payload": {
                                "kind": "steered",
                                "feature": f_idx,
                                "sign": sign,
                                "dose": dose,
                                "density": feat["density"],
                                "max_activation": feat["max_activation"],
                                "scenario": scenario["id"],
                                "text": gen["text"],
                                "finish_reason": gen["finish_reason"],
                                "num_tokens": gen["num_tokens"],
                                "coherence_logprob": coherence,
                            },
                        }
                    )

    # -- random-direction null control, swept across doses -----------------
    for r in range(2):
        random_unit = torch.randn(d_model, generator=generator)
        random_unit = random_unit / random_unit.norm().clamp_min(1e-8)
        for dose in config.doses:
            vector = dose * median_max_act * random_unit
            for scenario in scenarios:
                handle = register_additive_hook(model, config.layer, vector.to(device))
                try:
                    gen = generate_text(model, tokenizer, scenario["prompt"], config.max_new_tokens, device)
                finally:
                    if handle is not None:
                        handle.remove()
                coherence = coherence_logprob(
                    model, tokenizer, gen["prompt_ids"], gen["generated_ids"], device
                )
                records.append(
                    {
                        "recordId": f"random-{r}-{dose}-{scenario['id']}",
                        "payload": {
                            "kind": "random_control",
                            "random_index": r,
                            "dose": dose,
                            "scenario": scenario["id"],
                            "text": gen["text"],
                            "finish_reason": gen["finish_reason"],
                            "num_tokens": gen["num_tokens"],
                            "coherence_logprob": coherence,
                        },
                    }
                )

    return records
