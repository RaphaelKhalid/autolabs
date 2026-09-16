"""Dose-sweep steering + calibration.

Loads the *full* model (all decoder layers restored -- harvesting truncated
them) and the trained SAE, picks `steer_features` inner-shell (first shell)
features spread evenly across the log-density distribution (see
`select_steer_features`) with firing density in
[firing_density_min, firing_density_max], and for each feature/sign/dose
generates every scenario with a hook that adds
`sign * dose * max_act[f] * W_dec[f]` to the residual stream, but only at
*assistant-turn* positions -- the prompt positions from the generation-
prompt boundary onward (i.e. the "<|im_start|>assistant\\n" header) plus
every generated token. User-prompt tokens are left untouched. Smoke-1 added
the vector at every position including the user's own message, and used raw
multiples [2, 4, 8] of max activation; both amplified the effect far beyond
what a persona-direction hypothesis needs and pushed dose 2 into ~90% text
rewrites and dose 8 into pure repetition (see SMOKE-1.md problem 1). doses
are now fractions of max activation (config default [0.25, 0.5, 1.0, 2.0]).

Also generates, once per scenario: the unsteered baseline, and a
random-direction null control swept across the same doses (2 random unit
vectors scaled to `dose * median(max_act over selected features)`, matching
the real-feature dose scale so the control is comparable) -- the random
control uses the exact same assistant-turn-only masking.

Coherence: computed on the *generated* tokens only, per generation --
`distinct_ratio`, `max_run`, `repeat_4gram`, and the original mean
per-token log-prob of the steered text under the unsteered model
(`logprob`), plus a derived `coherent` boolean. Smoke-1's only coherence
signal was `logprob`, which scores degenerate repetition ("platform
platform platform...") *better* than fluent text, because repeating a
common token is exactly what an unsteered LM already assigns high
probability to (see SMOKE-1.md problem 2).

Everything needed here (model.generate, forward hooks on real decoder
layers) requires a GPU-resident 7B model; this module is not exercised by
the CPU test suite beyond feature selection, position-boundary computation,
coherence metrics, and record shaping, which are pure-Python/tensor math
that does not need a GPU or a real model.
"""
from __future__ import annotations

import logging
import math
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import torch

logger = logging.getLogger("autolabs_3c.steer")


# ---------------------------------------------------------------------------
# Feature selection (CPU-testable: pure dict/list math)
# ---------------------------------------------------------------------------
def quantile_indices(n_candidates: int, n_select: int) -> List[int]:
    """Indices (0-based, ascending) into a length-`n_candidates` sequence
    that are spread evenly across quantiles: for `n_select` picks, the
    centers of `n_select` equal-width percentile bins, e.g. for
    `n_select=10` these land at (approximately) the 5th, 15th, ..., 95th
    percentile positions. Guarantees strictly increasing, distinct indices
    (as long as `n_select <= n_candidates`) by nudging a collision forward
    to the next free slot."""
    if n_candidates <= 0 or n_select <= 0:
        return []
    n_select = min(n_select, n_candidates)
    idxs: List[int] = []
    for i in range(n_select):
        pct = (i + 0.5) / n_select
        target = pct * (n_candidates - 1)
        idx = int(target + 0.5)
        idx = max(0, min(n_candidates - 1, idx))
        if idxs and idx <= idxs[-1]:
            idx = min(n_candidates - 1, idxs[-1] + 1)
        idxs.append(idx)
    return idxs


def select_steer_features(
    feature_stats: Dict[str, Any],
    first_shell_size: int,
    steer_features: int,
    density_min: float,
    density_max: float,
) -> List[Dict[str, Any]]:
    """Pick `steer_features` features among the first `first_shell_size`
    (the innermost matryoshka shell) whose firing density falls in
    [density_min, density_max], spread evenly across quantiles of the
    *log*-density distribution rather than just the top-N by density.

    Smoke-1 took the `steer_features` densest first-shell features, which
    all landed within a hair of the density cap (~0.097, just under 0.1)
    -- a narrow, redundant slice of the density spectrum rather than a
    sample of qualitatively different feature types (see SMOKE-1.md
    problem 3). Sorting ascending and picking evenly-spaced quantile
    positions instead samples across "moderately common" through "fairly
    rare" first-shell features.

    Each returned dict also carries `quantile` (the target percentile, 0
    to 100, this feature was picked to represent) alongside `feature`,
    `density`, and `max_activation`.
    """
    densities = feature_stats["firing_density"]
    max_acts = feature_stats["max_activation"]
    candidates = []
    for idx in range(min(first_shell_size, len(densities))):
        d = densities[idx]
        if density_min <= d <= density_max:
            candidates.append({"feature": idx, "density": d, "max_activation": max_acts[idx]})
    if not candidates:
        return []

    candidates.sort(key=lambda c: c["density"])
    n = len(candidates)
    n_select = min(steer_features, n)
    picks = quantile_indices(n, n_select)

    selected = []
    for i, pos in enumerate(picks):
        entry = dict(candidates[pos])
        entry["quantile"] = round((i + 0.5) / n_select * 100.0, 2)
        selected.append(entry)
    return selected


def median_max_activation(selected_features: Sequence[Dict[str, Any]]) -> float:
    if not selected_features:
        return 0.0
    return statistics.median(c["max_activation"] for c in selected_features)


# ---------------------------------------------------------------------------
# Assistant-turn position boundary (CPU-testable with a fake tokenizer)
# ---------------------------------------------------------------------------
def compute_generation_boundary(tokenizer: Any, prompt: str) -> int:
    """Index (0-based, into the token sequence rendered with
    `add_generation_prompt=True`) where the assistant turn begins: the
    generation-prompt header (e.g. "<|im_start|>assistant\\n") and every
    token generated after it. This is the length of the *same* user
    message rendered *without* the generation prompt, which is a token
    prefix of the `add_generation_prompt=True` rendering under the same
    prefix-stability assumption `harvest.compute_assistant_mask` relies
    on."""
    messages = [{"role": "user", "content": prompt}]
    user_only_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    user_only_ids = tokenizer(user_only_text, add_special_tokens=False)["input_ids"]
    return len(user_only_ids)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def make_additive_hook(vector: torch.Tensor, boundary: int, position_state: List[int]):
    """Adds `vector` to the residual stream (output[0]), but only at
    absolute sequence positions >= `boundary` -- i.e. never on the user's
    own prompt tokens. `position_state` is a 1-element mutable list
    tracking the absolute start position of the next chunk this hook will
    see; `model.generate` calls the hook once per forward pass, and with
    KV-caching each call after the first covers exactly one new token, so
    this correctly covers "prompt positions from the generation-prompt
    boundary onward, plus every generated token" without needing to know
    in advance how many tokens will be generated."""

    def hook(module: Any, inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        seq_len = hidden.shape[1]
        start = position_state[0]
        end = start + seq_len
        position_state[0] = end

        if end <= boundary:
            return output  # entirely user-prompt tokens: no-op

        add_vec = vector.to(hidden.dtype).to(hidden.device)
        if start >= boundary:
            hidden = hidden + add_vec
        else:
            # Mixed chunk: only happens on the first forward pass, which
            # covers the whole prompt (user tokens + assistant header).
            local_boundary = boundary - start
            mask = torch.zeros(seq_len, dtype=hidden.dtype, device=hidden.device)
            mask[local_boundary:] = 1.0
            hidden = hidden + add_vec.view(1, 1, -1) * mask.view(1, seq_len, 1)

        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    return hook


def register_additive_hook(model: Any, layer: int, vector: Optional[torch.Tensor], boundary: int = 0):
    """Returns a handle, or None if vector is None (unsteered baseline)."""
    if vector is None:
        return None
    base = getattr(model, "model", model)
    position_state = [0]
    return base.layers[layer].register_forward_hook(make_additive_hook(vector, boundary, position_state))


# ---------------------------------------------------------------------------
# Generation
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


# ---------------------------------------------------------------------------
# Coherence (CPU-testable: pure Python over token-id lists)
# ---------------------------------------------------------------------------
def distinct_ratio(ids: Sequence[int]) -> float:
    """Unique tokens / total tokens. 1.0 for an empty sequence (vacuously
    "not repetitive")."""
    if not ids:
        return 1.0
    return len(set(ids)) / len(ids)


def max_run_length(ids: Sequence[int]) -> int:
    """Longest run of an identical token back-to-back. 0 for an empty
    sequence."""
    if not ids:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(ids)):
        if ids[i] == ids[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def repeat_4gram_fraction(ids: Sequence[int]) -> float:
    """Fraction of 4-gram occurrences that are repeats of an earlier
    occurrence of the same 4-gram (0.0 if fewer than 4 tokens, or if every
    4-gram is unique)."""
    n = len(ids)
    if n < 4:
        return 0.0
    grams = [tuple(ids[i : i + 4]) for i in range(n - 3)]
    counts = Counter(grams)
    repeat_occurrences = sum(c - 1 for c in counts.values() if c > 1)
    return repeat_occurrences / len(grams)


def is_coherent(distinct: float, max_run: int, repeat_4gram: float) -> bool:
    return distinct >= 0.5 and max_run <= 4 and repeat_4gram <= 0.2


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


def compute_coherence(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    generated_ids: Sequence[int],
    device: Any,
) -> Dict[str, Any]:
    """The full coherence dict for one generation, computed on the
    generated tokens only: `distinct_ratio`, `max_run`, `repeat_4gram`,
    the legacy mean log-prob (`logprob`), and the derived boolean
    `coherent`."""
    distinct = distinct_ratio(generated_ids)
    run = max_run_length(generated_ids)
    repeat_4gram = repeat_4gram_fraction(generated_ids)
    logprob = coherence_logprob(model, tokenizer, prompt_ids, generated_ids, device)
    return {
        "distinct_ratio": distinct,
        "max_run": run,
        "repeat_4gram": repeat_4gram,
        "logprob": logprob,
        "coherent": is_coherent(distinct, run, repeat_4gram),
    }


def max_coherent_dose(rows: Sequence[Dict[str, Any]], min_fraction: float = 0.75) -> Optional[float]:
    """The largest dose that is coherent on at least `min_fraction` of the
    scenarios it was run on (e.g. 3 of 4 for the smoke config's 4
    scenarios), or None if no dose clears that bar. `rows` is a sequence of
    dicts each with `dose` and `coherent` (bool) keys, already filtered to
    a single feature/sign (or a single random-control index)."""
    by_dose: Dict[Any, List[bool]] = {}
    for row in rows:
        by_dose.setdefault(row["dose"], []).append(bool(row.get("coherent")))

    best: Optional[float] = None
    for dose, flags in by_dose.items():
        needed = max(1, math.ceil(min_fraction * len(flags)))
        if sum(flags) >= needed and (best is None or dose > best):
            best = dose
    return best


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
                    boundary = compute_generation_boundary(tokenizer, scenario["prompt"])
                    handle = register_additive_hook(model, config.layer, vector, boundary)
                    try:
                        gen = generate_text(model, tokenizer, scenario["prompt"], config.max_new_tokens, device)
                    finally:
                        if handle is not None:
                            handle.remove()
                    coherence = compute_coherence(
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
                                "quantile": feat.get("quantile"),
                                "max_activation": feat["max_activation"],
                                "scenario": scenario["id"],
                                "text": gen["text"],
                                "finish_reason": gen["finish_reason"],
                                "num_tokens": gen["num_tokens"],
                                "coherence": coherence,
                                "coherent": coherence["coherent"],
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
                boundary = compute_generation_boundary(tokenizer, scenario["prompt"])
                handle = register_additive_hook(model, config.layer, vector.to(device), boundary)
                try:
                    gen = generate_text(model, tokenizer, scenario["prompt"], config.max_new_tokens, device)
                finally:
                    if handle is not None:
                        handle.remove()
                coherence = compute_coherence(
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
                            "coherence": coherence,
                            "coherent": coherence["coherent"],
                        },
                    }
                )

    return records
