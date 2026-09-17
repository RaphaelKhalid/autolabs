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

**Batching.** Generation goes through ``generation.generate_batch`` (see
that module's docstring for the left-padding + hook-factory contract)
rather than looping over ``model.generate`` one prompt at a time.
``run_calibration`` batches every (dose, scenario) pair for one
feature/sign into a single call -- the additive steering hook supports a
per-row dose *scale* (``make_additive_hook``'s ``scales`` argument) on top
of the existing per-row *boundary*, so one hook registration covers the
whole sweep instead of one per generation.
"""
from __future__ import annotations

import logging
import math
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import torch

import generation

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
    explicit_features: Optional[Sequence[Dict[str, Any]]] = None,
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

    If `explicit_features` is given (the rank stage's ranked candidate list,
    e.g. `candidates.json["screen_features"]` -- see rank.py), it is used
    as-is instead of density-quantile selection: `steer_features`,
    `first_shell_size`, `density_min`, and `density_max` are ignored
    entirely, since the rank stage already applied its own density/dead/
    shell filtering. Each entry becomes `{feature, density, max_activation,
    quantile: None, arm}` (`quantile` is meaningless for an explicit list
    but kept so every caller sees the same schema; `arm` is carried through
    from the rank stage's `"unsupervised"`/`"quantile"`/`"shift"` tag so
    `run_calibration` can propagate it onto every steered record).
    """
    if explicit_features:
        selected = []
        for entry in explicit_features:
            f_idx = entry["feature"]
            density = entry.get("density")
            if density is None:
                density = feature_stats["firing_density"][f_idx]
            max_act = entry.get("max_activation")
            if max_act is None:
                max_act = feature_stats["max_activation"][f_idx]
            selected.append(
                {
                    "feature": f_idx,
                    "density": density,
                    "max_activation": max_act,
                    "quantile": None,
                    "arm": entry.get("arm"),
                }
            )
        return selected

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
        entry["arm"] = None  # no rank stage in play; not one of its three arms
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
def _as_row_tensor(values: Any, batch: int, dtype: torch.dtype) -> torch.Tensor:
    """Coerces `values` (a tensor, a python sequence, or a single scalar
    broadcast to every row) to a 1-D tensor of length `batch`."""
    if isinstance(values, torch.Tensor):
        t = values
    elif isinstance(values, (list, tuple)):
        t = torch.tensor(list(values))
    else:
        t = torch.full((batch,), float(values))
    if t.dtype != dtype:
        t = t.to(dtype)
    return t


def make_additive_hook(
    vector: torch.Tensor,
    boundaries: Any,
    position_state: List[int],
    scales: Optional[Any] = None,
):
    """Adds `scales[row] * vector` to the residual stream (output[0]) of
    every row in a (possibly left-padded, possibly batched) chunk, but only
    at absolute sequence positions >= `boundaries[row]` -- i.e. never on
    that row's own left-padding or user-prompt tokens.

    `boundaries` is a `[batch]` tensor/sequence of *padded-coordinate*
    positions (a row's left-pad count plus its own unpadded assistant-turn
    boundary -- see `build_steering_hook_factory`, which is what actually
    shifts an unpadded boundary into this coordinate system before calling
    here); `scales` is an optional `[batch]` tensor/sequence of per-row
    multipliers (e.g. `sign * dose`), defaulting to all-ones so a single
    shared `vector` can still be used unscaled. `position_state` is a
    1-element mutable list tracking the absolute start position of the next
    forward-pass chunk this hook will see -- `model.generate` calls the
    hook once per forward pass: the first call covers the whole (padded)
    prompt, every call after that covers exactly one new decode step, with
    KV-caching keeping every row's absolute position in lock-step across
    the batch. This correctly covers "padding and prompt positions before
    the boundary are left untouched; the boundary column onward, plus every
    generated token, gets steered" for every row independently, without
    needing to know in advance how many tokens will be generated.

    A row whose boundary is `>=` a decode-step's absolute position is
    impossible in practice (the boundary is always strictly before the end
    of that row's own padded prompt -- there is always at least the
    assistant-header tokens after it), so every decode step steers every
    row; only the first (prefill) forward pass needs the per-row mask."""

    def hook(module: Any, inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        batch, seq_len, d_model = hidden.shape
        start = position_state[0]
        end = start + seq_len
        position_state[0] = end

        boundary_t = _as_row_tensor(boundaries, batch, torch.long).to(hidden.device)
        scale_t = (
            _as_row_tensor(scales, batch, hidden.dtype).to(hidden.device)
            if scales is not None
            else torch.ones(batch, dtype=hidden.dtype, device=hidden.device)
        )

        positions = torch.arange(start, end, device=hidden.device).view(1, seq_len)
        mask = (positions >= boundary_t.view(batch, 1)).to(hidden.dtype)  # [batch, seq_len]
        if not bool(mask.any()):
            return output  # this chunk is entirely before every row's boundary: no-op

        add_vec = vector.to(hidden.dtype).to(hidden.device)  # [d_model]
        per_row_vec = scale_t.view(batch, 1) * add_vec.view(1, d_model)  # [batch, d_model]
        hidden = hidden + per_row_vec.view(batch, 1, d_model) * mask.view(batch, seq_len, 1)

        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    return hook


def register_additive_hook(
    model: Any,
    layer: int,
    vector: Optional[torch.Tensor],
    boundaries: Any = 0,
    scales: Optional[Any] = None,
):
    """Returns a handle, or None if vector is None (unsteered baseline).
    `boundaries`/`scales` accept anything `make_additive_hook` does
    (tensor, plain sequence, or scalar broadcast to every row)."""
    if vector is None:
        return None
    base = getattr(model, "model", model)
    position_state = [0]
    return base.layers[layer].register_forward_hook(
        make_additive_hook(vector, boundaries, position_state, scales=scales)
    )


def build_steering_hook_factory(
    model: Any,
    layer: int,
    vector: Optional[torch.Tensor],
    unpadded_boundaries: Any,
    scales: Optional[Any] = None,
):
    """Returns a `generation.generate_batch`-compatible hook factory: a
    callable `factory(pad_lengths, row_start=0, row_end=None) ->
    Optional[handle]` that shifts each row's already-known *unpadded*
    boundary into padded coordinates (`pad_length[row] +
    unpadded_boundary[row]`) and registers the actual additive hook, fresh
    (a new `position_state`) for every chunk. `vector is None` gives a
    no-op factory (unsteered baseline), matching `register_additive_hook`'s
    `vector=None` convention.

    `unpadded_boundaries`/`scales` are per-row tensors for the *whole*
    request `generate_batch` was called with, not just one chunk --
    `generate_batch` may run the request in several `batch_size`-sized
    chunks (see its docstring), calling this factory once per chunk with
    only that chunk's `pad_lengths` (length `chunk_batch`, not the full
    request length). `row_start`/`row_end` (this chunk's row range within
    the full request -- `generate_batch` passes these automatically when
    it detects this factory's 3-argument signature, see `generation.
    _call_hook`) tell `factory` which slice of its own request-sized
    tensors matches this chunk's `pad_lengths`, so the two are never
    mismatched in size (the bug this signature fixes: a 16-row request
    chunked at `batch_size=8` used to call `factory(pad_lengths)` with an
    8-row `pad_lengths` against this closure's 16-row boundaries/scales,
    raising a tensor-size-mismatch `RuntimeError` inside the hook).
    `row_start`/`row_end` default to `(0, len(pad_lengths))` -- the whole
    closure -- so a caller that still invokes `factory(pad_lengths)`
    directly for an unchunked (single-chunk) request keeps working
    unchanged."""
    if vector is None:
        return lambda pad_lengths, row_start=0, row_end=None: None

    n_total = len(unpadded_boundaries) if hasattr(unpadded_boundaries, "__len__") else 1
    unpadded_t = _as_row_tensor(unpadded_boundaries, n_total, torch.long)
    scales_t = _as_row_tensor(scales, n_total, torch.float32) if scales is not None else None

    def factory(pad_lengths: torch.Tensor, row_start: int = 0, row_end: Optional[int] = None) -> Optional[Any]:
        n_chunk = pad_lengths.shape[0] if isinstance(pad_lengths, torch.Tensor) else len(pad_lengths)
        if row_end is None:
            row_end = row_start + n_chunk
        boundaries_chunk = unpadded_t[row_start:row_end]
        scales_chunk = scales_t[row_start:row_end] if scales_t is not None else None
        boundaries = pad_lengths.to(torch.long) + boundaries_chunk.to(pad_lengths.device)
        return register_additive_hook(model, layer, vector, boundaries, scales=scales_chunk)

    return factory


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    device: Any,
    hook: Optional[Any] = None,
) -> Dict[str, Any]:
    """Single-prompt generation: the `batch_size=1` special case of
    `generation.generate_batch`. `hook`, if given, is a
    `generate_batch`-style hook factory (see `build_steering_hook_factory`)
    -- there is no separate unbatched generation path any more."""
    messages = [{"role": "user", "content": prompt}]
    return generation.generate_batch(
        model, tokenizer, [messages], max_new_tokens, device, hook=hook, batch_size=1
    )[0]


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


def coherence_logprob_batch(
    model: Any,
    tokenizer: Any,
    prompt_ids_list: Sequence[Sequence[int]],
    generated_ids_list: Sequence[Sequence[int]],
    device: Any,
) -> List[float]:
    """Batched, right-padded version of `coherence_logprob`: one extra
    forward pass (no hook, no KV-cache growth -- a plain forward pass, so
    right-padding is safe: causal masking already blocks every real token
    from attending to anything after it, so trailing pad columns cannot
    affect an earlier real token's logits) over the whole chunk, mean
    log-prob of each row's own `generated_ids` under the *unsteered* model
    computed only over that row's real (non-pad) positions. Caller must
    ensure no steering hook is registered when this runs. 0.0 for a row
    with no generated tokens."""
    n = len(prompt_ids_list)
    results = [0.0] * n
    rows = [i for i in range(n) if len(generated_ids_list[i]) > 0]
    if not rows:
        return results

    full_seqs = [list(p) + list(g) for p, g in zip(prompt_ids_list, generated_ids_list)]
    max_len = max(len(s) for s in full_seqs)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0) or 0
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(full_seqs):
        m = len(seq)
        if m == 0:
            continue
        input_ids[i, :m] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, :m] = 1

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    for i in rows:
        prompt_len = len(prompt_ids_list[i])
        gen_len = len(generated_ids_list[i])
        # logits[t] predicts token[t+1]; we need predictions for positions
        # prompt_len .. prompt_len+gen_len-1, predicting the generated tokens.
        pred_logits = logits[i, prompt_len - 1 : prompt_len - 1 + gen_len, :]
        targets = input_ids[i, prompt_len : prompt_len + gen_len]
        log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
        token_logprobs = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
        results[i] = token_logprobs.mean().item()
    return results


def coherence_logprob(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    generated_ids: Sequence[int],
    device: Any,
) -> float:
    """Single-generation wrapper (batch of one) around
    `coherence_logprob_batch`."""
    return coherence_logprob_batch(model, tokenizer, [prompt_ids], [generated_ids], device)[0]


def compute_coherence_batch(
    model: Any,
    tokenizer: Any,
    prompt_ids_list: Sequence[Sequence[int]],
    generated_ids_list: Sequence[Sequence[int]],
    device: Any,
) -> List[Dict[str, Any]]:
    """Batched version of `compute_coherence`: `distinct_ratio`/`max_run`/
    `repeat_4gram` are pure-Python over each row's own token-id list
    (already cheap, not batched further); `logprob` is the one part that
    needs the model, computed for the whole chunk in a single forward pass
    via `coherence_logprob_batch`."""
    logprobs = coherence_logprob_batch(model, tokenizer, prompt_ids_list, generated_ids_list, device)
    out = []
    for generated_ids, logprob in zip(generated_ids_list, logprobs):
        distinct = distinct_ratio(generated_ids)
        run = max_run_length(generated_ids)
        repeat_4gram = repeat_4gram_fraction(generated_ids)
        out.append(
            {
                "distinct_ratio": distinct,
                "max_run": run,
                "repeat_4gram": repeat_4gram,
                "logprob": logprob,
                "coherent": is_coherent(distinct, run, repeat_4gram),
            }
        )
    return out


def compute_coherence(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    generated_ids: Sequence[int],
    device: Any,
) -> Dict[str, Any]:
    """Single-generation wrapper (batch of one) around
    `compute_coherence_batch`."""
    return compute_coherence_batch(model, tokenizer, [prompt_ids], [generated_ids], device)[0]


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
    explicit_features: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Runs the full dose-sweep calibration screen and returns a flat list of
    {recordId, payload} records, ready for `report.report(stage="calibrate",
    records=...)`. `explicit_features`, when given, is the rank stage's
    ranked candidate list (see `select_steer_features`) used in place of
    density-quantile selection.

    Batching: baselines are one call across every scenario; for each
    feature/sign (and each random-direction draw), every (dose, scenario)
    pair is generated in a single `generation.generate_batch` call (chunked
    internally at `config.generation_batch_size`) rather than one
    `model.generate` per pair -- the additive steering hook takes a
    per-row dose *scale* (`sign * dose`) on top of the existing per-row
    assistant-turn boundary, so one hook factory covers the whole sweep."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first_shell = config.matryoshka_shells[0]
    selected = select_steer_features(
        feature_stats,
        first_shell_size=first_shell,
        steer_features=config.steer_features,
        density_min=config.firing_density_min,
        density_max=config.firing_density_max,
        explicit_features=explicit_features,
    )
    if not selected:
        logger.warning("no features met the density window; calibration will only cover baselines/controls")

    median_max_act = median_max_activation(selected)
    d_model = sae.d_in
    W_dec = sae.W_dec.detach()
    batch_size = config.generation_batch_size

    boundary_by_scenario = {s["id"]: compute_generation_boundary(tokenizer, s["prompt"]) for s in scenarios}

    records: List[Dict[str, Any]] = []

    # -- baselines (unsteered), one batch across every scenario ------------
    baseline_msgs = [[{"role": "user", "content": s["prompt"]}] for s in scenarios]
    baseline_gens = generation.generate_batch(
        model, tokenizer, baseline_msgs, config.max_new_tokens, device, hook=None, batch_size=batch_size
    )
    baseline_by_scenario: Dict[str, Dict[str, Any]] = {}
    for scenario, gen in zip(scenarios, baseline_gens):
        baseline_by_scenario[scenario["id"]] = gen
        records.append(
            {
                "recordId": f"baseline-{scenario['id']}",
                "payload": {
                    "kind": "baseline",
                    "arm": None,
                    "scenario": scenario["id"],
                    "text": gen["text"],
                    "finish_reason": gen["finish_reason"],
                    "num_tokens": gen["num_tokens"],
                },
            }
        )

    # -- real feature dose sweep: batch every (dose, scenario) pair --------
    for feat in selected:
        f_idx = feat["feature"]
        direction = W_dec[f_idx]
        base_vector = feat["max_activation"] * direction  # dose/sign applied as a per-row scale below
        for sign in (1, -1):
            combos = [(dose, scenario) for dose in config.doses for scenario in scenarios]
            msgs = [[{"role": "user", "content": scenario["prompt"]}] for _dose, scenario in combos]
            boundaries = torch.tensor(
                [boundary_by_scenario[scenario["id"]] for _dose, scenario in combos], dtype=torch.long
            )
            scales = torch.tensor([sign * dose for dose, _scenario in combos], dtype=torch.float32)
            hook_factory = build_steering_hook_factory(model, config.layer, base_vector, boundaries, scales=scales)
            gens = generation.generate_batch(
                model, tokenizer, msgs, config.max_new_tokens, device, hook=hook_factory, batch_size=batch_size
            )
            coherences = compute_coherence_batch(
                model,
                tokenizer,
                [g["prompt_ids"] for g in gens],
                [g["generated_ids"] for g in gens],
                device,
            )
            for (dose, scenario), gen, coherence in zip(combos, gens, coherences):
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
                            "arm": feat.get("arm"),
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
        base_vector = median_max_act * random_unit
        combos = [(dose, scenario) for dose in config.doses for scenario in scenarios]
        msgs = [[{"role": "user", "content": scenario["prompt"]}] for _dose, scenario in combos]
        boundaries = torch.tensor(
            [boundary_by_scenario[scenario["id"]] for _dose, scenario in combos], dtype=torch.long
        )
        scales = torch.tensor([dose for dose, _scenario in combos], dtype=torch.float32)
        hook_factory = build_steering_hook_factory(model, config.layer, base_vector, boundaries, scales=scales)
        gens = generation.generate_batch(
            model, tokenizer, msgs, config.max_new_tokens, device, hook=hook_factory, batch_size=batch_size
        )
        coherences = compute_coherence_batch(
            model,
            tokenizer,
            [g["prompt_ids"] for g in gens],
            [g["generated_ids"] for g in gens],
            device,
        )
        for (dose, scenario), gen, coherence in zip(combos, gens, coherences):
            records.append(
                {
                    "recordId": f"random-{r}-{dose}-{scenario['id']}",
                    "payload": {
                        "kind": "random_control",
                        "arm": "random",
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
