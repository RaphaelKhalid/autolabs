"""Rank stage: label-free candidate ranking by persona-context activation
shift.

Runs after harvest+train (and the post-train check) and before calibrate.
The full run screens `config.screen_features` (256) of the 32,768 trained
SAE features; picking those 256 purely by firing-density quantile (what
`steer.select_steer_features` alone does) samples across "how common is
this feature" without any signal about whether a feature's activation
tracks anything persona-like at all. This module adds that signal without
naming a trait: it measures whether a feature's activation shifts when the
*context* (the system prompt the model is responding under) changes across
a diverse set of persona-style contexts, none of which is designated as
"the" target trait -- see README "Rank stage" for the bias this still
introduces (the contexts are diverse but hand-chosen) and why the
density-quantile slice is kept alongside it as an unbiased comparison.

Two independent pieces:

- `compute_shift_stats` (pure numpy/Python, CPU-testable): given each
  context's per-scenario mean SAE-feature activation (already captured
  from the model), computes, per feature, a one-way-ANOVA-style F-like
  ratio (between-context variance of the per-context mean, over pooled
  within-context variance across scenarios) and the max absolute
  difference between any two contexts' per-context mean, normalized by
  the feature's max activation.
- `capture_context_activations` (needs a real transformers model + GPU,
  not exercised by the CPU test suite, mirroring screen.py/steer.py):
  generates a reply per (context, scenario) pair and encodes the layer's
  residual stream at the generated assistant-token positions with
  `sae.encode` to get those per-context, per-scenario mean activations.

`rank_candidates` (pure dict/list math, CPU-testable) then filters
candidates by firing density and liveness and selects the final list as
the union of a top-shift-score slice and a density-quantile spread slice.
"""
from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import screen
import steer

logger = logging.getLogger("autolabs_3c.rank")

# Added to the within-context variance before dividing, so a feature with
# (near-)zero within-context noise gets a large-but-finite F-ratio instead
# of inf/NaN (which can't round-trip through JSON -- report.py's
# `_sanitize` would turn it into null, silently dropping the very signal
# that made the feature interesting).
_WITHIN_VAR_EPS = 1e-8


# ---------------------------------------------------------------------------
# Pure math (CPU-testable)
# ---------------------------------------------------------------------------
def is_dead(density: Optional[float]) -> bool:
    """A feature that never fired during the density-tracking window
    (`sae.FeatureStatsTracker`'s last-20%-of-training window) has
    `firing_density` 0.0 (or missing/None if `feature_stats.json` is
    malformed) and carries no activation signal to rank on."""
    return density is None or density <= 0.0


def feature_shell(feature_idx: int, shells: Sequence[int]) -> Optional[int]:
    """The smallest matryoshka shell (a value from `config.matryoshka_shells`,
    e.g. 1024/4096/8192) that contains `feature_idx`, i.e. the first shell
    boundary strictly greater than the index. None if `feature_idx` is past
    every shell (shouldn't happen when the largest shell equals the SAE
    width, as both shipped configs do)."""
    for shell in shells:
        if feature_idx < shell:
            return shell
    return None


def compute_shift_stats(
    context_scenario_means: Dict[str, Dict[str, Sequence[float]]],
    max_activation: Sequence[float],
) -> List[Dict[str, Any]]:
    """Per-feature label-free shift score from already-captured per-
    (context, scenario) mean activations.

    `context_scenario_means` is `{context_name: {scenario_id:
    [mean_activation_per_feature, ...]}}` -- each leaf is one (context,
    scenario) generation's per-feature mean SAE activation over generated
    assistant tokens (see `capture_context_activations`). Treats each
    context as an ANOVA group and each of its scenarios' per-context mean
    as one observation in that group.

    Returns one `{"feature": f, "shift_f": F, "shift_maxdiff": maxdiff}`
    dict per feature index (0..width-1, `width = len(max_activation)`):

    - `shift_f`: between-group variance (of the per-context mean, weighted
      by scenario count) divided by the pooled within-group variance
      (scenario-to-scenario spread inside each context), i.e. how much of
      this feature's variation is "which persona-style context" rather
      than "which scenario, regardless of context" -- large when a
      feature reliably shifts with context and is stable within a
      context, small when scenario noise dominates or every context
      looks the same. A small epsilon is added to the denominator so a
      near-zero within-group variance gives a large finite ratio instead
      of inf (which is not valid JSON).
    - `shift_maxdiff`: the largest absolute difference between any two
      contexts' per-context mean, normalized by the feature's max
      activation over training (0 if that max activation is ~0), so the
      magnitude of the shift is comparable across features on different
      absolute activation scales.

    Fewer than 2 contexts with any data for a feature yields
    `shift_f = shift_maxdiff = 0.0` for that feature (no shift is
    measurable from a single group).
    """
    contexts = sorted(context_scenario_means)
    width = len(max_activation)
    out: List[Dict[str, Any]] = []

    for f in range(width):
        group_values: List[List[float]] = []
        for c in contexts:
            scenario_vals = [vec[f] for vec in context_scenario_means[c].values() if len(vec) > f]
            if scenario_vals:
                group_values.append(scenario_vals)

        if len(group_values) < 2:
            out.append({"feature": f, "shift_f": 0.0, "shift_maxdiff": 0.0})
            continue

        group_means = [statistics.fmean(vals) for vals in group_values]
        all_vals = [v for vals in group_values for v in vals]
        grand_mean = statistics.fmean(all_vals)

        between_ss = sum(len(vals) * (gm - grand_mean) ** 2 for vals, gm in zip(group_values, group_means))
        between_df = len(group_values) - 1
        between_var = between_ss / between_df if between_df > 0 else 0.0

        within_ss = sum((v - gm) ** 2 for vals, gm in zip(group_values, group_means) for v in vals)
        within_df = len(all_vals) - len(group_values)
        within_var = within_ss / within_df if within_df > 0 else 0.0

        shift_f = between_var / (within_var + _WITHIN_VAR_EPS)

        max_diff = max(group_means) - min(group_means)
        max_act = max_activation[f] if f < len(max_activation) else 0.0
        shift_maxdiff = (max_diff / max_act) if max_act > 1e-12 else 0.0

        out.append({"feature": f, "shift_f": float(shift_f), "shift_maxdiff": float(shift_maxdiff)})

    return out


def rank_candidates(
    feature_stats: Dict[str, Any],
    shift_stats: Sequence[Dict[str, Any]],
    config: Any,
) -> Dict[str, Any]:
    """Filters and selects the rank stage's final candidate list.

    Filters (all three required): firing density in
    `[config.firing_density_min, config.firing_density_max]`, not dead
    (`is_dead`), and a computed `shift_stats` entry for the feature.

    Selection is the union of two disjoint slices sized off
    `config.screen_features` (the total budget) and
    `config.rank_shift_fraction` (the fraction of the budget from the
    shift score; rounded, then capped to the candidate pool size):

    - `"shift"`: the top slice by `shift_f` descending, ties broken by
      preferring inner shells (smaller `feature_shell` value) and then by
      feature index, for determinism.
    - `"quantile"`: a density-quantile spread (`steer.quantile_indices`,
      the same mechanism `steer.select_steer_features` uses) over the
      *remaining* candidates (those not already shift-selected), so the
      two slices never overlap and the total is exactly the budget
      whenever enough candidates exist.

    Every candidate that makes either slice is a
    `{feature, shift_f, shift_maxdiff, density, shell, selection, rank}`
    dict, where `selection` is `"shift"` or `"quantile"` and `rank` is
    that feature's 1-indexed position in the *full* candidate pool sorted
    by shift score (so a quantile pick's rank shows where it would have
    landed by shift score alone).
    """
    densities = feature_stats["firing_density"]
    max_acts = feature_stats["max_activation"]
    shift_by_feature = {row["feature"]: row for row in shift_stats}
    shells = list(config.matryoshka_shells)

    candidates: List[Dict[str, Any]] = []
    for idx, density in enumerate(densities):
        if is_dead(density):
            continue
        if not (config.firing_density_min <= density <= config.firing_density_max):
            continue
        shift = shift_by_feature.get(idx)
        if shift is None:
            continue
        candidates.append(
            {
                "feature": idx,
                "density": density,
                "max_activation": max_acts[idx] if idx < len(max_acts) else 0.0,
                "shell": feature_shell(idx, shells),
                "shift_f": shift["shift_f"],
                "shift_maxdiff": shift["shift_maxdiff"],
            }
        )

    if not candidates:
        return {"screen_features": [], "shift_budget": 0, "quantile_budget": 0, "candidates_considered": 0}

    ranked_by_shift = sorted(
        candidates,
        key=lambda c: (-c["shift_f"], c["shell"] if c["shell"] is not None else float("inf"), c["feature"]),
    )
    rank_by_feature = {c["feature"]: i + 1 for i, c in enumerate(ranked_by_shift)}

    budget = min(max(0, int(config.screen_features)), len(candidates))
    shift_budget = min(len(candidates), round(budget * config.rank_shift_fraction))
    quantile_budget = budget - shift_budget

    shift_selected = ranked_by_shift[:shift_budget]
    shift_ids = {c["feature"] for c in shift_selected}

    remaining_sorted = sorted(
        (c for c in candidates if c["feature"] not in shift_ids),
        key=lambda c: (c["density"], c["feature"]),
    )
    quantile_idxs = steer.quantile_indices(len(remaining_sorted), quantile_budget)
    quantile_selected = [remaining_sorted[i] for i in quantile_idxs]

    def _record(c: Dict[str, Any], selection: str) -> Dict[str, Any]:
        return {
            "feature": c["feature"],
            "shift_f": c["shift_f"],
            "shift_maxdiff": c["shift_maxdiff"],
            "density": c["density"],
            "shell": c["shell"],
            "selection": selection,
            "rank": rank_by_feature[c["feature"]],
        }

    screen_features = [_record(c, "shift") for c in shift_selected] + [
        _record(c, "quantile") for c in quantile_selected
    ]

    return {
        "screen_features": screen_features,
        "shift_budget": len(shift_selected),
        "quantile_budget": len(quantile_selected),
        "candidates_considered": len(candidates),
    }


def build_rank_records(candidates: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flattens `rank_candidates`' `screen_features` into report records.
    Reported under stage `"train"` with `recordId` prefix `"rank-"` rather
    than a dedicated `"rank"` Worker stage, since
    `orchestrator-worker/src/persona-3c.ts`'s `PERSONA_3C_STAGES` enum does
    not include one and this pipeline change is not meant to require a
    Worker change -- see README "Rank stage"."""
    records = []
    for entry in candidates.get("screen_features", []):
        records.append(
            {"recordId": f"rank-{entry['feature']}-{entry['selection']}", "payload": entry}
        )
    return records


# ---------------------------------------------------------------------------
# Model-touching helpers (need a real transformers model + GPU)
# ---------------------------------------------------------------------------
def build_contexts(config: Any) -> List[Tuple[str, str]]:
    """The diverse context set the shift score is measured across:
    `config.control_prompts`' positive/negative system prompts (named
    `control_<name>_pos`/`_neg`) plus `config.context_prompts` (named
    `context_<i>`). Order is stable (control prompts first, in config
    order, then context prompts in config order) so `capture_context_
    activations`' output is deterministic given the same config."""
    contexts: List[Tuple[str, str]] = []
    for control in config.control_prompts:
        contexts.append((f"control_{control['name']}_pos", control["positive_system_prompt"]))
        contexts.append((f"control_{control['name']}_neg", control["negative_system_prompt"]))
    for i, prompt in enumerate(config.context_prompts):
        contexts.append((f"context_{i}", prompt))
    return contexts


def capture_context_activations(
    config: Any,
    model: Any,
    tokenizer: Any,
    sae: Any,
    contexts: Sequence[Tuple[str, str]],
    scenarios: Sequence[Dict[str, str]],
    device: Any,
) -> Dict[str, Dict[str, List[float]]]:
    """For every (context, scenario) pair: generates a reply under that
    context's system prompt (`screen.generate_with_system_prompt`, the
    same construction the persona-vector controls use), captures the
    layer's residual stream at the generated assistant-token positions via
    a forward hook (the "extra pass with a hook" pattern `steer.
    coherence_logprob`/`screen.capture_mean_residual` already use, kept
    per-token here rather than pooled to residual so `sae.encode` can be
    applied per token before pooling to a per-feature mean), and encodes
    with `sae.encode` -- the same post-batch-topk sparse codes `forward_
    loss` trains on -- then mean-pools the codes over generated tokens.

    Returns `{context_name: {scenario_id: [mean_activation_per_feature,
    ...]}}`. Needs a real transformers model + GPU; not exercised by the
    CPU test suite (mirrors screen.py/steer.py)."""
    base = getattr(model, "model", model)
    out: Dict[str, Dict[str, List[float]]] = {}

    for context_name, system_prompt in contexts:
        per_scenario: Dict[str, List[float]] = {}
        for scenario in scenarios:
            gen = screen.generate_with_system_prompt(
                model, tokenizer, system_prompt, scenario["prompt"], config.max_new_tokens, device
            )
            if not gen["generated_ids"]:
                per_scenario[scenario["id"]] = [0.0] * sae.width
                continue

            full_ids = torch.tensor([list(gen["prompt_ids"]) + list(gen["generated_ids"])], device=device)
            captured: Dict[str, torch.Tensor] = {}

            def hook(module: Any, inputs: Any, output: Any) -> None:
                captured["hidden"] = output[0] if isinstance(output, tuple) else output

            handle = base.layers[config.layer].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    model(input_ids=full_ids)
            finally:
                handle.remove()

            hidden = captured["hidden"][0]
            prompt_len = len(gen["prompt_ids"])
            gen_hidden = hidden[prompt_len:].to(torch.float32)
            with torch.no_grad():
                codes = sae.encode(gen_hidden)
            per_scenario[scenario["id"]] = codes.mean(dim=0).cpu().tolist()

        out[context_name] = per_scenario

    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_rank(
    config: Any,
    model: Any,
    tokenizer: Any,
    sae: Any,
    feature_stats: Dict[str, Any],
    scenarios: Sequence[Dict[str, str]],
    device: Any,
    seed: int = 0,
) -> Dict[str, Any]:
    """Runs the rank stage end to end: builds the diverse context set,
    captures per-(context, scenario) mean SAE-feature activations on
    `scenarios` (the calibrate stage's `config.steer_scenarios`-sized
    slice), computes the label-free shift score per feature, and selects
    `config.screen_features` candidates. `seed` is accepted for interface
    symmetry with `steer.run_calibration`/`screen.run_screen` but unused --
    this stage's only randomness (the quantile slice) is deterministic
    given the candidate pool. Returns the same dict `rank_candidates`
    does; the caller (`run_smoke.stage_rank`) persists/reports it."""
    del seed  # no randomness of our own to seed; kept for call-site symmetry
    contexts = build_contexts(config)
    context_scenario_means = capture_context_activations(config, model, tokenizer, sae, contexts, scenarios, device)
    shift_stats = compute_shift_stats(context_scenario_means, feature_stats["max_activation"])
    return rank_candidates(feature_stats, shift_stats, config)
