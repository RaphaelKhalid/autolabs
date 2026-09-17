"""Rank stage: label-free candidate ranking for calibrate/screen.

Runs after harvest+train (and the post-train check) and before calibrate.
The full run screens the arm-sized total (see `config.arm_sizes`, default
256 full / 8 smoke) of the 32,768 trained SAE features. Picking those
purely by firing-density quantile (what `steer.select_steer_features`
alone does) samples across "how common is this feature" without any
signal about whether a feature's activation tracks anything
persona-relevant at all.

**Claim under test.** The paper this experiment extends lists three
limitations of prompt-based persona-vector discovery: (1) it is
supervised -- the target trait must be specified in advance; (2) it needs
a precise natural-language description of that trait; (3) it needs the
trait to be inducible by prompting at all. Experiment 3C's headline claim
is (1)+(2): unsupervised discovery of persona-relevant SAE directions with
no trait specified and no natural-language description required. Clause
(3) -- prompt reachability -- is not a gate on that finding here; it is a
measured *outcome*, reported per selection arm, because on Qwen the paper
itself observes that most traits worth discovering are prompt-inducible
anyway, so failing to *also* be prompt-reachable is informative, not
disqualifying.

**The problem this module fixes.** Before this change, 75% of screened
candidates were picked by activation shift across hand-written persona
system prompts (`rank_shift_fraction`) -- which biases the *entire*
candidate pool toward prompt-reachable directions and contaminates the
unsupervised-discovery claim: a result built mostly from prompt-shift-
selected features would not actually test (1)+(2) at all. The fix is
three disjoint selection arms (see `rank_candidates`), only one of which
uses any prompt-derived signal.

Three independent pieces of label-free signal feed the arms:

- `compute_shift_stats` (pure numpy/Python, CPU-testable): given each
  context's per-scenario mean SAE-feature activation (already captured
  from the model under a diverse, hand-written set of persona-style system
  prompts), computes, per feature, a one-way-ANOVA-style F-like ratio and
  a normalized max-difference -- the *positive control* for this
  comparison (see "Rank stage" in the README for why), not part of the
  unsupervised claim.
- `compute_specificity_stats` (pure numpy/Python, CPU-testable): given
  per-conversation assistant-vs-user activation aggregates from a plain
  dataset harvest pass (no system prompts, no trait, no persona context of
  any kind), computes `assistant_specificity`, `breadth`, and
  `topic_invariance` -- the signal the `"unsupervised"` arm actually uses.
- `feature_shell` / density / liveness -- structural properties of the
  trained SAE itself, also prompt-free.

`capture_context_activations` and `capture_specificity_activations` need a
real transformers model + GPU and are not exercised by the CPU test suite
(mirroring screen.py/steer.py/harvest.py). `rank_candidates` (pure
dict/list math, CPU-testable) filters by firing density and liveness and
selects three disjoint arms.
"""
from __future__ import annotations

import logging
import math
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import generation
import harvest
import screen
import steer

logger = logging.getLogger("autolabs_3c.rank")

# Added to a denominator before dividing, so a near-zero denominator gives
# a large-but-finite ratio instead of inf/NaN (which can't round-trip
# through JSON -- report.py's `_sanitize` would turn it into null, silently
# dropping the very signal that made the feature interesting).
_WITHIN_VAR_EPS = 1e-8
_SPEC_EPS = 1e-8


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


def feature_shell_index(feature_idx: int, shells: Sequence[int]) -> Optional[int]:
    """0-based index of `feature_shell`'s result within `shells` (0 =
    innermost shell), used for `shell_bonus` and the shift arm's inner-
    shell tie-break. None if `feature_idx` is past every shell."""
    for i, shell in enumerate(shells):
        if feature_idx < shell:
            return i
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


def compute_specificity_stats(
    conversations: Sequence[Dict[str, Any]],
    width: int,
) -> List[Dict[str, Any]]:
    """Per-feature label-free assistant-specificity / breadth / topic-
    invariance statistics from a plain (no system prompt, no trait) dataset
    harvest pass -- the signal the `"unsupervised"` rank arm uses, and the
    only one of the three rank statistics that never sees a persona-style
    prompt of any kind.

    `conversations` is one dict per harvested conversation (see
    `capture_specificity_activations`), each already split into
    assistant-turn and non-assistant ("user") real-token positions by
    `harvest.compute_assistant_mask`:

        {"assistant_mean": [width floats], "assistant_var": [width floats],
         "assistant_fire_rate": [width floats],
         "user_mean": [width floats], "user_fire_rate": [width floats]}

    (`_var`/`_mean`/`_fire_rate` are 0.0 for a feature on a side with zero
    tokens, e.g. a conversation with no assistant turn at all -- shouldn't
    happen given `harvest.extract_messages`' assistant-turn filter, but
    handled defensively.) Every conversation contributes one observation to
    each per-feature average below, regardless of how many tokens it had
    on either side -- this is "how many conversations show this feature
    behaving a given way", not "how many tokens".

    Returns one dict per feature index (0..width-1):

    - `assistant_specificity`: mean-over-conversations assistant-token mean
      activation, divided by the same for user-token mean activation (plus
      a small epsilon) -- how much more this feature fires *as a
      response*, not as a function of whatever the user just said. This is
      the value the `"unsupervised"` arm's composite score uses.
    - `assistant_specificity_fire_rate`: the analogous ratio of mean
      firing *rates* (fraction of tokens where the feature is active at
      all) rather than mean activation magnitude -- a secondary,
      reported-but-not-composited diagnostic.
    - `breadth`: fraction of conversations where the feature fires at
      least once on an assistant-turn token.
    - `topic_invariance`: `1 - (between-conversation variance of each
      firing conversation's own mean assistant-token activation) /
      (that same between-conversation variance + the mean of each firing
      conversation's own within-conversation variance)`, i.e. how much of
      this feature's *total* activation variance (pooled across
      conversations, decomposed the usual ANOVA way into between- and
      within-conversation parts) is "which conversation" rather than
      "token-to-token noise inside one conversation" -- computed only on
      conversations where the feature fired at least once (`breadth`'s
      numerator), since a conversation where it never fires contributes no
      information about how *invariant* its firing is. High
      topic_invariance means the feature behaves similarly regardless of
      what the conversation happens to be about; low means it is bound to
      specific topics/conversations. `0.0` if fewer than 2 conversations
      fired (nothing to compare), `1.0` if the total variance among firing
      conversations is ~0 (perfectly stable, trivially invariant).
    """
    out: List[Dict[str, Any]] = []
    n = len(conversations)

    for f in range(width):
        assistant_means = [c["assistant_mean"][f] for c in conversations]
        assistant_vars = [c["assistant_var"][f] for c in conversations]
        assistant_fire_rates = [c["assistant_fire_rate"][f] for c in conversations]
        user_means = [c["user_mean"][f] for c in conversations]
        user_fire_rates = [c["user_fire_rate"][f] for c in conversations]

        mean_assistant = statistics.fmean(assistant_means) if assistant_means else 0.0
        mean_user = statistics.fmean(user_means) if user_means else 0.0
        mean_assistant_fr = statistics.fmean(assistant_fire_rates) if assistant_fire_rates else 0.0
        mean_user_fr = statistics.fmean(user_fire_rates) if user_fire_rates else 0.0

        assistant_specificity = mean_assistant / (mean_user + _SPEC_EPS)
        assistant_specificity_fire_rate = mean_assistant_fr / (mean_user_fr + _SPEC_EPS)

        fired_flags = [fr > 0.0 for fr in assistant_fire_rates]
        breadth = (sum(fired_flags) / n) if n else 0.0

        firing_idxs = [i for i, fired in enumerate(fired_flags) if fired]
        if len(firing_idxs) >= 2:
            firing_means = [assistant_means[i] for i in firing_idxs]
            firing_vars = [assistant_vars[i] for i in firing_idxs]
            between_var = statistics.pvariance(firing_means)
            within_var_mean = statistics.fmean(firing_vars)
            total_var = between_var + within_var_mean
            topic_invariance = 1.0 - (between_var / total_var) if total_var > _SPEC_EPS else 1.0
        else:
            topic_invariance = 0.0

        out.append(
            {
                "feature": f,
                "assistant_specificity": float(assistant_specificity),
                "assistant_specificity_fire_rate": float(assistant_specificity_fire_rate),
                "breadth": float(breadth),
                "topic_invariance": float(topic_invariance),
            }
        )

    return out


def zscore(values: Sequence[float]) -> List[float]:
    """Population z-score (ddof=0) of `values` against their own mean/std.
    All-zero (every value equal, or fewer than 2 values) rather than NaN
    when the population has ~0 variance -- a constant signal carries no
    ranking information, not an infinite one."""
    if not values:
        return []
    mean = statistics.fmean(values)
    if len(values) < 2:
        return [0.0 for _ in values]
    variance = statistics.pvariance(values, mu=mean)
    std = math.sqrt(variance)
    if std < _SPEC_EPS:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def shell_bonus(shell_index: Optional[int]) -> float:
    """Composite-score bonus for the unsupervised arm: 1.0 for the
    innermost matryoshka shell (index 0), 0.5 for the second, 0.0
    otherwise -- the matryoshka training objective forces inner-shell
    features to carry a self-contained coarse reconstruction rather than
    only refining an outer one, making them the more likely place for a
    coarse, generalizable direction to live (same rationale the old
    shift-score shell tie-break used)."""
    if shell_index == 0:
        return 1.0
    if shell_index == 1:
        return 0.5
    return 0.0


def compute_composite_scores(candidates: Sequence[Dict[str, Any]]) -> None:
    """In place: adds a `"composite"` key to every dict in `candidates`
    (each already carrying `assistant_specificity`, `breadth`,
    `topic_invariance`, and `shell_index`), the score the `"unsupervised"`
    rank arm sorts by:

        composite = z(log assistant_specificity) + z(breadth)
                    + z(topic_invariance) + shell_bonus(shell_index)

    No prompt-derived quantity (`shift_f`/`shift_maxdiff`) enters this
    score at all -- that is the point of this arm, see the module
    docstring's "Claim under test". Every z-score is computed against the
    *candidate pool passed in* (already density/dead/shift/specificity-
    filtered by the caller), so the score is relative to this run's actual
    candidate distribution, not some universal scale.
    """
    log_specificity = [math.log(max(c["assistant_specificity"], _SPEC_EPS)) for c in candidates]
    breadth = [c["breadth"] for c in candidates]
    topic_invariance = [c["topic_invariance"] for c in candidates]

    z_specificity = zscore(log_specificity)
    z_breadth = zscore(breadth)
    z_topic_invariance = zscore(topic_invariance)

    for c, zs, zb, zt in zip(candidates, z_specificity, z_breadth, z_topic_invariance):
        c["composite"] = zs + zb + zt + shell_bonus(c.get("shell_index"))


def rank_candidates(
    feature_stats: Dict[str, Any],
    shift_stats: Sequence[Dict[str, Any]],
    specificity_stats: Sequence[Dict[str, Any]],
    config: Any,
) -> Dict[str, Any]:
    """Filters and selects the rank stage's final candidate list as three
    disjoint arms (`config.arm_sizes`, e.g. `{"unsupervised": 96,
    "quantile": 64, "shift": 96}` full / `{"unsupervised": 3, "quantile":
    2, "shift": 3}` smoke).

    Filters (all required): firing density in `[config.firing_density_min,
    config.firing_density_max]`, not dead (`is_dead`), a computed
    `shift_stats` entry, and a computed `specificity_stats` entry for the
    feature.

    Arms are filled in order -- `"unsupervised"` first, then `"quantile"`,
    then `"shift"` -- each drawn from whatever the previous arms didn't
    already take, so the three are always disjoint. Each arm's size is
    capped to however many candidates remain when its turn comes (e.g. a
    tiny candidate pool may leave the `"shift"` arm short of its
    configured size, or empty):

    - `"unsupervised"`: top slice by the composite score
      (`compute_composite_scores` -- assistant-specificity, breadth, and
      topic-invariance from a plain dataset harvest, plus a shell bonus;
      no prompt-derived quantity), ties broken by feature index.
    - `"quantile"`: a density-quantile spread (`steer.quantile_indices`)
      over the pool remaining after the unsupervised arm's picks are
      removed -- an unbiased-by-construction comparison slice.
    - `"shift"`: the top slice by `shift_f` descending (ties broken by
      preferring inner shells, then feature index) from whatever's left
      after both other arms -- the positive control for this comparison,
      since it *is* biased toward directions that shift under the
      hand-written persona-style system prompts (see module docstring).

    Every selected candidate is a dict carrying every computed statistic
    (`density`, `shell`, `shift_f`, `shift_maxdiff`, `assistant_specificity`,
    `assistant_specificity_fire_rate`, `breadth`, `topic_invariance`,
    `composite`) plus `arm` (`"unsupervised"`/`"quantile"`/`"shift"`),
    `selection` (kept equal to `arm`, for callers written against the old
    two-arm schema's field name), and `rank_within_arm` (1-indexed position
    within that arm's own selection order).
    """
    densities = feature_stats["firing_density"]
    max_acts = feature_stats["max_activation"]
    shift_by_feature = {row["feature"]: row for row in shift_stats}
    specificity_by_feature = {row["feature"]: row for row in specificity_stats}
    shells = list(config.matryoshka_shells)
    arm_sizes = dict(config.arm_sizes)

    candidates: List[Dict[str, Any]] = []
    for idx, density in enumerate(densities):
        if is_dead(density):
            continue
        if not (config.firing_density_min <= density <= config.firing_density_max):
            continue
        shift = shift_by_feature.get(idx)
        specificity = specificity_by_feature.get(idx)
        if shift is None or specificity is None:
            continue
        candidates.append(
            {
                "feature": idx,
                "density": density,
                "max_activation": max_acts[idx] if idx < len(max_acts) else 0.0,
                "shell": feature_shell(idx, shells),
                "shell_index": feature_shell_index(idx, shells),
                "shift_f": shift["shift_f"],
                "shift_maxdiff": shift["shift_maxdiff"],
                "assistant_specificity": specificity["assistant_specificity"],
                "assistant_specificity_fire_rate": specificity["assistant_specificity_fire_rate"],
                "breadth": specificity["breadth"],
                "topic_invariance": specificity["topic_invariance"],
            }
        )

    if not candidates:
        return {
            "screen_features": [],
            "arm_counts": {arm: 0 for arm in arm_sizes},
            "candidates_considered": 0,
        }

    compute_composite_scores(candidates)

    remaining = list(candidates)

    # -- unsupervised arm: top by composite, no prompt-derived quantity ---
    unsupervised_budget = min(max(0, int(arm_sizes.get("unsupervised", 0))), len(remaining))
    ranked_unsupervised = sorted(remaining, key=lambda c: (-c["composite"], c["feature"]))
    unsupervised_selected = ranked_unsupervised[:unsupervised_budget]
    unsupervised_ids = {c["feature"] for c in unsupervised_selected}
    remaining = [c for c in remaining if c["feature"] not in unsupervised_ids]

    # -- quantile arm: density-quantile spread over what's left -----------
    quantile_budget = min(max(0, int(arm_sizes.get("quantile", 0))), len(remaining))
    remaining_by_density = sorted(remaining, key=lambda c: (c["density"], c["feature"]))
    quantile_idxs = steer.quantile_indices(len(remaining_by_density), quantile_budget)
    quantile_selected = [remaining_by_density[i] for i in quantile_idxs]
    quantile_ids = {c["feature"] for c in quantile_selected}
    remaining = [c for c in remaining if c["feature"] not in quantile_ids]

    # -- shift arm: top by shift_f from whatever's left --------------------
    shift_budget = min(max(0, int(arm_sizes.get("shift", 0))), len(remaining))
    ranked_shift = sorted(
        remaining,
        key=lambda c: (
            -c["shift_f"],
            c["shell_index"] if c["shell_index"] is not None else float("inf"),
            c["feature"],
        ),
    )
    shift_selected = ranked_shift[:shift_budget]

    def _record(c: Dict[str, Any], arm: str, rank_within_arm: int) -> Dict[str, Any]:
        return {
            "feature": c["feature"],
            "density": c["density"],
            "shell": c["shell"],
            "shift_f": c["shift_f"],
            "shift_maxdiff": c["shift_maxdiff"],
            "assistant_specificity": c["assistant_specificity"],
            "assistant_specificity_fire_rate": c["assistant_specificity_fire_rate"],
            "breadth": c["breadth"],
            "topic_invariance": c["topic_invariance"],
            "composite": c["composite"],
            "arm": arm,
            "selection": arm,
            "rank_within_arm": rank_within_arm,
        }

    screen_features = (
        [_record(c, "unsupervised", i + 1) for i, c in enumerate(unsupervised_selected)]
        + [_record(c, "quantile", i + 1) for i, c in enumerate(quantile_selected)]
        + [_record(c, "shift", i + 1) for i, c in enumerate(shift_selected)]
    )

    return {
        "screen_features": screen_features,
        "arm_counts": {
            "unsupervised": len(unsupervised_selected),
            "quantile": len(quantile_selected),
            "shift": len(shift_selected),
        },
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
            {"recordId": f"rank-{entry['feature']}-{entry['arm']}", "payload": entry}
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
    context's system prompt (batched via `generation.generate_batch`, the
    same construction the persona-vector controls use), captures the
    layer's residual stream at the generated assistant-token positions via
    a batched forward-hook pass (`screen.capture_generated_hidden_batch`,
    kept per-token here rather than pooled to residual so `sae.encode` can
    be applied per token before pooling to a per-feature mean), and encodes
    with `sae.encode` -- the same post-batch-topk sparse codes `forward_
    loss` trains on -- then mean-pools the codes over generated tokens.

    Returns `{context_name: {scenario_id: [mean_activation_per_feature,
    ...]}}`. Needs a real transformers model + GPU; not exercised by the
    CPU test suite (mirrors screen.py/steer.py). Feeds `compute_shift_
    stats` -- the positive-control statistic, not the unsupervised arm's
    signal (see `capture_specificity_activations` for that).

    Batched: every (context, scenario) pair is generated in one
    `generation.generate_batch` call (chunked at
    `config.generation_batch_size`), and the layer's residual stream at
    each row's own generated-token positions is captured in one further
    batched forward pass (`screen.capture_generated_hidden_batch`) before
    `sae.encode` + mean-pooling per row."""
    system_prompt_by_name = dict(contexts)
    combos = [(context_name, scenario) for context_name, _sp in contexts for scenario in scenarios]
    msgs = [
        [
            {"role": "system", "content": system_prompt_by_name[context_name]},
            {"role": "user", "content": scenario["prompt"]},
        ]
        for context_name, scenario in combos
    ]
    gens = generation.generate_batch(
        model, tokenizer, msgs, config.max_new_tokens, device, hook=None, batch_size=config.generation_batch_size
    )
    hiddens = screen.capture_generated_hidden_batch(
        model,
        config.layer,
        [g["prompt_ids"] for g in gens],
        [g["generated_ids"] for g in gens],
        device,
    )

    out: Dict[str, Dict[str, List[float]]] = {}
    for (context_name, scenario), hidden in zip(combos, hiddens):
        if hidden.shape[0] == 0:
            feature_means = [0.0] * sae.width
        else:
            with torch.no_grad():
                codes = sae.encode(hidden.to(torch.float32))
            feature_means = codes.mean(dim=0).cpu().tolist()
        out.setdefault(context_name, {})[scenario["id"]] = feature_means

    return out


def _prepare_specificity_batch(tokenizer: Any, convs: Sequence[Sequence[Dict[str, str]]], config: Any, device: Any):
    """Pads a chunk of conversations to a dense batch, keeping *every* real
    token (not just assistant-turn ones, unlike `run_smoke._prepare_batch`)
    plus the per-position assistant/non-assistant label from
    `harvest.compute_assistant_mask`. Returns `None` if every conversation
    in the chunk has zero assistant-turn tokens (nothing usable)."""
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
    batch_assistant_mask = torch.zeros((len(prepared), max_len), dtype=torch.long, device=device)
    for i, (ids, mask) in enumerate(prepared):
        n = len(ids)
        batch_ids[i, :n] = torch.tensor(ids, device=device)
        batch_attn[i, :n] = 1
        batch_assistant_mask[i, : len(mask)] = torch.tensor(mask, device=device)
    return batch_ids, batch_attn, batch_assistant_mask


def capture_specificity_activations(
    config: Any,
    model: Any,
    tokenizer: Any,
    sae: Any,
    device: Any,
) -> List[Dict[str, Any]]:
    """Extra harvest pass (`config.rank_specificity_conversations`, ~300)
    over the same dataset stream `harvest.py` trains on, feeding
    `compute_specificity_stats` -- the `"unsupervised"` rank arm's signal.

    Unlike the training harvest (assistant-turn-only mask), this keeps
    every real (non-padding) token position -- the FULL token mask the
    task calls for -- and uses `harvest.compute_assistant_mask` only to
    *label* each position as assistant-turn content or not; every other
    real position (user-turn content, and chat-template control tokens
    like the assistant header) is treated as the "user" side of the
    assistant/user contrast this stage draws. No system prompt, no trait,
    no persona-style context of any kind is involved -- this is the one
    rank-stage signal that never sees a prompt.

    Truncates the model to `config.layer` with `harvest.
    ActivationHarvester` (same as training; restored on exit, so this is
    safe to call with the full model already reloaded for steering, same
    as `run_smoke.stage_harvest_train` already relies on), and encodes the
    captured residual with `sae.encode` (the same post-batch-topk sparse
    codes `forward_loss` trains on) before splitting into assistant/user
    per-feature mean, variance, and firing-rate per conversation.

    Needs a real transformers model + GPU; not exercised by the CPU test
    suite (mirrors harvest.py/screen.py/steer.py)."""
    d_model = model.config.hidden_size
    conversations_needed = config.rank_specificity_conversations
    per_conversation: List[Dict[str, Any]] = []

    conv_stream = harvest.stream_conversations(config.dataset_name, config.dataset_split, seed=config.seed + 7)
    with harvest.ActivationHarvester(model, config.layer) as harvester:
        batch_msgs: List[List[Dict[str, str]]] = []
        for messages in conv_stream:
            if len(per_conversation) >= conversations_needed:
                break
            batch_msgs.append(messages)
            if len(batch_msgs) < config.harvest_batch_size:
                continue

            batch = _prepare_specificity_batch(tokenizer, batch_msgs, config, device)
            batch_msgs = []
            if batch is None:
                continue
            batch_ids, batch_attn, batch_assistant_mask = batch

            hidden = harvester.forward(batch_ids, attention_mask=batch_attn)
            with torch.no_grad():
                flat = hidden.reshape(-1, d_model).to(torch.float32)
                codes_flat = sae.encode(flat)
            codes = codes_flat.view(hidden.shape[0], hidden.shape[1], sae.width)

            for i in range(batch_ids.shape[0]):
                real = batch_attn[i].bool()
                is_assistant = batch_assistant_mask[i].bool() & real
                is_user = real & (~batch_assistant_mask[i].bool())
                assistant_codes = codes[i][is_assistant]
                user_codes = codes[i][is_user]
                per_conversation.append(
                    {
                        "assistant_mean": (
                            assistant_codes.mean(dim=0).cpu().tolist() if assistant_codes.shape[0] else [0.0] * sae.width
                        ),
                        "assistant_var": (
                            assistant_codes.var(dim=0, unbiased=False).cpu().tolist()
                            if assistant_codes.shape[0]
                            else [0.0] * sae.width
                        ),
                        "assistant_fire_rate": (
                            (assistant_codes > 0).float().mean(dim=0).cpu().tolist()
                            if assistant_codes.shape[0]
                            else [0.0] * sae.width
                        ),
                        "user_mean": (
                            user_codes.mean(dim=0).cpu().tolist() if user_codes.shape[0] else [0.0] * sae.width
                        ),
                        "user_fire_rate": (
                            (user_codes > 0).float().mean(dim=0).cpu().tolist() if user_codes.shape[0] else [0.0] * sae.width
                        ),
                    }
                )

    return per_conversation[:conversations_needed]


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
    slice) for the shift-arm positive control, runs the extra plain-
    dataset harvest pass for the unsupervised arm's assistant-specificity/
    breadth/topic-invariance statistics, and selects the three-arm
    candidate list (`rank_candidates`). `seed` is accepted for interface
    symmetry with `steer.run_calibration`/`screen.run_screen` but unused --
    this stage's only randomness (the quantile slice) is deterministic
    given the candidate pool. Returns the same dict `rank_candidates`
    does; the caller (`run_smoke.stage_rank`) persists/reports it."""
    del seed  # no randomness of our own to seed; kept for call-site symmetry
    contexts = build_contexts(config)
    context_scenario_means = capture_context_activations(config, model, tokenizer, sae, contexts, scenarios, device)
    shift_stats = compute_shift_stats(context_scenario_means, feature_stats["max_activation"])

    specificity_conversations = capture_specificity_activations(config, model, tokenizer, sae, device)
    specificity_stats = compute_specificity_stats(specificity_conversations, sae.width)

    return rank_candidates(feature_stats, shift_stats, specificity_stats, config)
