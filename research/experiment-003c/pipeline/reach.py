"""Reach stage: can prompting reproduce a named direction's steered effect?

The paper this experiment extends lists clause (3) of prompt-based persona-
vector discovery's limitations as "the trait must be inducible by
prompting". Experiment 3C's headline claim is (1)+(2) -- unsupervised
discovery with no target trait and no natural-language description -- and
clause (3) is deliberately *not* a gate on that finding (see pipeline/
README.md "Rank stage" / "Claim under test"): a direction the unsupervised
arm finds that turns out not to be prompt-reachable is an interesting
result in its own right, not evidence the pipeline failed. This module
reports (3) as a measured **outcome**, per selection arm, for whichever
directions the describe stage (describe.py) named -- plus every persona-
vector control that was named, which are this test's *positive controls*
and are expected to come out reachable regardless of arm.

For each such direction, four system prompts are built locally (no judge
call) from the describe stage's own `cluster_property` text `P` -- the
blinded judge's clustered description of what differs about the steered
condition -- and used, unsteered, on the model:

- ``direct``:       "In your replies, {P}."
- ``rewrite``:       "Adopt this manner throughout: {P}. Keep it natural
  and consistent."
- ``intensified``:   ``rewrite`` + " Do this strongly and in every reply."
- ``fewshot``:       ``rewrite``'s system prompt, followed by two
  demonstration exchanges (scenario prompt -> that direction's own steered
  reply) taken from two of the direction's screened scenarios, which are
  then *excluded* from that direction's evaluation set for every variant
  that skips them intentionally -- fewshot alone, so the same steered
  replies used as demonstrations can never also be scored as evidence the
  demonstration worked. ``direct``/``rewrite``/``intensified`` are scored
  on every scenario the direction was screened on; ``fewshot`` is scored
  on that set minus its own two demo scenarios.

**Behavioral reach** reuses the direction's own steered-vs-baseline
screen data: a fresh logistic classifier (the same numpy-only mechanism
``screen.leave_one_scenario_out_separability`` fits internally, but fit
once on *all* scenarios here rather than leave-one-out, since the
prompted replies -- not a held-out steered/baseline scenario -- are what
needs scoring) for both the residual and lexical views. ``effect_fraction``
rescales the prompted replies' mean classifier logit onto the
baseline->steered scale (0 = no effect, 1 = matches steering, clipped to
[-0.5, 1.5] so a wrong-direction or overshooting effect stays visible
rather than exploding); ``axis_cosine`` (residual view only) is the cosine
between the prompted-vs-baseline and steered-vs-baseline mean residual
diff vectors, an orientation check independent of magnitude. The best
variant is whichever maximizes the residual-view effect_fraction.

**Mechanistic reach** asks whether the same mechanism moved: for a
feature direction, the mean activation of that feature's own SAE code at
the generated tokens (``sae.encode`` on the layer's residual stream),
rescaled the same baseline->steered way (``feature_fraction``); a persona-
vector control has no single feature, so its mechanism proxy is the
projection onto that control's own steered-minus-baseline mean residual
axis instead.

**Judge check** (optional, ``config.reach_judge``) asks the Worker's
blinded judge (the same ``/api/persona-3c/judge/*`` queue describe.py
drives) to compare the best variant's prompted replies against the
direction's own steered replies, on the same scenarios, both pair orders
-- ``judge_steered_share`` is the fraction of pairs the judge says the
*steered* side shows more of `P` (0.5 means prompting reproduced it as
well as steering did, from the judge's perspective).

Everything through variant/effect/feature math is pure numpy/Python and
CPU-testable (mirrors screen.py/describe.py's split); the generation,
residual/SAE capture, and judge-driving calls need a real model (and, for
the judge check, the Worker) and are not exercised by the CPU test suite.
"""
from __future__ import annotations

import html
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import describe
import generation
import screen
from report import WorkerClient, _progress

logger = logging.getLogger("autolabs_3c.reach")

VARIANTS = ("direct", "rewrite", "intensified", "fewshot")
N_FEWSHOT_DEMOS = 2
MIN_SCENARIOS_NEEDED = N_FEWSHOT_DEMOS + 1  # >=1 left to evaluate after excluding demos
# Fixed judge-check cutoff (not a config knob): part of the `reachable`
# definition alongside config.reach_effect_threshold, only consulted when
# a direction was actually judged (config.reach_judge).
JUDGE_STEERED_SHARE_MAX = 0.65
CONTROL_KEY_PREFIX = "control-"


def is_control_key(direction_key_value: str) -> bool:
    return direction_key_value.startswith(CONTROL_KEY_PREFIX)


# ---------------------------------------------------------------------------
# Prompt construction (pure Python/string; CPU-testable)
# ---------------------------------------------------------------------------
def build_system_prompt(variant: str, property_text: str) -> str:
    """The three plain-string variants, plus the system prompt `fewshot`
    shares with `rewrite` (its distinguishing demonstration turns are added
    separately by `build_fewshot_messages`, since they are conversation
    turns, not more system-prompt text)."""
    rewrite = f"Adopt this manner throughout: {property_text}. Keep it natural and consistent."
    if variant == "direct":
        return f"In your replies, {property_text}."
    if variant == "rewrite" or variant == "fewshot":
        return rewrite
    if variant == "intensified":
        return rewrite + " Do this strongly and in every reply."
    raise ValueError(f"unknown reach prompt variant: {variant!r}")


def select_fewshot_demo_scenarios(scenario_ids: Sequence[str], n: int = N_FEWSHOT_DEMOS) -> List[str]:
    """Deterministic pick of `n` scenario ids to use as few-shot
    demonstrations: sorted ascending, first `n`. Sorted (not e.g. random)
    so a resumed/re-run reach stage picks the exact same demos every time
    without needing to persist a choice."""
    return sorted(scenario_ids)[:n]


def build_variant_messages(
    variant: str,
    property_text: str,
    eval_prompt: str,
    demo_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Builds the chat message list for one (variant, scenario) generation.
    `demo_pairs` (only used/required for `variant="fewshot"`) is a sequence
    of (demo_scenario_prompt, demo_steered_reply) taken from the
    direction's *own* steered generations on scenarios excluded from
    `eval_prompt`'s evaluation set -- see module docstring."""
    system_prompt = build_system_prompt(variant, property_text)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if variant == "fewshot":
        for demo_prompt, demo_reply in demo_pairs or []:
            messages.append({"role": "user", "content": demo_prompt})
            messages.append({"role": "assistant", "content": demo_reply})
    messages.append({"role": "user", "content": eval_prompt})
    return messages


# ---------------------------------------------------------------------------
# Behavioral/mechanistic reach math (pure numpy; CPU-testable)
# ---------------------------------------------------------------------------
def effect_fraction(
    prompted_mean: float,
    baseline_mean: float,
    steered_mean: float,
    clip_min: float = -0.5,
    clip_max: float = 1.5,
    eps: float = 1e-8,
) -> float:
    """Rescales `prompted_mean` onto the [baseline_mean, steered_mean]
    scale: 0.0 if prompting looks like baseline, 1.0 if it looks exactly
    like steering, clipped to [clip_min, clip_max] so a wrong-direction
    (negative) or overshooting (>1) effect stays a finite, visible number
    instead of exploding. `nan` if the direction has (near) no
    baseline-vs-steered separation to rescale onto in the first place
    (denominator ~0)."""
    denom = steered_mean - baseline_mean
    if abs(denom) < eps:
        return float("nan")
    value = (prompted_mean - baseline_mean) / denom
    return float(min(clip_max, max(clip_min, value)))


def feature_fraction(prompted: float, baseline: float, steered: float, eps: float = 1e-8) -> float:
    """Same rescaling as `effect_fraction` but for the mechanistic check
    (a single SAE feature's mean activation, or a control's projection
    onto its own persona axis) -- unclipped, since an activation/projection
    has no natural [-0.5, 1.5] behavioral-effect interpretation to bound
    it to; `nan` on a (near-)zero denominator, same as `effect_fraction`."""
    denom = steered - baseline
    if abs(denom) < eps:
        return float("nan")
    return float((prompted - baseline) / denom)


def axis_cosine(prompted_vec: np.ndarray, baseline_vec: np.ndarray, steered_vec: np.ndarray) -> float:
    """Cosine similarity between (prompted - baseline) and
    (steered - baseline) mean residual vectors: an orientation check
    independent of `effect_fraction`'s magnitude rescaling."""
    return screen.cosine_similarity(
        np.asarray(prompted_vec, dtype=np.float64) - np.asarray(baseline_vec, dtype=np.float64),
        np.asarray(steered_vec, dtype=np.float64) - np.asarray(baseline_vec, dtype=np.float64),
    )


def fit_classifier(
    steered_vecs: Dict[str, np.ndarray], baseline_vecs: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fits `screen.train_logreg` on *every* scenario's steered+baseline
    embedding (unlike the screen stage's own leave-one-scenario-out split
    -- there is no held-out fold to protect here, since what's being
    scored next is a prompted reply, never one of this classifier's own
    training points). Returns `(w, b, mu, sigma)`; `sigma` includes the
    screen stage's own +1e-8 floor against a zero-variance dimension."""
    scenarios = sorted(set(steered_vecs) & set(baseline_vecs))
    X = np.stack([steered_vecs[s] for s in scenarios] + [baseline_vecs[s] for s in scenarios])
    y = np.array([1] * len(scenarios) + [0] * len(scenarios))
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    w, b = screen.train_logreg((X - mu) / sigma, y)
    return w, b, mu, sigma


def classifier_logit(vec: np.ndarray, w: np.ndarray, b: float, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Raw linear score (pre-sigmoid) of `vec` under a classifier fit by
    `fit_classifier` -- `effect_fraction` is computed on this logit, not on
    a 0..1 probability, so it isn't compressed by the sigmoid near either
    end."""
    x_n = (np.asarray(vec, dtype=np.float64) - mu) / sigma
    return float(x_n @ w + b)


def mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# Best-variant selection, classification, arm summary (pure Python; CPU-testable)
# ---------------------------------------------------------------------------
def pick_best_variant(variant_results: Dict[str, Dict[str, float]]) -> Optional[str]:
    """The variant with the highest `effect_fraction_resid` (nan-safe: a
    variant whose behavioral effect could not be computed at all never
    wins over one that could). `None` if every variant is nan/missing."""
    best_key: Optional[str] = None
    best_value = float("-inf")
    for key, result in variant_results.items():
        value = result.get("effect_fraction_resid")
        if value is None or not np.isfinite(value):
            continue
        if value > best_value:
            best_value = value
            best_key = key
    return best_key


def classify_reachability(
    best_effect_fraction: float,
    judge_steered_share: Optional[float],
    effect_threshold: float,
    not_threshold: float,
    judge_max: float = JUDGE_STEERED_SHARE_MAX,
) -> str:
    """`reachable` if the best variant's residual effect_fraction clears
    `effect_threshold` and, only when a judge check actually ran
    (`judge_steered_share is not None`), the judge didn't still favor the
    steered side more than `judge_max`. `not_reachable` if the best
    variant -- which already includes `fewshot` in the max `pick_best_
    variant` takes -- falls below `not_threshold`. Otherwise `partial`.
    `nan` (no direction had a usable behavioral signal at all) satisfies
    neither comparison (`nan` comparisons are always False in Python/
    numpy) and safely falls through to `partial`."""
    if best_effect_fraction >= effect_threshold and (judge_steered_share is None or judge_steered_share <= judge_max):
        return "reachable"
    if best_effect_fraction < not_threshold:
        return "not_reachable"
    return "partial"


def mechanism_same(best_feature_fraction: float, feature_threshold: float) -> bool:
    """`nan` (no usable mechanistic signal) is never >= a real threshold,
    so this is `False` by construction on a degenerate denominator."""
    return bool(np.isfinite(best_feature_fraction) and best_feature_fraction >= feature_threshold)


def select_reach_directions(
    clusters: Dict[str, Dict[str, Any]], max_directions: int
) -> List[Tuple[str, Dict[str, Any]]]:
    """Selects the directions the describe stage `named` (excludes
    `describe.is_null_key` random-direction nulls, which were never
    candidates for a persona-relevance claim in the first place), highest
    `consistency_score` first, capped at `max_directions` -- *except* every
    named persona-vector control (`is_control_key`) is always included on
    top of that cap: they are this test's positive controls (expected to
    come out `reachable`), not part of the budget the cap is rationing
    across the unsupervised/quantile/shift arms. Returns
    `[(directionKey, cluster_summary), ...]`, which may exceed
    `max_directions` in length exactly when a named control would
    otherwise have been cut."""
    named = [
        (key, summary)
        for key, summary in clusters.items()
        if summary.get("named") and not describe.is_null_key(key)
    ]
    ranked = sorted(named, key=lambda kv: kv[1].get("consistency_score") or 0.0, reverse=True)
    top = ranked[: max(0, max_directions)]
    top_keys = {key for key, _ in top}
    for key, summary in ranked:
        if is_control_key(key) and key not in top_keys:
            top.append((key, summary))
            top_keys.add(key)
    return top


def summarize_by_arm(direction_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts of `reachable_class` per arm (including `"control"`), plus
    the control pass rate (`reachable_class == "reachable"` fraction among
    `arm == "control"` rows) -- controls are this test's positive controls
    and are expected to come out reachable, so a low control pass rate
    says more about this reach methodology than about any one direction."""
    by_arm: Dict[str, Dict[str, int]] = {}
    for row in direction_results:
        arm = row.get("arm") or "unknown"
        entry = by_arm.setdefault(arm, {"reachable": 0, "partial": 0, "not_reachable": 0, "total": 0})
        cls = row.get("reachable_class")
        if cls in entry:
            entry[cls] += 1
        entry["total"] += 1

    controls = [row for row in direction_results if row.get("arm") == "control"]
    control_pass_rate = (
        sum(1 for row in controls if row.get("reachable_class") == "reachable") / len(controls)
        if controls
        else None
    )
    return {"by_arm": by_arm, "control_pass_rate": control_pass_rate, "n_controls": len(controls)}


# ---------------------------------------------------------------------------
# Model-touching helpers (need a real transformers model; not CPU-tested)
# ---------------------------------------------------------------------------
def retokenize_for_capture(tokenizer: Any, scenario_prompt: str, text: str) -> Tuple[List[int], List[int]]:
    """Approximates the token ids behind a screen-stage baseline/steered
    generation: `screen_generations.json` stores only the decoded text, not
    the original `prompt_ids`/`generated_ids`. Retokenizes the single-
    user-message prompt exactly as `screen.generate_baseline_set`/
    `generate_steered_set` built it, and the decoded text as a fresh token
    sequence. Not guaranteed byte-identical to the original generated ids
    (detokenize -> retokenize is not a perfect round trip), but close
    enough for a mean-pooled residual capture -- the same numerical-noise
    caveat as README 'Batching and generation numerics'."""
    messages = [{"role": "user", "content": scenario_prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    generated_ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    return prompt_ids, generated_ids


def capture_direction_features(
    model: Any,
    sae: Any,
    layer: int,
    prompt_ids_list: Sequence[Sequence[int]],
    generated_ids_list: Sequence[Sequence[int]],
    device: Any,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """One forward pass (`screen.capture_generated_hidden_batch`) reused for
    both the behavioral view (mean-pooled residual vector per row) and the
    mechanistic view (mean per-feature SAE code vector per row, via
    `sae.encode`) -- avoids capturing the same generation's activations
    twice for the two checks."""
    import torch  # local import: only needed on this model-touching path

    hiddens = screen.capture_generated_hidden_batch(model, layer, prompt_ids_list, generated_ids_list, device)
    d_model = model.config.hidden_size
    resid_vecs: List[np.ndarray] = []
    code_vecs: List[np.ndarray] = []
    for h in hiddens:
        if h.shape[0] == 0:
            resid_vecs.append(np.zeros(d_model, dtype=np.float64))
            code_vecs.append(np.zeros(sae.width, dtype=np.float64))
            continue
        resid_vecs.append(h.float().mean(dim=0).cpu().numpy().astype(np.float64))
        with torch.no_grad():
            codes = sae.encode(h.to(sae.W_enc.dtype))
        code_vecs.append(codes.float().mean(dim=0).cpu().numpy().astype(np.float64))
    return resid_vecs, code_vecs


def _direction_lookup(screen_directions: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        describe.direction_key(row["kind"], row["id"], row["sign"]): row for row in screen_directions
    }


def _generations_by_direction(
    screen_generations: Sequence[Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """{directionKey: {scenario_id: {"text": steered, "baseline_text": baseline}}}."""
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in screen_generations:
        key = describe.direction_key(row["kind"], row["id"], row["sign"])
        out.setdefault(key, {})[row["scenario"]] = {
            "text": row.get("text") or "",
            "baseline_text": row.get("baseline_text") or "",
        }
    return out


def _run_variant(
    variant: str,
    info: Dict[str, Any],
    property_text: str,
    eval_ids: List[str],
    demo_ids: List[str],
    gens: Dict[str, Dict[str, str]],
    scenario_prompt_by_id: Dict[str, str],
    model: Any,
    tokenizer: Any,
    sae: Any,
    layer: int,
    max_new_tokens: int,
    generation_batch_size: int,
    device: Any,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Generates + scores one prompt variant on `eval_ids`. Returns
    `(metrics, extra)` where `extra` carries the prompted generations
    (`{scenario_id: {"text":..., "prompt_ids":..., "generated_ids":...}}`),
    kept for the optional judge check on the best variant only."""
    demo_pairs = [(scenario_prompt_by_id[d], gens[d]["text"]) for d in demo_ids] if variant == "fewshot" else None
    messages_list = [
        build_variant_messages(variant, property_text, scenario_prompt_by_id[sid], demo_pairs) for sid in eval_ids
    ]
    prompted_gens = generation.generate_batch(
        model, tokenizer, messages_list, max_new_tokens, device, hook=None, batch_size=generation_batch_size
    )
    prompted_by_scenario = dict(zip(eval_ids, prompted_gens))

    # -- lexical view: pure text, no model needed -------------------------
    lexical_prompted = {sid: screen.hashed_lexical_vector(g["text"]) for sid, g in prompted_by_scenario.items()}
    lexical_baseline = {sid: screen.hashed_lexical_vector(gens[sid]["baseline_text"]) for sid in eval_ids}
    lexical_steered = {sid: screen.hashed_lexical_vector(gens[sid]["text"]) for sid in eval_ids}
    w_l, b_l, mu_l, sigma_l = fit_classifier(lexical_steered, lexical_baseline)
    logit_prompted_l = mean_or_nan([classifier_logit(lexical_prompted[sid], w_l, b_l, mu_l, sigma_l) for sid in eval_ids])
    logit_baseline_l = mean_or_nan([classifier_logit(lexical_baseline[sid], w_l, b_l, mu_l, sigma_l) for sid in eval_ids])
    logit_steered_l = mean_or_nan([classifier_logit(lexical_steered[sid], w_l, b_l, mu_l, sigma_l) for sid in eval_ids])
    ef_lex = effect_fraction(logit_prompted_l, logit_baseline_l, logit_steered_l)

    # -- residual + SAE-feature view: one forward pass per condition ------
    prompted_prompt_ids = [prompted_by_scenario[sid]["prompt_ids"] for sid in eval_ids]
    prompted_generated_ids = [prompted_by_scenario[sid]["generated_ids"] for sid in eval_ids]
    resid_prompted_list, code_prompted_list = capture_direction_features(
        model, sae, layer, prompted_prompt_ids, prompted_generated_ids, device
    )
    resid_prompted = dict(zip(eval_ids, resid_prompted_list))
    code_prompted = dict(zip(eval_ids, code_prompted_list))

    baseline_ids = [retokenize_for_capture(tokenizer, scenario_prompt_by_id[sid], gens[sid]["baseline_text"]) for sid in eval_ids]
    resid_baseline_list, code_baseline_list = capture_direction_features(
        model, sae, layer, [p for p, _ in baseline_ids], [g for _, g in baseline_ids], device
    )
    resid_baseline = dict(zip(eval_ids, resid_baseline_list))
    code_baseline = dict(zip(eval_ids, code_baseline_list))

    steered_ids = [retokenize_for_capture(tokenizer, scenario_prompt_by_id[sid], gens[sid]["text"]) for sid in eval_ids]
    resid_steered_list, code_steered_list = capture_direction_features(
        model, sae, layer, [p for p, _ in steered_ids], [g for _, g in steered_ids], device
    )
    resid_steered = dict(zip(eval_ids, resid_steered_list))
    code_steered = dict(zip(eval_ids, code_steered_list))

    w_r, b_r, mu_r, sigma_r = fit_classifier(resid_steered, resid_baseline)
    logit_prompted_r = mean_or_nan([classifier_logit(resid_prompted[sid], w_r, b_r, mu_r, sigma_r) for sid in eval_ids])
    logit_baseline_r = mean_or_nan([classifier_logit(resid_baseline[sid], w_r, b_r, mu_r, sigma_r) for sid in eval_ids])
    logit_steered_r = mean_or_nan([classifier_logit(resid_steered[sid], w_r, b_r, mu_r, sigma_r) for sid in eval_ids])
    ef_resid = effect_fraction(logit_prompted_r, logit_baseline_r, logit_steered_r)

    mean_prompted_vec = np.mean([resid_prompted[sid] for sid in eval_ids], axis=0)
    mean_baseline_vec = np.mean([resid_baseline[sid] for sid in eval_ids], axis=0)
    mean_steered_vec = np.mean([resid_steered[sid] for sid in eval_ids], axis=0)
    ax_cos = axis_cosine(mean_prompted_vec, mean_baseline_vec, mean_steered_vec)

    kind = info.get("kind")
    if kind == "feature":
        feature_idx = info["id"]
        feat_prompted = mean_or_nan([code_prompted[sid][feature_idx] for sid in eval_ids])
        feat_baseline = mean_or_nan([code_baseline[sid][feature_idx] for sid in eval_ids])
        feat_steered = mean_or_nan([code_steered[sid][feature_idx] for sid in eval_ids])
        ff = feature_fraction(feat_prompted, feat_baseline, feat_steered)
    else:
        axis = mean_steered_vec - mean_baseline_vec
        norm = np.linalg.norm(axis)
        unit_axis = axis / norm if norm > 1e-8 else axis
        proj_prompted = mean_or_nan([float(np.dot(resid_prompted[sid], unit_axis)) for sid in eval_ids])
        proj_baseline = mean_or_nan([float(np.dot(resid_baseline[sid], unit_axis)) for sid in eval_ids])
        proj_steered = mean_or_nan([float(np.dot(resid_steered[sid], unit_axis)) for sid in eval_ids])
        ff = feature_fraction(proj_prompted, proj_baseline, proj_steered)

    metrics = {
        "effect_fraction_resid": ef_resid,
        "effect_fraction_lex": ef_lex,
        "axis_cosine": ax_cos,
        "feature_fraction": ff,
    }
    extra = {"prompted": prompted_by_scenario, "eval_ids": eval_ids}
    return metrics, extra


def _run_judge_check(
    client: WorkerClient,
    run_id: str,
    direction_key: str,
    property_text: str,
    prompted_by_scenario: Dict[str, Dict[str, Any]],
    steered_by_scenario: Dict[str, Dict[str, str]],
    eval_ids: List[str],
    judge_budget_usd: float,
    judge_call_ceiling: int,
) -> Optional[float]:
    """Plans/drives/fetches a blinded prompted-vs-steered judge comparison
    for the best variant only, both pair orders, on `eval_ids`. Returns
    `judge_steered_share` (fraction of classifiable pairs where the judge
    said the steered side showed more of `property_text`), or `None` if
    the plan was refused (budget/ceiling) or nothing came back -- the
    caller then leaves the direction's judge check un-consulted rather
    than failing the reach stage over it."""
    judge_key = f"reach-{direction_key}"
    pairs: List[Dict[str, Any]] = []
    for sid in eval_ids:
        prompted_text = prompted_by_scenario[sid]["text"]
        steered_text = steered_by_scenario[sid]["text"]
        pairs.append({"directionKey": judge_key, "scenario": sid, "textA": prompted_text, "textB": steered_text, "orderSwap": False})
        pairs.append({"directionKey": judge_key, "scenario": sid, "textA": steered_text, "textB": prompted_text, "orderSwap": True})

    plan = client.judge_plan(run_id, pairs, judge_budget_usd, judge_call_ceiling)
    if not plan:
        logger.warning("[reach] judge plan for %s was refused (budget/ceiling); skipping judge check", judge_key)
        return None

    while True:
        result = client.judge_run(run_id, 10)
        if result is None:
            logger.warning("[reach] judge_run failed for %s; skipping judge check", judge_key)
            return None
        if int(result.get("remainingQueued", 0) or 0) <= 0:
            break
        time.sleep(5.0)

    fetched = client.judge_results(run_id)
    if not fetched:
        return None
    rows = [r for r in fetched.get("results", []) or [] if r.get("directionKey") == judge_key]
    if not rows:
        return None

    flags = [
        describe._steered_has_more(r.get("response", {}).get("more_in"), bool(r.get("orderSwap")))
        for r in rows
    ]
    classifiable = [f for f in flags if f is not None]
    if not classifiable:
        return None
    return sum(1 for f in classifiable if f) / len(classifiable)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_reach_report(output: Dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value))

    def fmt(value: Any) -> Any:
        return None if value is None or not isinstance(value, (int, float)) or not np.isfinite(value) else round(value, 3)

    def variant_cell(variants: Dict[str, Any], variant: str, field: str) -> str:
        v = (variants.get(variant) or {}).get(field)
        return esc(fmt(v))

    rows_html = []
    for row in output.get("directions", []):
        variants = row.get("variants") or {}
        variant_cells = "".join(
            f"<td>{variant_cell(variants, v, 'effect_fraction_resid')} / {variant_cell(variants, v, 'feature_fraction')}</td>"
            for v in VARIANTS
        )
        rows_html.append(
            "<tr>"
            f"<td>{esc(row.get('directionKey'))}</td>"
            f"<td>{esc(row.get('arm'))}</td>"
            f"<td>{esc(row.get('property'))}</td>"
            f"{variant_cells}"
            f"<td>{esc(row.get('best_variant'))}</td>"
            f"<td>{esc(row.get('reachable_class'))}</td>"
            f"<td>{'yes' if row.get('mechanism_same') else 'no'}</td>"
            f"<td>{esc(fmt(row.get('judge_steered_share')))}</td>"
            "</tr>"
        )

    summary = output.get("summary_by_arm") or {}
    by_arm = summary.get("by_arm") or {}
    arm_line = ", ".join(
        f"{esc(arm)}: reachable {esc(c.get('reachable'))}, partial {esc(c.get('partial'))}, "
        f"not_reachable {esc(c.get('not_reachable'))} of {esc(c.get('total'))}"
        for arm, c in sorted(by_arm.items())
    ) or "no directions evaluated"
    control_rate = fmt(summary.get("control_pass_rate"))

    variant_headers = "".join(f"<th>{esc(v)} (effect/feature)</th>" for v in VARIANTS)
    body = "\n".join(rows_html) or "<tr><td colspan=\"12\">No reach results.</td></tr>"
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Experiment 3C reach stage</title></head><body>"
        "<h1>Reach stage (measured outcome, not a gate)</h1>"
        "<p>Clause (3) of the paper's prompt-based-discovery limitations -- "
        "whether a trait is prompt-inducible at all -- is reported here per "
        "direction and rolled up per selection arm; it does not gate the "
        "unsupervised-discovery claim (see README \"Rank stage\").</p>"
        f"<p>By arm: {arm_line}. Control pass rate: {esc(control_rate)} "
        f"(of {esc(summary.get('n_controls'))} named controls; controls are "
        "the positive control for this test and are expected to be reachable).</p>"
        "<table border=\"1\" cellpadding=\"4\"><thead><tr>"
        "<th>direction</th><th>arm</th><th>property</th>"
        f"{variant_headers}"
        "<th>best variant</th><th>reachable_class</th><th>mechanism_same</th>"
        "<th>judge_steered_share</th>"
        "</tr></thead><tbody>" + body + "</tbody></table></body></html>"
    )


# ---------------------------------------------------------------------------
# Orchestration (called from run_smoke.stage_reach)
# ---------------------------------------------------------------------------
def run_reach(
    config: Any,
    workdir: Path,
    client: WorkerClient,
    run_id: str,
    model: Any,
    tokenizer: Any,
    sae: Any,
    device: Any,
    describe_output: Dict[str, Any],
    screen_directions: Sequence[Dict[str, Any]],
    screen_generations: Sequence[Dict[str, Any]],
    scenario_prompt_by_id: Dict[str, str],
) -> Dict[str, Any]:
    """Runs the full reach stage: selects the directions describe.py named
    (plus every named control, uncapped -- see `select_reach_directions`),
    generates and scores the four prompt variants for each, optionally
    drives a judge check on the best variant, classifies `reachable_class`/
    `mechanism_same`, writes `reach_results.json` + `reach-report.html`
    under `workdir`, and reports a compact per-direction summary to the
    harness under stage `"judge"` (recordId `f"reach-{directionKey}"`, no
    dedicated Worker stage -- same rationale as rank.py's `"train"` reuse).
    Resumable: skips entirely if `reach_results.json` already exists."""
    path = workdir / "reach_results.json"
    if path.exists():
        logger.info("[reach] results already exist, skipping reach stage")
        return json.loads(path.read_text(encoding="utf-8"))

    clusters = describe_output.get("clusters", {})
    selected = select_reach_directions(clusters, config.reach_max_directions)
    directions_lookup = _direction_lookup(screen_directions)
    gens_by_direction = _generations_by_direction(screen_generations)

    direction_results: List[Dict[str, Any]] = []
    for key, cluster_summary in selected:
        info = directions_lookup.get(key)
        gens = gens_by_direction.get(key)
        if info is None or gens is None:
            logger.warning("[reach] no screen data for named direction %s; skipping", key)
            continue
        property_text = cluster_summary.get("cluster_property") or ""
        scenario_ids = sorted(gens.keys())
        if len(scenario_ids) < MIN_SCENARIOS_NEEDED:
            logger.warning("[reach] direction %s has too few scenarios (%d) for reach; skipping", key, len(scenario_ids))
            continue
        demo_ids = select_fewshot_demo_scenarios(scenario_ids, N_FEWSHOT_DEMOS)
        fewshot_eval_ids = [sid for sid in scenario_ids if sid not in demo_ids]

        variant_results: Dict[str, Dict[str, float]] = {}
        variant_extra: Dict[str, Dict[str, Any]] = {}
        for variant in VARIANTS:
            eval_ids = fewshot_eval_ids if variant == "fewshot" else scenario_ids
            metrics, extra = _run_variant(
                variant, info, property_text, eval_ids, demo_ids, gens, scenario_prompt_by_id,
                model, tokenizer, sae, config.layer, config.max_new_tokens, config.generation_batch_size, device,
            )
            variant_results[variant] = metrics
            variant_extra[variant] = extra

        best_variant = pick_best_variant(variant_results)
        best_effect = variant_results[best_variant]["effect_fraction_resid"] if best_variant else float("nan")
        best_feature = variant_results[best_variant]["feature_fraction"] if best_variant else float("nan")

        judge_steered_share: Optional[float] = None
        if config.reach_judge and best_variant:
            eval_ids = variant_extra[best_variant]["eval_ids"]
            judge_steered_share = _run_judge_check(
                client, run_id, key, property_text,
                variant_extra[best_variant]["prompted"], gens, eval_ids,
                config.judge_budget_usd, config.judge_call_ceiling,
            )

        reachable_class = classify_reachability(
            best_effect, judge_steered_share, config.reach_effect_threshold, config.reach_not_threshold
        )
        mech_same = mechanism_same(best_feature, config.reach_feature_threshold)

        direction_results.append(
            {
                "directionKey": key,
                "arm": cluster_summary.get("arm"),
                "property": property_text,
                "variants": variant_results,
                "best_variant": best_variant,
                "reachable_class": reachable_class,
                "mechanism_same": mech_same,
                "judge_steered_share": judge_steered_share,
            }
        )

    summary_by_arm = summarize_by_arm(direction_results)
    output = {"directions": direction_results, "summary_by_arm": summary_by_arm}
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    (workdir / "reach-report.html").write_text(render_reach_report(output), encoding="utf-8")

    records = [
        {"recordId": f"reach-{row['directionKey']}", "payload": row} for row in direction_results
    ]
    client.report("judge", progress=_progress(len(records), max(len(records), 1)), records=records)
    return output
