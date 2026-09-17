"""CPU-only tests for the reach stage (reach.py).

Mirrors screen.py/describe.py's split: prompt construction, the effect/
feature-fraction rescaling math, classification thresholds, and the
per-arm summary are pure Python/numpy and tested here on toy data.
Anything that needs a real transformers model (generation, residual/SAE
activation capture) or the Worker (the judge check) is out of scope, same
as steer.py/screen.py's own model-touching helpers.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import reach  # noqa: E402


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def test_build_system_prompt_direct():
    assert reach.build_system_prompt("direct", "speaks warmly") == "In your replies, speaks warmly."


def test_build_system_prompt_rewrite_and_fewshot_share_the_same_system_text():
    rewrite = reach.build_system_prompt("rewrite", "speaks warmly")
    fewshot = reach.build_system_prompt("fewshot", "speaks warmly")
    assert rewrite == fewshot
    assert rewrite == "Adopt this manner throughout: speaks warmly. Keep it natural and consistent."


def test_build_system_prompt_intensified_extends_rewrite():
    rewrite = reach.build_system_prompt("rewrite", "speaks warmly")
    intensified = reach.build_system_prompt("intensified", "speaks warmly")
    assert intensified.startswith(rewrite)
    assert "strongly" in intensified and "every reply" in intensified


def test_build_system_prompt_unknown_variant_raises():
    with pytest.raises(ValueError):
        reach.build_system_prompt("bogus", "speaks warmly")


def test_select_fewshot_demo_scenarios_deterministic_sorted_first_n():
    ids = ["s3", "s1", "s4", "s2"]
    assert reach.select_fewshot_demo_scenarios(ids, n=2) == ["s1", "s2"]
    # Deterministic across calls / a shuffled input order.
    assert reach.select_fewshot_demo_scenarios(list(reversed(ids)), n=2) == ["s1", "s2"]


def test_select_fewshot_demo_scenarios_caps_at_available_count():
    assert reach.select_fewshot_demo_scenarios(["only-one"], n=2) == ["only-one"]


def test_build_variant_messages_direct_rewrite_intensified_are_system_plus_user():
    for variant in ("direct", "rewrite", "intensified"):
        messages = reach.build_variant_messages(variant, "speaks warmly", "hello there")
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[-1]["content"] == "hello there"


def test_build_variant_messages_fewshot_includes_demo_turns_then_eval_prompt():
    demo_pairs = [("demo prompt A", "demo reply A"), ("demo prompt B", "demo reply B")]
    messages = reach.build_variant_messages("fewshot", "speaks warmly", "eval prompt", demo_pairs)
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[1]["content"] == "demo prompt A"
    assert messages[2]["content"] == "demo reply A"
    assert messages[3]["content"] == "demo prompt B"
    assert messages[4]["content"] == "demo reply B"
    assert messages[-1]["content"] == "eval prompt"


def test_fewshot_eval_set_excludes_its_own_demo_scenarios():
    """The orchestration contract reach.run_reach relies on: fewshot's
    evaluation set is the full scenario set minus its own two demo
    scenarios, so a demonstration's own steered reply is never also
    scored as evidence the demonstration worked."""
    scenario_ids = ["s1", "s2", "s3", "s4"]
    demo_ids = reach.select_fewshot_demo_scenarios(scenario_ids, reach.N_FEWSHOT_DEMOS)
    fewshot_eval_ids = [sid for sid in scenario_ids if sid not in demo_ids]
    assert set(demo_ids) & set(fewshot_eval_ids) == set()
    assert set(demo_ids) | set(fewshot_eval_ids) == set(scenario_ids)
    assert len(demo_ids) == reach.N_FEWSHOT_DEMOS


# ---------------------------------------------------------------------------
# effect_fraction (toy logits, including clipping)
# ---------------------------------------------------------------------------
def test_effect_fraction_matches_steering_exactly_is_one():
    assert reach.effect_fraction(prompted_mean=2.0, baseline_mean=0.0, steered_mean=2.0) == pytest.approx(1.0)


def test_effect_fraction_matches_baseline_exactly_is_zero():
    assert reach.effect_fraction(prompted_mean=0.0, baseline_mean=0.0, steered_mean=2.0) == pytest.approx(0.0)


def test_effect_fraction_halfway_is_one_half():
    assert reach.effect_fraction(prompted_mean=1.0, baseline_mean=0.0, steered_mean=2.0) == pytest.approx(0.5)


def test_effect_fraction_clips_overshoot_to_max():
    # prompted overshoots past the steered value (raw ratio would be 2.0)
    assert reach.effect_fraction(prompted_mean=4.0, baseline_mean=0.0, steered_mean=2.0) == pytest.approx(1.5)


def test_effect_fraction_clips_wrong_direction_to_min():
    # prompted moves the *opposite* way from baseline->steered (raw ratio -1.0 -> clip to -0.5)
    assert reach.effect_fraction(prompted_mean=-2.0, baseline_mean=0.0, steered_mean=2.0) == pytest.approx(-0.5)


def test_effect_fraction_nan_when_steered_equals_baseline():
    result = reach.effect_fraction(prompted_mean=1.0, baseline_mean=1.0, steered_mean=1.0)
    assert math.isnan(result)


def test_effect_fraction_custom_clip_bounds_respected():
    assert reach.effect_fraction(10.0, 0.0, 1.0, clip_min=-1.0, clip_max=2.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# feature_fraction
# ---------------------------------------------------------------------------
def test_feature_fraction_matches_steering_is_one():
    assert reach.feature_fraction(prompted=3.0, baseline=1.0, steered=3.0) == pytest.approx(1.0)


def test_feature_fraction_no_effect_is_zero():
    assert reach.feature_fraction(prompted=1.0, baseline=1.0, steered=3.0) == pytest.approx(0.0)


def test_feature_fraction_unclipped_can_exceed_one():
    assert reach.feature_fraction(prompted=5.0, baseline=1.0, steered=3.0) == pytest.approx(2.0)


def test_feature_fraction_nan_on_zero_denominator():
    assert math.isnan(reach.feature_fraction(prompted=1.0, baseline=1.0, steered=1.0))


# ---------------------------------------------------------------------------
# axis_cosine
# ---------------------------------------------------------------------------
def test_axis_cosine_same_direction_is_one():
    baseline = np.array([0.0, 0.0])
    steered = np.array([1.0, 0.0])
    prompted = np.array([2.0, 0.0])  # (prompted - baseline) is parallel to (steered - baseline)
    assert reach.axis_cosine(prompted, baseline, steered) == pytest.approx(1.0)


def test_axis_cosine_orthogonal_is_zero():
    baseline = np.array([0.0, 0.0])
    steered = np.array([1.0, 0.0])
    prompted = np.array([0.0, 1.0])
    assert reach.axis_cosine(prompted, baseline, steered) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# pick_best_variant
# ---------------------------------------------------------------------------
def test_pick_best_variant_picks_max_effect_fraction_resid():
    variant_results = {
        "direct": {"effect_fraction_resid": 0.2},
        "rewrite": {"effect_fraction_resid": 0.9},
        "intensified": {"effect_fraction_resid": 0.5},
        "fewshot": {"effect_fraction_resid": 0.1},
    }
    assert reach.pick_best_variant(variant_results) == "rewrite"


def test_pick_best_variant_ignores_nan_entries():
    variant_results = {
        "direct": {"effect_fraction_resid": float("nan")},
        "rewrite": {"effect_fraction_resid": 0.4},
    }
    assert reach.pick_best_variant(variant_results) == "rewrite"


def test_pick_best_variant_all_nan_returns_none():
    variant_results = {
        "direct": {"effect_fraction_resid": float("nan")},
        "rewrite": {"effect_fraction_resid": float("nan")},
    }
    assert reach.pick_best_variant(variant_results) is None


# ---------------------------------------------------------------------------
# classify_reachability
# ---------------------------------------------------------------------------
def test_classify_reachability_reachable_above_effect_threshold_no_judge():
    result = reach.classify_reachability(
        best_effect_fraction=0.8, judge_steered_share=None, effect_threshold=0.7, not_threshold=0.3
    )
    assert result == "reachable"


def test_classify_reachability_not_reachable_below_not_threshold():
    result = reach.classify_reachability(
        best_effect_fraction=0.1, judge_steered_share=None, effect_threshold=0.7, not_threshold=0.3
    )
    assert result == "not_reachable"


def test_classify_reachability_partial_in_between():
    result = reach.classify_reachability(
        best_effect_fraction=0.5, judge_steered_share=None, effect_threshold=0.7, not_threshold=0.3
    )
    assert result == "partial"


def test_classify_reachability_judge_can_downgrade_reachable_to_partial():
    # effect fraction clears the bar, but the judge still favors "steered" too strongly
    result = reach.classify_reachability(
        best_effect_fraction=0.9, judge_steered_share=0.9, effect_threshold=0.7, not_threshold=0.3
    )
    assert result == "partial"


def test_classify_reachability_judge_at_or_below_max_still_reachable():
    result = reach.classify_reachability(
        best_effect_fraction=0.9, judge_steered_share=0.65, effect_threshold=0.7, not_threshold=0.3
    )
    assert result == "reachable"


def test_classify_reachability_nan_effect_falls_through_to_partial():
    result = reach.classify_reachability(
        best_effect_fraction=float("nan"), judge_steered_share=None, effect_threshold=0.7, not_threshold=0.3
    )
    assert result == "partial"


# ---------------------------------------------------------------------------
# mechanism_same
# ---------------------------------------------------------------------------
def test_mechanism_same_true_at_or_above_threshold():
    assert reach.mechanism_same(0.5, 0.5) is True
    assert reach.mechanism_same(0.9, 0.5) is True


def test_mechanism_same_false_below_threshold():
    assert reach.mechanism_same(0.4, 0.5) is False


def test_mechanism_same_false_on_nan():
    assert reach.mechanism_same(float("nan"), 0.5) is False


# ---------------------------------------------------------------------------
# select_reach_directions
# ---------------------------------------------------------------------------
def _cluster(named: bool, consistency_score: float, arm: str) -> Dict[str, Any]:
    return {"named": named, "consistency_score": consistency_score, "arm": arm, "cluster_property": "p"}


def test_select_reach_directions_excludes_unnamed_and_nulls():
    clusters = {
        "feature-1-pos": _cluster(True, 0.9, "unsupervised"),
        "feature-2-pos": _cluster(False, 0.99, "unsupervised"),
        "random-3-na": _cluster(True, 0.99, "random"),
    }
    selected = reach.select_reach_directions(clusters, max_directions=5)
    assert [k for k, _ in selected] == ["feature-1-pos"]


def test_select_reach_directions_orders_by_consistency_score_descending():
    clusters = {
        "feature-1-pos": _cluster(True, 0.3, "unsupervised"),
        "feature-2-pos": _cluster(True, 0.9, "unsupervised"),
        "feature-3-pos": _cluster(True, 0.6, "unsupervised"),
    }
    selected = reach.select_reach_directions(clusters, max_directions=5)
    assert [k for k, _ in selected] == ["feature-2-pos", "feature-3-pos", "feature-1-pos"]


def test_select_reach_directions_caps_non_control_directions():
    clusters = {f"feature-{i}-pos": _cluster(True, float(i), "unsupervised") for i in range(5)}
    selected = reach.select_reach_directions(clusters, max_directions=2)
    assert len(selected) == 2
    assert [k for k, _ in selected] == ["feature-4-pos", "feature-3-pos"]


def test_select_reach_directions_always_includes_named_controls_beyond_the_cap():
    clusters = {f"feature-{i}-pos": _cluster(True, float(i), "unsupervised") for i in range(5)}
    clusters["control-evil_benevolent-pos"] = _cluster(True, -1.0, "control")  # would rank last
    selected = reach.select_reach_directions(clusters, max_directions=2)
    keys = [k for k, _ in selected]
    assert "control-evil_benevolent-pos" in keys
    # the cap is still respected for everything except the forced-in control
    assert len(keys) == 3


def test_select_reach_directions_unnamed_control_is_not_forced_in():
    clusters = {
        "feature-0-pos": _cluster(True, 0.9, "unsupervised"),
        "control-evil_benevolent-pos": _cluster(False, 0.5, "control"),
    }
    selected = reach.select_reach_directions(clusters, max_directions=1)
    assert [k for k, _ in selected] == ["feature-0-pos"]


# ---------------------------------------------------------------------------
# summarize_by_arm
# ---------------------------------------------------------------------------
def test_summarize_by_arm_counts_reachable_classes_per_arm():
    rows = [
        {"arm": "unsupervised", "reachable_class": "reachable"},
        {"arm": "unsupervised", "reachable_class": "partial"},
        {"arm": "shift", "reachable_class": "reachable"},
        {"arm": "shift", "reachable_class": "not_reachable"},
    ]
    result = reach.summarize_by_arm(rows)
    assert result["by_arm"]["unsupervised"] == {"reachable": 1, "partial": 1, "not_reachable": 0, "total": 2}
    assert result["by_arm"]["shift"] == {"reachable": 1, "partial": 0, "not_reachable": 1, "total": 2}


def test_summarize_by_arm_control_pass_rate():
    rows = [
        {"arm": "control", "reachable_class": "reachable"},
        {"arm": "control", "reachable_class": "reachable"},
        {"arm": "control", "reachable_class": "partial"},
        {"arm": "unsupervised", "reachable_class": "reachable"},
    ]
    result = reach.summarize_by_arm(rows)
    assert result["n_controls"] == 3
    assert result["control_pass_rate"] == pytest.approx(2 / 3)


def test_summarize_by_arm_no_controls_gives_none_pass_rate():
    rows = [{"arm": "unsupervised", "reachable_class": "reachable"}]
    result = reach.summarize_by_arm(rows)
    assert result["n_controls"] == 0
    assert result["control_pass_rate"] is None


def test_summarize_by_arm_empty_input():
    result = reach.summarize_by_arm([])
    assert result == {"by_arm": {}, "control_pass_rate": None, "n_controls": 0}


# ---------------------------------------------------------------------------
# fit_classifier / classifier_logit (numpy-only; sanity on toy separable data)
# ---------------------------------------------------------------------------
def test_fit_classifier_and_logit_separates_toy_clusters():
    steered_vecs = {"s1": np.array([5.0, 5.0]), "s2": np.array([5.2, 4.8]), "s3": np.array([4.9, 5.1])}
    baseline_vecs = {"s1": np.array([-5.0, -5.0]), "s2": np.array([-4.8, -5.2]), "s3": np.array([-5.1, -4.9])}
    w, b, mu, sigma = reach.fit_classifier(steered_vecs, baseline_vecs)
    steered_logit = np.mean([reach.classifier_logit(steered_vecs[s], w, b, mu, sigma) for s in steered_vecs])
    baseline_logit = np.mean([reach.classifier_logit(baseline_vecs[s], w, b, mu, sigma) for s in baseline_vecs])
    assert steered_logit > baseline_logit


# ---------------------------------------------------------------------------
# is_control_key
# ---------------------------------------------------------------------------
def test_is_control_key():
    assert reach.is_control_key("control-evil_benevolent-pos")
    assert not reach.is_control_key("feature-12-pos")
    assert not reach.is_control_key("random-3-na")
