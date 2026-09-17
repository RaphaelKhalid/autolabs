"""CPU-only tests for the rank stage (rank.py): the label-free persona-
context shift score, the label-free assistant-specificity/breadth/topic-
invariance statistics, the composite score, and three-arm candidate
selection. Mirrors test_pipeline.py's pattern of testing every pure-
Python/numpy helper and leaving anything that needs a real transformers
model + GPU (`rank.capture_context_activations`, `rank.
capture_specificity_activations`, `rank.run_rank`) untested here, same as
steer.py/screen.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from rank import (  # noqa: E402
    compute_composite_scores,
    compute_shift_stats,
    compute_specificity_stats,
    feature_shell,
    feature_shell_index,
    is_dead,
    rank_candidates,
    shell_bonus,
    zscore,
)
from steer import select_steer_features  # noqa: E402


class _FakeConfig:
    """Minimal stand-in for config.Config carrying only the fields
    rank_candidates reads."""

    def __init__(
        self,
        matryoshka_shells,
        firing_density_min,
        firing_density_max,
        arm_sizes,
    ):
        self.matryoshka_shells = matryoshka_shells
        self.firing_density_min = firing_density_min
        self.firing_density_max = firing_density_max
        self.arm_sizes = arm_sizes


# ---------------------------------------------------------------------------
# rank.compute_shift_stats: the F-like ratio helper (positive-control arm)
# ---------------------------------------------------------------------------
def test_shift_f_ranks_a_context_shifting_feature_above_a_flat_one():
    # Feature 0: a clear, consistent shift between three contexts (large
    # between-context spread, ~zero within-context/scenario noise).
    # Feature 1: no real context effect, just scenario-to-scenario noise
    # scattered around the same mean in every context.
    context_scenario_means = {
        "context_a": {"s1": [1.0, 3.0], "s2": [1.1, 2.9], "s3": [0.9, 3.1]},
        "context_b": {"s1": [5.0, 3.05], "s2": [5.1, 2.95], "s3": [4.9, 3.0]},
        "context_c": {"s1": [9.0, 3.1], "s2": [9.1, 2.9], "s3": [8.9, 3.0]},
    }
    max_activation = [10.0, 10.0]

    stats = compute_shift_stats(context_scenario_means, max_activation)
    by_feature = {row["feature"]: row for row in stats}

    assert by_feature[0]["shift_f"] > by_feature[1]["shift_f"]
    # The shifting feature also has a much larger normalized max-diff.
    assert by_feature[0]["shift_maxdiff"] > by_feature[1]["shift_maxdiff"]
    # The flat feature's contexts barely differ, so its shift score should
    # be close to the noise floor.
    assert by_feature[1]["shift_f"] < by_feature[0]["shift_f"] / 10


def test_shift_stats_single_context_group_is_zero():
    # Only one context has data for this run -- no between-group variance
    # is measurable, so shift_f/shift_maxdiff must both be exactly 0, not
    # NaN/inf.
    context_scenario_means = {"context_a": {"s1": [1.0], "s2": [2.0]}}
    stats = compute_shift_stats(context_scenario_means, max_activation=[5.0])
    assert stats == [{"feature": 0, "shift_f": 0.0, "shift_maxdiff": 0.0}]


def test_shift_maxdiff_normalized_by_zero_max_activation_is_zero_not_inf():
    context_scenario_means = {
        "context_a": {"s1": [1.0]},
        "context_b": {"s1": [5.0]},
    }
    stats = compute_shift_stats(context_scenario_means, max_activation=[0.0])
    assert stats[0]["shift_maxdiff"] == 0.0
    assert stats[0]["shift_f"] > 0.0  # the shift is still visible in the F ratio


# ---------------------------------------------------------------------------
# rank.compute_specificity_stats: the unsupervised arm's own signal
# ---------------------------------------------------------------------------
def _conv(assistant_mean, assistant_var, assistant_fire_rate, user_mean, user_fire_rate):
    return {
        "assistant_mean": assistant_mean,
        "assistant_var": assistant_var,
        "assistant_fire_rate": assistant_fire_rate,
        "user_mean": user_mean,
        "user_fire_rate": user_fire_rate,
    }


def test_specificity_stats_assistant_specific_broad_stable_feature():
    # Feature 0: fires strongly and consistently on assistant tokens across
    # every conversation, barely at all on user tokens -- assistant-
    # specific, broad (breadth 1.0), and topic-invariant (same mean in
    # every conversation, ~0 within-conversation variance).
    # Feature 1: never fires on either side -- breadth 0, specificity
    # ratio collapses to ~1 (both means ~0).
    conversations = [
        _conv([5.00, 0.0], [0.01, 0.0], [0.9, 0.0], [0.1, 0.0], [0.05, 0.0]),
        _conv([5.01, 0.0], [0.01, 0.0], [0.9, 0.0], [0.1, 0.0], [0.05, 0.0]),
        _conv([4.99, 0.0], [0.01, 0.0], [0.9, 0.0], [0.1, 0.0], [0.05, 0.0]),
    ]
    stats = compute_specificity_stats(conversations, width=2)
    by_feature = {row["feature"]: row for row in stats}

    assert by_feature[0]["breadth"] == 1.0
    assert by_feature[0]["assistant_specificity"] > 10.0  # ~5.0 / 0.1
    assert by_feature[0]["topic_invariance"] > 0.9  # near-zero between-conv variance

    assert by_feature[1]["breadth"] == 0.0
    assert by_feature[1]["topic_invariance"] == 0.0  # <2 firing conversations


def test_specificity_stats_topic_bound_feature_has_low_topic_invariance():
    # Feature 0 fires on assistant tokens, but its mean swings wildly from
    # conversation to conversation (topic-bound) with low within-
    # conversation noise -- between-conversation variance dominates the
    # total, so topic_invariance should be low.
    conversations = [
        _conv([1.0], [0.01], [0.9], [0.0], [0.0]),
        _conv([10.0], [0.01], [0.9], [0.0], [0.0]),
        _conv([1.0], [0.01], [0.9], [0.0], [0.0]),
        _conv([10.0], [0.01], [0.9], [0.0], [0.0]),
    ]
    stats = compute_specificity_stats(conversations, width=1)
    assert stats[0]["topic_invariance"] < 0.1


def test_specificity_stats_zero_conversations_is_empty_not_error():
    assert compute_specificity_stats([], width=3) == [
        {
            "feature": f,
            "assistant_specificity": 0.0,
            "assistant_specificity_fire_rate": 0.0,
            "breadth": 0.0,
            "topic_invariance": 0.0,
        }
        for f in range(3)
    ]


# ---------------------------------------------------------------------------
# rank.zscore / rank.shell_bonus / rank.feature_shell_index
# ---------------------------------------------------------------------------
def test_zscore_mean_zero_std_one_on_varied_input():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    z = zscore(values)
    assert sum(z) == pytest.approx(0.0, abs=1e-9)
    assert z[0] < 0 < z[-1]


def test_zscore_constant_input_is_all_zero_not_nan():
    assert zscore([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]
    assert zscore([]) == []
    assert zscore([5.0]) == [0.0]


def test_shell_bonus_values():
    assert shell_bonus(0) == 1.0
    assert shell_bonus(1) == 0.5
    assert shell_bonus(2) == 0.0
    assert shell_bonus(None) == 0.0


def test_feature_shell_and_shell_index_agree():
    shells = [1024, 4096, 8192]
    assert feature_shell(0, shells) == 1024 and feature_shell_index(0, shells) == 0
    assert feature_shell(1024, shells) == 4096 and feature_shell_index(1024, shells) == 1
    assert feature_shell(8191, shells) == 8192 and feature_shell_index(8191, shells) == 2
    assert feature_shell(8192, shells) is None and feature_shell_index(8192, shells) is None


# ---------------------------------------------------------------------------
# rank.is_dead
# ---------------------------------------------------------------------------
def test_is_dead():
    assert is_dead(0.0) is True
    assert is_dead(None) is True
    assert is_dead(1e-6) is False


# ---------------------------------------------------------------------------
# rank.compute_composite_scores: favors assistant-specific, broad,
# topic-invariant features over topic-bound ones, independent of shift_f
# ---------------------------------------------------------------------------
def test_compute_composite_scores_favors_assistant_specific_broad_features():
    candidates = [
        {
            "feature": 0,
            "assistant_specificity": 8.0,
            "breadth": 0.9,
            "topic_invariance": 0.95,
            "shell_index": 0,
            "shift_f": 0.01,  # deliberately low -- composite must not care
        },
        {
            "feature": 1,
            "assistant_specificity": 1.0,
            "breadth": 0.05,
            "topic_invariance": 0.05,
            "shell_index": 2,
            "shift_f": 500.0,  # deliberately high -- composite must not care
        },
    ]
    compute_composite_scores(candidates)
    by_feature = {c["feature"]: c for c in candidates}
    assert by_feature[0]["composite"] > by_feature[1]["composite"]


def test_compute_composite_scores_uses_log_specificity_not_raw():
    # Two features with wildly different raw assistant_specificity but the
    # same rank-order relationship should still just reflect ordering --
    # this mostly guards against a crash on assistant_specificity <= 0
    # (log of a non-positive number), which the eps clamp must prevent.
    candidates = [
        {"feature": 0, "assistant_specificity": 0.0, "breadth": 0.5, "topic_invariance": 0.5, "shell_index": None},
        {"feature": 1, "assistant_specificity": 100.0, "breadth": 0.5, "topic_invariance": 0.5, "shell_index": None},
    ]
    compute_composite_scores(candidates)
    assert candidates[0]["composite"] < candidates[1]["composite"]
    assert all(isinstance(c["composite"], float) for c in candidates)


# ---------------------------------------------------------------------------
# rank.rank_candidates: filters + three-arm disjoint selection
# ---------------------------------------------------------------------------
def _toy_feature_stats(n=40):
    # Densities alternate: half inside [1e-4, 0.1], half dead or out of
    # range, so the filter step actually excludes something.
    densities = []
    for i in range(n):
        if i % 4 == 0:
            densities.append(0.0)  # dead
        elif i % 4 == 1:
            densities.append(0.5)  # out of range (too dense)
        else:
            densities.append(1e-3 * (i + 1))  # in range
    return {"firing_density": densities, "max_activation": [1.0] * n}


def _toy_shift_stats(n=40):
    # Give every feature a distinct shift_f so the shift-slice ordering is
    # unambiguous; feature index itself is (inversely) the score.
    return [{"feature": i, "shift_f": float(n - i), "shift_maxdiff": 0.1} for i in range(n)]


def _toy_specificity_stats(n=40):
    # Give every feature a distinct composite-relevant profile, increasing
    # with feature index, so the unsupervised slice ordering is
    # unambiguous and (deliberately) anti-correlated with shift_f's
    # ordering above -- the two arms should pick disjoint, differently-
    # ordered features.
    return [
        {
            "feature": i,
            "assistant_specificity": 1.0 + 0.1 * i,
            "assistant_specificity_fire_rate": 1.0,
            "breadth": min(1.0, 0.01 * i),
            "topic_invariance": min(1.0, 0.01 * i),
        }
        for i in range(n)
    ]


def test_rank_candidates_filters_dead_and_out_of_range_density():
    feature_stats = _toy_feature_stats(20)
    shift_stats = _toy_shift_stats(20)
    specificity_stats = _toy_specificity_stats(20)
    config = _FakeConfig(
        matryoshka_shells=[20],
        firing_density_min=1e-4,
        firing_density_max=0.1,
        arm_sizes={"unsupervised": 2, "quantile": 1, "shift": 1},
    )
    result = rank_candidates(feature_stats, shift_stats, specificity_stats, config)
    # Only i % 4 in (2, 3) survive the density filter -> 10 candidates of 20.
    assert result["candidates_considered"] == 10
    selected_features = {c["feature"] for c in result["screen_features"]}
    for f in selected_features:
        assert f % 4 in (2, 3)


def test_rank_candidates_arms_disjoint_and_correctly_sized():
    feature_stats = _toy_feature_stats(40)
    shift_stats = _toy_shift_stats(40)
    specificity_stats = _toy_specificity_stats(40)
    config = _FakeConfig(
        matryoshka_shells=[40],
        firing_density_min=1e-4,
        firing_density_max=0.1,
        arm_sizes={"unsupervised": 4, "quantile": 3, "shift": 3},
    )
    result = rank_candidates(feature_stats, shift_stats, specificity_stats, config)
    screen_features = result["screen_features"]

    assert result["arm_counts"] == {"unsupervised": 4, "quantile": 3, "shift": 3}
    assert len(screen_features) == 10

    by_arm = {"unsupervised": [], "quantile": [], "shift": []}
    for c in screen_features:
        by_arm[c["arm"]].append(c)
        assert c["selection"] == c["arm"]  # kept equal for backward compat

    assert len(by_arm["unsupervised"]) == 4
    assert len(by_arm["quantile"]) == 3
    assert len(by_arm["shift"]) == 3

    all_features = [c["feature"] for c in screen_features]
    assert len(all_features) == len(set(all_features))  # disjoint across all three arms

    # Every record carries the full schema.
    for c in screen_features:
        assert set(c) == {
            "feature", "density", "shell", "shift_f", "shift_maxdiff",
            "assistant_specificity", "assistant_specificity_fire_rate",
            "breadth", "topic_invariance", "composite", "arm", "selection",
            "rank_within_arm",
        }
        assert c["rank_within_arm"] >= 1


def test_rank_candidates_unsupervised_arm_is_top_by_composite():
    feature_stats = _toy_feature_stats(40)
    shift_stats = _toy_shift_stats(40)
    specificity_stats = _toy_specificity_stats(40)
    config = _FakeConfig(
        matryoshka_shells=[40],
        firing_density_min=1e-4,
        firing_density_max=0.1,
        arm_sizes={"unsupervised": 4, "quantile": 0, "shift": 0},
    )
    result = rank_candidates(feature_stats, shift_stats, specificity_stats, config)
    unsupervised = [c for c in result["screen_features"] if c["arm"] == "unsupervised"]

    candidate_features = [i for i, d in enumerate(feature_stats["firing_density"]) if 1e-4 <= d <= 0.1]
    # specificity/breadth/topic_invariance all increase with feature index
    # in _toy_specificity_stats, so the composite-top candidates are the
    # largest surviving feature indices.
    expected_top4 = sorted(candidate_features)[-4:]
    assert sorted(c["feature"] for c in unsupervised) == expected_top4


def test_rank_candidates_shift_arm_is_top_by_shift_f_from_remainder():
    feature_stats = _toy_feature_stats(40)
    shift_stats = _toy_shift_stats(40)
    specificity_stats = _toy_specificity_stats(40)
    config = _FakeConfig(
        matryoshka_shells=[40],
        firing_density_min=1e-4,
        firing_density_max=0.1,
        arm_sizes={"unsupervised": 0, "quantile": 0, "shift": 4},
    )
    result = rank_candidates(feature_stats, shift_stats, specificity_stats, config)
    shift_selected = [c for c in result["screen_features"] if c["arm"] == "shift"]

    candidate_features = [i for i, d in enumerate(feature_stats["firing_density"]) if 1e-4 <= d <= 0.1]
    # shift_f = n - feature, so the smallest feature indices score highest.
    expected_top4 = sorted(candidate_features)[:4]
    assert sorted(c["feature"] for c in shift_selected) == expected_top4


def test_rank_candidates_arm_sizes_capped_when_pool_smaller_than_total():
    # Only 10 candidates survive filtering (see _toy_feature_stats), but
    # arm_sizes asks for 40 total -- every arm must be capped, filled in
    # order unsupervised -> quantile -> shift, still disjoint, and the
    # total must equal candidates_considered exactly (nothing left over,
    # nothing double-counted).
    feature_stats = _toy_feature_stats(20)
    shift_stats = _toy_shift_stats(20)
    specificity_stats = _toy_specificity_stats(20)
    config = _FakeConfig(
        matryoshka_shells=[20],
        firing_density_min=1e-4,
        firing_density_max=0.1,
        arm_sizes={"unsupervised": 20, "quantile": 20, "shift": 20},
    )
    result = rank_candidates(feature_stats, shift_stats, specificity_stats, config)

    assert result["candidates_considered"] == 10
    total_selected = sum(result["arm_counts"].values())
    assert total_selected == 10
    assert len(result["screen_features"]) == 10
    all_features = [c["feature"] for c in result["screen_features"]]
    assert len(all_features) == len(set(all_features))
    # unsupervised (filled first) should have taken everything, leaving
    # nothing for quantile/shift.
    assert result["arm_counts"]["unsupervised"] == 10
    assert result["arm_counts"]["quantile"] == 0
    assert result["arm_counts"]["shift"] == 0


def test_rank_candidates_empty_when_no_candidates_survive():
    feature_stats = {"firing_density": [0.0, 0.0], "max_activation": [1.0, 1.0]}
    shift_stats = [{"feature": 0, "shift_f": 1.0, "shift_maxdiff": 0.1}, {"feature": 1, "shift_f": 2.0, "shift_maxdiff": 0.2}]
    specificity_stats = [
        {"feature": 0, "assistant_specificity": 1.0, "assistant_specificity_fire_rate": 1.0, "breadth": 0.1, "topic_invariance": 0.1},
        {"feature": 1, "assistant_specificity": 1.0, "assistant_specificity_fire_rate": 1.0, "breadth": 0.1, "topic_invariance": 0.1},
    ]
    config = _FakeConfig(
        matryoshka_shells=[2],
        firing_density_min=1e-4,
        firing_density_max=0.1,
        arm_sizes={"unsupervised": 4, "quantile": 2, "shift": 2},
    )
    result = rank_candidates(feature_stats, shift_stats, specificity_stats, config)
    assert result == {
        "screen_features": [],
        "arm_counts": {"unsupervised": 0, "quantile": 0, "shift": 0},
        "candidates_considered": 0,
    }


def test_rank_candidates_requires_both_shift_and_specificity_entries():
    # A feature missing either a shift_stats or a specificity_stats entry
    # must be excluded from the candidate pool entirely, same as a missing
    # shift entry did before specificity_stats existed.
    feature_stats = {"firing_density": [0.01, 0.01, 0.01], "max_activation": [1.0, 1.0, 1.0]}
    shift_stats = [{"feature": 0, "shift_f": 1.0, "shift_maxdiff": 0.1}]  # feature 1, 2 missing
    specificity_stats = [
        {"feature": 0, "assistant_specificity": 1.0, "assistant_specificity_fire_rate": 1.0, "breadth": 0.1, "topic_invariance": 0.1},
        {"feature": 1, "assistant_specificity": 1.0, "assistant_specificity_fire_rate": 1.0, "breadth": 0.1, "topic_invariance": 0.1},
        # feature 2 missing
    ]
    config = _FakeConfig(
        matryoshka_shells=[3],
        firing_density_min=1e-4,
        firing_density_max=0.1,
        arm_sizes={"unsupervised": 5, "quantile": 5, "shift": 5},
    )
    result = rank_candidates(feature_stats, shift_stats, specificity_stats, config)
    assert result["candidates_considered"] == 1
    assert [c["feature"] for c in result["screen_features"]] == [0]


# ---------------------------------------------------------------------------
# steer.select_steer_features: explicit feature list bypasses density
# selection entirely, and carries `arm` through
# ---------------------------------------------------------------------------
def test_select_steer_features_explicit_list_is_honored_verbatim():
    feature_stats = {
        "firing_density": [0.9, 0.9, 0.9],  # all out of [1e-4, 0.1] range
        "max_activation": [1.0, 2.0, 3.0],
    }
    explicit = [
        {"feature": 0, "density": 0.9, "max_activation": 1.0, "arm": "unsupervised", "shell": 4, "shift_f": 5.0, "rank_within_arm": 1},
        {"feature": 2, "density": 0.9, "max_activation": 3.0, "arm": "shift", "shell": 4, "shift_f": 1.0, "rank_within_arm": 3},
    ]
    selected = select_steer_features(
        feature_stats,
        first_shell_size=3,
        steer_features=1,  # would normally cap the result to 1
        density_min=1e-4,
        density_max=0.1,  # explicit features are all out of this range
        explicit_features=explicit,
    )
    assert [c["feature"] for c in selected] == [0, 2]
    assert all(c["quantile"] is None for c in selected)
    assert [c["max_activation"] for c in selected] == [1.0, 3.0]
    # The rank stage's arm tags ride through onto the calibrate-stage
    # feature-selection dicts verbatim (see rank.py "Arm propagation").
    assert [c["arm"] for c in selected] == ["unsupervised", "shift"]


def test_select_steer_features_no_explicit_list_falls_back_to_density_selection():
    feature_stats = {
        "firing_density": [0.5, 0.05, 1e-5, 0.02, 0.0001, 0.2],
        "max_activation": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
    selected = select_steer_features(
        feature_stats, first_shell_size=6, steer_features=2, density_min=1e-4, density_max=0.1, explicit_features=None
    )
    assert [c["feature"] for c in selected] == [3, 1]
    # No rank stage in play: every selected feature gets arm=None, not one
    # of the three arm tags.
    assert all(c["arm"] is None for c in selected)
