"""CPU-only tests for the rank stage (rank.py): the label-free persona-
context shift score and candidate selection. Mirrors test_pipeline.py's
pattern of testing every pure-Python/numpy helper and leaving anything
that needs a real transformers model + GPU (`rank.capture_context_
activations`, `rank.run_rank`) untested here, same as steer.py/screen.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from rank import (  # noqa: E402
    compute_shift_stats,
    feature_shell,
    is_dead,
    rank_candidates,
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
        screen_features,
        rank_shift_fraction,
    ):
        self.matryoshka_shells = matryoshka_shells
        self.firing_density_min = firing_density_min
        self.firing_density_max = firing_density_max
        self.screen_features = screen_features
        self.rank_shift_fraction = rank_shift_fraction


# ---------------------------------------------------------------------------
# rank.compute_shift_stats: the F-like ratio helper
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
# rank.is_dead / rank.feature_shell
# ---------------------------------------------------------------------------
def test_is_dead():
    assert is_dead(0.0) is True
    assert is_dead(None) is True
    assert is_dead(1e-6) is False


def test_feature_shell_smallest_containing_shell():
    shells = [1024, 4096, 8192]
    assert feature_shell(0, shells) == 1024
    assert feature_shell(1023, shells) == 1024
    assert feature_shell(1024, shells) == 4096
    assert feature_shell(8191, shells) == 8192
    assert feature_shell(8192, shells) is None  # past every shell


# ---------------------------------------------------------------------------
# rank.rank_candidates: filters + union selection + tags
# ---------------------------------------------------------------------------
def _toy_feature_stats(n=20):
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


def _toy_shift_stats(n=20):
    # Give every feature a distinct shift_f so the top-shift slice is
    # unambiguous; feature index itself is the score so ordering is
    # trivially checkable.
    return [{"feature": i, "shift_f": float(n - i), "shift_maxdiff": 0.1} for i in range(n)]


def test_rank_candidates_filters_dead_and_out_of_range_density():
    feature_stats = _toy_feature_stats(20)
    shift_stats = _toy_shift_stats(20)
    config = _FakeConfig(
        matryoshka_shells=[20],
        firing_density_min=1e-4,
        firing_density_max=0.1,
        screen_features=4,
        rank_shift_fraction=0.75,
    )
    result = rank_candidates(feature_stats, shift_stats, config)
    # Only i % 4 in (2, 3) survive the density filter -> 10 candidates of 20.
    assert result["candidates_considered"] == 10
    selected_features = {c["feature"] for c in result["screen_features"]}
    for f in selected_features:
        assert f % 4 in (2, 3)


def test_rank_candidates_union_sizes_and_tags():
    feature_stats = _toy_feature_stats(40)
    shift_stats = _toy_shift_stats(40)
    config = _FakeConfig(
        matryoshka_shells=[40],
        firing_density_min=1e-4,
        firing_density_max=0.1,
        screen_features=8,
        rank_shift_fraction=0.75,
    )
    result = rank_candidates(feature_stats, shift_stats, config)
    screen_features = result["screen_features"]

    assert result["shift_budget"] == 6
    assert result["quantile_budget"] == 2
    assert len(screen_features) == 8

    shift_tagged = [c for c in screen_features if c["selection"] == "shift"]
    quantile_tagged = [c for c in screen_features if c["selection"] == "quantile"]
    assert len(shift_tagged) == 6
    assert len(quantile_tagged) == 2

    # No feature appears in both slices.
    shift_ids = {c["feature"] for c in shift_tagged}
    quantile_ids = {c["feature"] for c in quantile_tagged}
    assert shift_ids.isdisjoint(quantile_ids)

    # Every record carries the full schema.
    for c in screen_features:
        assert set(c) == {"feature", "shift_f", "shift_maxdiff", "density", "shell", "selection", "rank"}
        assert c["rank"] >= 1


def test_rank_candidates_shift_slice_is_top_by_shift_f():
    feature_stats = _toy_feature_stats(40)
    shift_stats = _toy_shift_stats(40)
    config = _FakeConfig(
        matryoshka_shells=[40],
        firing_density_min=1e-4,
        firing_density_max=0.1,
        screen_features=8,
        rank_shift_fraction=0.75,
    )
    result = rank_candidates(feature_stats, shift_stats, config)
    shift_tagged = [c for c in result["screen_features"] if c["selection"] == "shift"]

    candidate_features = [i for i, d in enumerate(feature_stats["firing_density"]) if 1e-4 <= d <= 0.1]
    # shift_f = n - feature, so the candidates with the smallest feature
    # index have the highest shift_f.
    expected_top6 = sorted(candidate_features)[:6]
    assert sorted(c["feature"] for c in shift_tagged) == expected_top6


def test_rank_candidates_empty_when_no_candidates_survive():
    feature_stats = {"firing_density": [0.0, 0.0], "max_activation": [1.0, 1.0]}
    shift_stats = [{"feature": 0, "shift_f": 1.0, "shift_maxdiff": 0.1}, {"feature": 1, "shift_f": 2.0, "shift_maxdiff": 0.2}]
    config = _FakeConfig(
        matryoshka_shells=[2],
        firing_density_min=1e-4,
        firing_density_max=0.1,
        screen_features=8,
        rank_shift_fraction=0.75,
    )
    result = rank_candidates(feature_stats, shift_stats, config)
    assert result == {"screen_features": [], "shift_budget": 0, "quantile_budget": 0, "candidates_considered": 0}


# ---------------------------------------------------------------------------
# steer.select_steer_features: explicit feature list bypasses density
# selection entirely
# ---------------------------------------------------------------------------
def test_select_steer_features_explicit_list_is_honored_verbatim():
    feature_stats = {
        "firing_density": [0.9, 0.9, 0.9],  # all out of [1e-4, 0.1] range
        "max_activation": [1.0, 2.0, 3.0],
    }
    explicit = [
        {"feature": 0, "density": 0.9, "max_activation": 1.0, "selection": "shift", "shell": 4, "shift_f": 5.0, "rank": 1},
        {"feature": 2, "density": 0.9, "max_activation": 3.0, "selection": "quantile", "shell": 4, "shift_f": 1.0, "rank": 3},
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


def test_select_steer_features_no_explicit_list_falls_back_to_density_selection():
    feature_stats = {
        "firing_density": [0.5, 0.05, 1e-5, 0.02, 0.0001, 0.2],
        "max_activation": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
    selected = select_steer_features(
        feature_stats, first_shell_size=6, steer_features=2, density_min=1e-4, density_max=0.1, explicit_features=None
    )
    assert [c["feature"] for c in selected] == [3, 1]
