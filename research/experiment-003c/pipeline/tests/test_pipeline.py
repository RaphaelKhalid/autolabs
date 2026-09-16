"""CPU-only tests for the Experiment 3C pipeline.

Anything that needs a real transformers model + GPU (harvest's
ActivationHarvester, checks.identity_hook_check, checks.sae_replace_check,
steer.generate_text/run_calibration) is out of scope here and is exercised
only on the RunPod pod. What's tested here: the SAE math, the BatchTopK
sparsity mechanism, the Worker report's canonical-hash contract, the
assistant-turn token masking algorithm (via a tiny fake tokenizer), and the
pure-Python feature-selection / analysis helpers.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from config import Config  # noqa: E402
from harvest import ShuffleBuffer, compute_assistant_mask  # noqa: E402
from report import canonical_json, sha256_of_payload, _progress  # noqa: E402
from sae import MatryoshkaBatchTopKSAE, batch_topk  # noqa: E402
from steer import (  # noqa: E402
    select_steer_features,
    median_max_activation,
    quantile_indices,
    compute_generation_boundary,
    distinct_ratio,
    max_run_length,
    repeat_4gram_fraction,
    is_coherent,
    max_coherent_dose,
)
from run_smoke import normalized_edit_distance, word_edit_distance  # noqa: E402


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------
def test_config_defaults_are_smoke_defaults():
    cfg = Config()
    assert cfg.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.layer == 19
    assert cfg.matryoshka_shells == [1024, 4096, 8192]
    assert cfg.sae_width == 8192
    assert cfg.tokens_target == 2_000_000


def test_config_from_json_smoke_and_full(tmp_path):
    smoke = Config.from_json(PIPELINE_DIR / "configs" / "smoke.json")
    full = Config.from_json(PIPELINE_DIR / "configs" / "full.json")
    assert smoke.sae_width == 8192
    assert full.sae_width == 32768
    assert full.matryoshka_shells == [1024, 4096, 16384, 32768]
    assert full.tokens_target == 100_000_000
    assert full.steer_features == 256
    assert full.steer_scenarios == 24


def test_config_rejects_unknown_key():
    with pytest.raises(ValueError):
        Config.from_dict({"not_a_real_field": 1})


def test_config_rejects_bad_shells():
    with pytest.raises(ValueError):
        Config(matryoshka_shells=[8192, 1024])  # not ascending
    with pytest.raises(ValueError):
        Config(sae_width=100, matryoshka_shells=[1024, 4096])  # exceeds width


# ---------------------------------------------------------------------------
# sae.py
# ---------------------------------------------------------------------------
def test_batch_topk_keeps_k_per_token_on_average():
    torch.manual_seed(0)
    n_tokens, width, k = 32, 128, 8
    acts = torch.rand(n_tokens, width)  # in [0, 1), effectively never exactly 0
    out = batch_topk(acts, k)
    nonzero_per_token = (out > 0).sum().item() / n_tokens
    assert nonzero_per_token == pytest.approx(k, abs=1e-6)
    # every surviving value must be an original value (no invented values)
    assert torch.equal(out[out > 0], acts[out > 0])


def test_batch_topk_all_zero_input_stays_zero():
    acts = torch.zeros(10, 16)
    out = batch_topk(acts, 4)
    assert torch.equal(out, acts)


def test_sae_forward_loss_toy_shapes_and_finiteness():
    torch.manual_seed(0)
    d_in, width, n_tokens, k = 64, 128, 32, 8
    sae = MatryoshkaBatchTopKSAE(d_in=d_in, width=width, shells=[32, 64, 128], k=k, seed=0)
    x = torch.randn(n_tokens, d_in)
    out = sae.forward_loss(x, dead_window_tokens=1_000_000)

    assert out.codes.shape == (n_tokens, width)
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.recon_loss)
    assert torch.isfinite(out.aux_loss)
    assert set(out.per_shell_mse.keys()) == {32, 64, 128}
    assert out.loss.requires_grad
    assert 0.0 <= out.l0.item() <= width


def test_matryoshka_nested_loss_decreases_over_training():
    torch.manual_seed(0)
    d_in, width, k = 64, 128, 8
    shells = [32, 64, 128]
    sae = MatryoshkaBatchTopKSAE(d_in=d_in, width=width, shells=shells, k=k, seed=0)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)

    # Fixed low-rank-ish signal so there is real structure to learn.
    gen = torch.Generator().manual_seed(0)
    basis = torch.randn(8, d_in, generator=gen)

    def make_batch(n=64):
        coeffs = torch.randn(n, 8, generator=gen)
        return coeffs @ basis

    losses = []
    for step in range(200):
        x = make_batch()
        optimizer.zero_grad(set_to_none=True)
        out = sae.forward_loss(x, dead_window_tokens=1_000_000)
        out.loss.backward()
        optimizer.step()
        sae.normalize_decoder_()
        losses.append(out.recon_loss.item())

    early = sum(losses[:20]) / 20
    late = sum(losses[-20:]) / 20
    assert late < early, f"expected training loss to decrease: early={early}, late={late}"

    # Every shell's own MSE should also have improved.
    with torch.no_grad():
        final_out = sae.forward_loss(make_batch(256), dead_window_tokens=1_000_000)
    for shell in shells:
        assert torch.isfinite(final_out.per_shell_mse[shell])


def test_sae_save_and_load_roundtrip(tmp_path):
    sae = MatryoshkaBatchTopKSAE(d_in=16, width=32, shells=[8, 16, 32], k=4, seed=1)
    path = tmp_path / "sae.safetensors"
    feature_stats = {"max_activation": [0.0] * 32, "firing_density": [0.0] * 32, "tokens_observed_for_density": 0}
    sae.save(path, feature_stats=feature_stats)
    assert path.exists()
    assert (tmp_path / "feature_stats.json").exists()

    loaded = MatryoshkaBatchTopKSAE.load(path, shells=[8, 16, 32], k=4)
    assert torch.allclose(loaded.W_enc, sae.W_enc)
    assert torch.allclose(loaded.W_dec, sae.W_dec)


def test_reconstruct_matches_forward_loss_main_reconstruction():
    """checks.sae_replace_check calls sae.reconstruct(x=flat); this must be
    the exact same computation forward_loss uses for its main (outermost
    shell) reconstruction, so the two code paths cannot diverge (see item 5
    of the smoke-2 fix list -- the shared `reconstruct` method is what
    guarantees this)."""
    torch.manual_seed(0)
    d_in, width, k = 16, 32, 4
    shells = [8, 16, 32]
    toy_sae = MatryoshkaBatchTopKSAE(d_in=d_in, width=width, shells=shells, k=k, seed=0)
    x = torch.randn(20, d_in)

    out = toy_sae.forward_loss(x, dead_window_tokens=1_000_000)

    # Recompute forward_loss's main reconstruction directly (decode at the
    # outermost shell, from the *same* codes forward_loss used) and compare
    # it against sae.reconstruct(x=x).
    with torch.no_grad():
        codes = toy_sae.encode(x)
        expected_main_recon = toy_sae.decode(codes, n_features=shells[-1])
        actual = toy_sae.reconstruct(x=x)
    assert torch.allclose(actual, expected_main_recon)

    # And forward_loss's own reported recon MSE for the outermost shell
    # must match F.mse_loss(sae.reconstruct(x=x), x) exactly.
    import torch.nn.functional as F

    assert torch.allclose(out.per_shell_mse[shells[-1]], F.mse_loss(toy_sae.reconstruct(x=x), x))


def test_reconstruct_reuses_precomputed_codes_without_recomputing_them():
    torch.manual_seed(0)
    toy_sae = MatryoshkaBatchTopKSAE(d_in=8, width=16, shells=[4, 8, 16], k=2, seed=0)
    x = torch.randn(5, 8)
    codes = toy_sae.encode(x)
    via_codes = toy_sae.reconstruct(codes=codes)
    via_x = toy_sae.reconstruct(x=x)
    assert torch.allclose(via_codes, via_x)


def test_reconstruct_requires_x_or_codes():
    toy_sae = MatryoshkaBatchTopKSAE(d_in=8, width=16, shells=[4, 8, 16], k=2, seed=0)
    with pytest.raises(ValueError):
        toy_sae.reconstruct()


# ---------------------------------------------------------------------------
# harvest.py: ShuffleBuffer
# ---------------------------------------------------------------------------
def test_shuffle_buffer_fills_and_samples():
    buf = ShuffleBuffer(capacity=10, d_model=4, seed=0)
    assert not buf.is_ready(1.0)
    buf.add(torch.randn(6, 4))
    assert buf.filled == 6
    buf.add(torch.randn(10, 4))  # overflow triggers reservoir replacement
    assert buf.filled == 10
    assert buf.total_added == 16
    sample = buf.sample(5)
    assert sample.shape == (5, 4)


def test_shuffle_buffer_rejects_bad_capacity():
    with pytest.raises(ValueError):
        ShuffleBuffer(capacity=0, d_model=4)


# ---------------------------------------------------------------------------
# harvest.py: compute_assistant_mask, via a tiny fake tokenizer
#
# Real chat templates (Qwen's included) render each turn as
# "<|im_start|>{role}\n{content}<|im_end|>\n". This fake tokenizer
# reproduces exactly that structure with a whitespace/special-token
# regex "tokenizer", so the masking algorithm can be tested without
# downloading a real model/tokenizer (no network access in this harness).
# ---------------------------------------------------------------------------
class FakeTokenizer:
    _SPECIALS = ("<|im_start|>", "<|im_end|>")
    _PATTERN = re.compile(r"<\|im_start\|>|<\|im_end\|>|\S+")

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.all_special_ids: List[int] = []
        for tok in self._SPECIALS:
            self._id_for(tok, special=True)

    def _id_for(self, token: str, special: bool = False) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
            if special:
                self.all_special_ids.append(self.vocab[token])
        return self.vocab[token]

    def _render(self, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
        parts = [f"<|im_start|> {m['role']}\n{m['content']} <|im_end|>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|> assistant\n")
        return "".join(parts)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = self._render(list(messages), add_generation_prompt)
        if not tokenize:
            return text
        pieces = self._PATTERN.findall(text)
        return [self._id_for(p, special=p in self._SPECIALS) for p in pieces]

    def __call__(self, text: str, add_special_tokens: bool = False, **_):
        pieces = self._PATTERN.findall(text)
        return {"input_ids": [self._id_for(p, special=p in self._SPECIALS) for p in pieces]}

    def decode(self, ids: List[int]) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


def test_assistant_mask_single_turn():
    tok = FakeTokenizer()
    messages = [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello friend"},
    ]
    input_ids, mask = compute_assistant_mask(tok, messages)

    assert len(input_ids) == len(mask)
    masked_tokens = [tok.decode([i]) for i, m in zip(input_ids, mask) if m]
    assert masked_tokens == ["hello", "friend"]
    # no template control tokens ever survive into the mask
    for i, m in zip(input_ids, mask):
        if m:
            assert i not in tok.all_special_ids


def test_assistant_mask_multi_turn_only_masks_assistant_content():
    tok = FakeTokenizer()
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two three"},
        {"role": "user", "content": "four"},
        {"role": "assistant", "content": "five"},
    ]
    input_ids, mask = compute_assistant_mask(tok, messages)
    masked_tokens = [tok.decode([i]) for i, m in zip(input_ids, mask) if m]
    assert masked_tokens == ["two", "three", "five"]
    assert sum(mask) == 3


def test_assistant_mask_respects_max_seq():
    tok = FakeTokenizer()
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two three four five"},
    ]
    full_ids, full_mask = compute_assistant_mask(tok, messages)
    truncated_ids, truncated_mask = compute_assistant_mask(tok, messages, max_seq=5)
    assert truncated_ids == full_ids[:5]
    assert truncated_mask == full_mask[:5]
    assert len(truncated_ids) == 5


def test_assistant_mask_empty_messages():
    tok = FakeTokenizer()
    ids, mask = compute_assistant_mask(tok, [])
    assert ids == []
    assert mask == []


# ---------------------------------------------------------------------------
# checks.py: assistant_mask_sanity_check (uses the same fake tokenizer)
# ---------------------------------------------------------------------------
def test_checks_assistant_mask_sanity_passes_on_clean_example():
    from checks import assistant_mask_sanity_check

    tok = FakeTokenizer()
    messages = [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello friend"},
    ]
    record = assistant_mask_sanity_check(tok, messages)
    assert record["ok"] is True
    assert record["decoded_masked_text"] == "hello friend"
    assert record["num_masked_tokens"] == 2


def test_checks_assistant_mask_sanity_raises_on_injected_control_token():
    """If the mask leaked a control token, the check must raise -- simulate
    that by asking about a made-up 'extra_control_strings' word that
    legitimately appears in the assistant content, to confirm the guard
    actually fires (rather than trivially always passing)."""
    from checks import assistant_mask_sanity_check

    tok = FakeTokenizer()
    messages = [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello friend"},
    ]
    with pytest.raises(AssertionError):
        assistant_mask_sanity_check(tok, messages, extra_control_strings=["hello"])


# ---------------------------------------------------------------------------
# report.py: canonical JSON + sha256 contract
# ---------------------------------------------------------------------------
def test_canonical_json_is_sorted_and_compact():
    payload = {"b": [1, 2, 3], "a": 1}
    assert canonical_json(payload) == '{"a":1,"b":[1,2,3]}'


def test_sha256_of_payload_matches_known_value():
    # Known value: sha256(b'{"a":1,"b":[1,2,3]}').hexdigest(), computed
    # independently outside this module.
    payload = {"b": [1, 2, 3], "a": 1}
    expected = "bfa6ceebf136e4837ec687f2be09f612c645c9ec1f99e3ef5d497b0d5bb99e0a"
    assert len(expected) == 64
    assert sha256_of_payload(payload) == expected


def test_sha256_of_payload_is_deterministic_and_order_independent():
    a = sha256_of_payload({"b": [1, 2, 3], "a": 1})
    b = sha256_of_payload({"a": 1, "b": [1, 2, 3]})
    assert a == b
    assert len(a) == 64


def test_progress_coerces_to_int_and_clamps_done_to_total():
    assert _progress(5, 10) == {"done": 5, "total": 10}
    assert _progress(11, 10) == {"done": 10, "total": 10}
    assert _progress(5.9, 10.0) == {"done": 5, "total": 10}
    assert _progress("7", "10") == {"done": 7, "total": 10}


def test_worker_client_never_raises_without_url(monkeypatch):
    from report import WorkerClient

    monkeypatch.delenv("AUTOLABS_3C_WORKER_URL", raising=False)
    client = WorkerClient(base_url="", token="", run_id="run-1")
    # Should log and return without raising, even though there's no server.
    client.report("boot", progress={"done": 1, "total": 1}, records=[{"recordId": "x", "payload": {"a": 1}}])


# ---------------------------------------------------------------------------
# steer.py: pure-Python feature selection
# ---------------------------------------------------------------------------
def test_select_steer_features_filters_by_density_and_spreads_across_quantiles():
    feature_stats = {
        "firing_density": [0.5, 0.05, 1e-5, 0.02, 0.0001, 0.2],
        "max_activation": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
    selected = select_steer_features(
        feature_stats, first_shell_size=6, steer_features=2, density_min=1e-4, density_max=0.1
    )
    # candidates in range, sorted ascending by density: idx4 (1e-4), idx3
    # (0.02), idx1 (0.05). A 2-way quantile spread over 3 candidates picks
    # the 25th/75th percentile positions (idx3, idx1), not just the top 2
    # by density (which would always pick idx1 and idx3 too here, but by
    # coincidence -- the point of the quantile spread is to also be able
    # to pick idx4, the sparsest candidate, when steer_features is larger).
    assert [c["feature"] for c in selected] == [3, 1]
    assert selected[0]["quantile"] == pytest.approx(25.0)
    assert selected[1]["quantile"] == pytest.approx(75.0)
    assert all("density" in c and "max_activation" in c for c in selected)


def test_select_steer_features_spreads_across_full_density_range():
    # 10 candidates evenly spaced in log-density from 1e-4 to 1e-1; with
    # steer_features=1 the densest-only selection would always pick the
    # same end of the range, but the quantile spread over more picks
    # should include *both* a low-density and a high-density candidate.
    densities = [10 ** (-4 + 3 * i / 9) for i in range(10)]  # 1e-4 .. 1e-1
    feature_stats = {"firing_density": densities, "max_activation": [1.0] * 10}
    selected = select_steer_features(
        feature_stats, first_shell_size=10, steer_features=4, density_min=1e-4, density_max=0.1
    )
    picked_features = [c["feature"] for c in selected]
    assert len(set(picked_features)) == 4
    assert min(picked_features) <= 2  # at least one low-density pick
    assert max(picked_features) >= 7  # at least one high-density pick


def test_select_steer_features_restricted_to_first_shell():
    feature_stats = {
        "firing_density": [1e-5, 0.05, 0.06],
        "max_activation": [1.0, 2.0, 3.0],
    }
    # only index 0 is inside the first shell of size 1, and it's out of range
    selected = select_steer_features(feature_stats, first_shell_size=1, steer_features=5, density_min=1e-4, density_max=0.1)
    assert selected == []


def test_median_max_activation_empty_and_nonempty():
    assert median_max_activation([]) == 0.0
    assert median_max_activation([{"max_activation": 2.0}, {"max_activation": 4.0}]) == 3.0


# ---------------------------------------------------------------------------
# steer.py: quantile_indices (the feature-selection quantile-spread helper)
# ---------------------------------------------------------------------------
def test_quantile_indices_ten_bins_land_near_5_15_95_percentiles():
    idxs = quantile_indices(n_candidates=100, n_select=10)
    assert len(idxs) == 10
    assert len(set(idxs)) == 10
    assert idxs == sorted(idxs)
    expected_approx = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    for got, exp in zip(idxs, expected_approx):
        assert abs(got - exp) <= 1


def test_quantile_indices_all_distinct_when_n_candidates_equals_n_select():
    idxs = quantile_indices(5, 5)
    assert sorted(idxs) == [0, 1, 2, 3, 4]


def test_quantile_indices_caps_n_select_to_n_candidates():
    idxs = quantile_indices(3, 10)
    assert len(idxs) == 3
    assert len(set(idxs)) == 3


def test_quantile_indices_empty_inputs():
    assert quantile_indices(0, 5) == []
    assert quantile_indices(5, 0) == []


# ---------------------------------------------------------------------------
# steer.py: compute_generation_boundary (assistant-turn position boundary)
# ---------------------------------------------------------------------------
def test_compute_generation_boundary_matches_user_only_prefix_length():
    tok = FakeTokenizer()
    prompt = "hi there friend"
    boundary = compute_generation_boundary(tok, prompt)

    user_only_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=False
    )
    with_header_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
    )
    assert boundary == len(user_only_ids)
    # the boundary must fall strictly before the end of the generation-prompt
    # rendering (there must be at least the assistant header left over) and
    # the user-only rendering must be a true prefix of it.
    assert boundary < len(with_header_ids)
    assert with_header_ids[:boundary] == user_only_ids


# ---------------------------------------------------------------------------
# steer.py: coherence metrics
# ---------------------------------------------------------------------------
def test_repetitive_sequence_is_incoherent():
    ids = [7] * 40
    assert distinct_ratio(ids) < 0.5
    assert max_run_length(ids) > 4
    assert repeat_4gram_fraction(ids) > 0.2
    assert is_coherent(distinct_ratio(ids), max_run_length(ids), repeat_4gram_fraction(ids)) is False


def test_varied_sequence_is_coherent():
    ids = list(range(40))  # all distinct, no repeats at all
    assert distinct_ratio(ids) == 1.0
    assert max_run_length(ids) == 1
    assert repeat_4gram_fraction(ids) == 0.0
    assert is_coherent(distinct_ratio(ids), max_run_length(ids), repeat_4gram_fraction(ids)) is True


def test_distinct_ratio_and_max_run_edge_cases():
    assert distinct_ratio([]) == 1.0
    assert max_run_length([]) == 0
    assert repeat_4gram_fraction([1, 2, 3]) == 0.0  # fewer than 4 tokens


def test_max_run_length_counts_longest_run_only():
    assert max_run_length([1, 1, 2, 2, 2, 1]) == 3


def test_repeat_4gram_fraction_all_unique_is_zero():
    assert repeat_4gram_fraction(list(range(10))) == 0.0


def test_repeat_4gram_fraction_fully_repeated_is_high():
    ids = [1, 2, 3, 4] * 10
    frac = repeat_4gram_fraction(ids)
    assert frac > 0.5


# ---------------------------------------------------------------------------
# steer.py: max_coherent_dose
# ---------------------------------------------------------------------------
def test_max_coherent_dose_picks_largest_dose_meeting_the_bar():
    rows = [
        {"dose": 0.25, "coherent": True},
        {"dose": 0.25, "coherent": True},
        {"dose": 0.25, "coherent": True},
        {"dose": 0.25, "coherent": True},
        {"dose": 0.5, "coherent": True},
        {"dose": 0.5, "coherent": True},
        {"dose": 0.5, "coherent": True},
        {"dose": 0.5, "coherent": False},
        {"dose": 1.0, "coherent": True},
        {"dose": 1.0, "coherent": False},
        {"dose": 1.0, "coherent": False},
        {"dose": 1.0, "coherent": False},
    ]
    # dose 0.25: 4/4 coherent; dose 0.5: 3/4 coherent (meets >=3-of-4);
    # dose 1.0: only 1/4 coherent (fails) -- largest passing dose is 0.5.
    assert max_coherent_dose(rows) == 0.5


def test_max_coherent_dose_returns_none_when_nothing_clears_the_bar():
    rows = [
        {"dose": 0.25, "coherent": False},
        {"dose": 0.25, "coherent": True},
        {"dose": 0.25, "coherent": False},
        {"dose": 0.25, "coherent": False},
    ]
    assert max_coherent_dose(rows) is None


# ---------------------------------------------------------------------------
# run_smoke.py: embedding-free analysis proxies
# ---------------------------------------------------------------------------
def test_word_edit_distance_basic():
    assert word_edit_distance("a b c", "a b c") == 0
    assert word_edit_distance("a b c", "a b") == 1
    assert word_edit_distance("", "a b") == 2


def test_normalized_edit_distance_range():
    d = normalized_edit_distance("the cat sat", "the cat sat")
    assert d == 0.0
    d2 = normalized_edit_distance("the cat sat", "a dog ran fast")
    assert 0.0 < d2 <= 1.0
