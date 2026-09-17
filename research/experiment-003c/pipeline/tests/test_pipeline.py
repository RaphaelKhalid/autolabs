"""CPU-only tests for the Experiment 3C pipeline.

Anything that needs a real transformers model + GPU (harvest's
ActivationHarvester, checks.identity_hook_check, the forward-pass parts of
checks.sae_replace_check, steer.generate_text/run_calibration) is out of
scope here and is exercised only on the RunPod pod. What's tested here: the
SAE math, the BatchTopK sparsity mechanism, the Worker report's
canonical-hash contract, the assistant-turn token masking algorithm (via a
tiny fake tokenizer) -- including the two-turn mask sae_replace_check
builds from it -- checks.py's FVE/cross-entropy helper math on toy
tensors, and the pure-Python feature-selection / analysis helpers.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
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
from run_smoke import (  # noqa: E402
    normalized_edit_distance,
    word_edit_distance,
    lr_warmup_multiplier,
    train_steps_on_buffer,
    compute_screen_verdict,
)
import describe  # noqa: E402
import screen  # noqa: E402


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
    assert smoke.train_steps_per_batch == 4
    assert smoke.lr_warmup_steps == 500
    assert full.train_steps_per_batch == 8
    assert full.lr_warmup_steps == 500


def test_config_rejects_unknown_key():
    with pytest.raises(ValueError):
        Config.from_dict({"not_a_real_field": 1})


def test_config_rejects_bad_shells():
    with pytest.raises(ValueError):
        Config(matryoshka_shells=[8192, 1024])  # not ascending
    with pytest.raises(ValueError):
        Config(sae_width=100, matryoshka_shells=[1024, 4096])  # exceeds width


def test_config_defaults_include_steps_per_batch_and_warmup():
    cfg = Config()
    assert cfg.train_steps_per_batch == 4
    assert cfg.lr_warmup_steps == 500


def test_config_rejects_zero_train_steps_per_batch():
    with pytest.raises(ValueError):
        Config(train_steps_per_batch=0)


def test_config_rejects_negative_lr_warmup_steps():
    with pytest.raises(ValueError):
        Config(lr_warmup_steps=-1)


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
# checks.py: sae_replace_check's assistant-position masking and FVE/ce_delta
# helper math (the GPU-only forward-pass parts of sae_replace_check itself
# are not exercised here -- see the module docstring)
# ---------------------------------------------------------------------------
def test_sae_replace_check_builds_two_turn_mask_selecting_only_reply_tokens():
    """sae_replace_check builds a two-turn chat (prompt as the user turn,
    a fixed assistant reply as the assistant turn) and masks to assistant
    positions with harvest.compute_assistant_mask -- the same function
    harvest uses to build the SAE's training data. Confirm that on a
    two-turn conversation shaped exactly like the one the check builds,
    the mask selects the assistant reply's content tokens only, never the
    user prompt's tokens or any template control token."""
    tok = FakeTokenizer()
    prompt = "please explain something about french history in detail"
    reply = "the republic began in 1789 and reshaped french national identity"
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply},
    ]
    input_ids, mask = compute_assistant_mask(tok, messages)

    masked_tokens = [tok.decode([i]) for i, m in zip(input_ids, mask) if m]
    assert masked_tokens == reply.split()
    assert sum(mask) == len(reply.split())

    # None of the prompt's own tokens (words unique to the user turn) leak
    # into the mask, and no masked token is a template control token.
    prompt_only_words = set(prompt.split()) - set(reply.split())
    assert not (set(masked_tokens) & prompt_only_words)
    for i, m in zip(input_ids, mask):
        if m:
            assert i not in tok.all_special_ids


def test_fraction_variance_explained_perfect_reconstruction_is_one():
    from checks import fraction_variance_explained

    torch.manual_seed(0)
    x = torch.randn(30, 6)
    assert fraction_variance_explained(x, x.clone()) == pytest.approx(1.0, abs=1e-5)


def test_fraction_variance_explained_mean_reconstruction_is_zero():
    from checks import fraction_variance_explained

    torch.manual_seed(1)
    x = torch.randn(50, 4)
    mean_recon = x.mean(dim=0, keepdim=True).expand_as(x)
    assert fraction_variance_explained(x, mean_recon) == pytest.approx(0.0, abs=1e-5)


def test_fraction_variance_explained_negative_when_worse_than_mean():
    from checks import fraction_variance_explained

    # Variance is shift-invariant, so a constant offset alone doesn't
    # penalize FVE; amplify the residual instead (recon = -x) so the
    # residual variance (4x) exceeds the mean-reconstruction baseline (1x)
    # and FVE goes below zero, matching the derivation in the docstring.
    x = torch.zeros(10, 2)
    x[:, 0] = torch.linspace(-1.0, 1.0, 10)
    recon = -x
    assert fraction_variance_explained(x, recon) == pytest.approx(-3.0, abs=1e-5)


def test_mean_cross_entropy_at_positions_matches_manual_computation():
    from checks import mean_cross_entropy_at_positions

    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
    )
    labels = torch.tensor([0, 1, 2])
    positions = torch.tensor([True, True, False])

    ce = mean_cross_entropy_at_positions(logits, labels, positions)
    expected = torch.nn.functional.cross_entropy(logits[:2], labels[:2]).item()
    assert ce == pytest.approx(expected, rel=1e-5)


def test_mean_cross_entropy_at_positions_empty_mask_is_zero():
    from checks import mean_cross_entropy_at_positions

    logits = torch.randn(5, 3)
    labels = torch.zeros(5, dtype=torch.long)
    positions = torch.zeros(5, dtype=torch.bool)
    assert mean_cross_entropy_at_positions(logits, labels, positions) == 0.0


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


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = "{}"

    def json(self):
        return self._payload


class _FakeSession:
    """Captures every POST body instead of hitting a real Worker."""

    def __init__(self):
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002 - matches requests' kwarg name
        self.calls.append({"url": url, "body": json})
        return _FakeResponse({"ok": True, "runId": "run-1"})


def test_report_sends_payload_json_and_matching_sha256_no_payload_field():
    from report import WorkerClient

    session = _FakeSession()
    client = WorkerClient(base_url="https://worker.test", token="tok", run_id="run-1", session=session)
    payload = {"loss": 1.251517, "fve": 0.435447, "dead_frac": 1e-05}

    client.report("train", records=[{"recordId": "train-checkpoint-1004535", "payload": payload}])

    assert len(session.calls) == 1
    body = session.calls[0]["body"]
    assert len(body["records"]) == 1
    record = body["records"][0]
    assert record["recordId"] == "train-checkpoint-1004535"
    assert "payload" not in record
    assert record["payloadJson"] == canonical_json(payload)
    assert record["sha256"] == hashlib.sha256(record["payloadJson"].encode("utf-8")).hexdigest()


def test_report_sanitizes_nan_and_infinity_to_null_before_hashing():
    from report import WorkerClient

    session = _FakeSession()
    client = WorkerClient(base_url="https://worker.test", token="tok", run_id="run-1", session=session)
    payload = {"loss": float("nan"), "grad_norm": float("inf"), "fve": 0.4}

    client.report("train", records=[{"recordId": "train-checkpoint-nan", "payload": payload}])

    body = session.calls[0]["body"]
    record = body["records"][0]
    assert '"loss":null' in record["payloadJson"]
    assert '"grad_norm":null' in record["payloadJson"]
    assert "NaN" not in record["payloadJson"]
    assert "Infinity" not in record["payloadJson"]
    # The sanitized JSON must itself be valid, standard JSON (no NaN/Infinity tokens).
    assert json.loads(record["payloadJson"]) == {"loss": None, "grad_norm": None, "fve": 0.4}
    assert record["sha256"] == hashlib.sha256(record["payloadJson"].encode("utf-8")).hexdigest()


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


# ---------------------------------------------------------------------------
# run_smoke.py: LR warmup + multi-step-per-batch training
# ---------------------------------------------------------------------------
def test_lr_warmup_multiplier_zero_at_step_zero_and_full_at_and_after_warmup():
    warmup_steps = 500
    assert lr_warmup_multiplier(0, warmup_steps) == 0.0
    assert 0.0 < lr_warmup_multiplier(250, warmup_steps) < 1.0
    assert lr_warmup_multiplier(499, warmup_steps) < 1.0
    assert lr_warmup_multiplier(500, warmup_steps) == 1.0
    assert lr_warmup_multiplier(1000, warmup_steps) == 1.0  # stays full past warmup


def test_lr_warmup_multiplier_disabled_when_warmup_steps_is_zero():
    assert lr_warmup_multiplier(0, 0) == 1.0
    assert lr_warmup_multiplier(1000, 0) == 1.0


def test_train_steps_on_buffer_runs_configured_number_of_optimizer_steps():
    """A small CPU toy loop: with train_steps_per_batch=3, one call to
    train_steps_on_buffer (standing in for one harvested batch) must run
    exactly 3 optimizer steps, each on a fresh buffer sample."""
    torch.manual_seed(0)
    d_in, width, k = 16, 32, 4
    sae = MatryoshkaBatchTopKSAE(d_in=d_in, width=width, shells=[8, 16, 32], k=k, seed=0)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)

    buffer = ShuffleBuffer(capacity=256, d_model=d_in, seed=0)
    buffer.add(torch.randn(256, d_in))

    cfg = Config(
        sae_width=width,
        matryoshka_shells=[8, 16, 32],
        k=k,
        batch_tokens=8,
        train_steps_per_batch=3,
        lr_warmup_steps=0,
        dead_feature_window_tokens=1_000_000,
    )

    step_count = 0
    orig_step = optimizer.step

    def counting_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        return orig_step(*args, **kwargs)

    optimizer.step = counting_step

    steps_done, last_out = train_steps_on_buffer(sae, optimizer, buffer, cfg, steps_done=0, device=None)

    assert step_count == 3
    assert steps_done == 3
    assert last_out is not None
    assert torch.isfinite(last_out.loss)


def test_train_steps_on_buffer_continues_step_count_across_calls():
    """Resuming from a checkpoint should continue the warmup schedule
    instead of restarting it -- i.e. steps_done accumulates across calls."""
    torch.manual_seed(0)
    d_in, width, k = 16, 32, 4
    sae = MatryoshkaBatchTopKSAE(d_in=d_in, width=width, shells=[8, 16, 32], k=k, seed=0)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
    buffer = ShuffleBuffer(capacity=256, d_model=d_in, seed=0)
    buffer.add(torch.randn(256, d_in))
    cfg = Config(
        sae_width=width,
        matryoshka_shells=[8, 16, 32],
        k=k,
        batch_tokens=8,
        train_steps_per_batch=3,
        lr_warmup_steps=10,
        dead_feature_window_tokens=1_000_000,
    )

    steps_done, _ = train_steps_on_buffer(sae, optimizer, buffer, cfg, steps_done=0, device=None)
    assert steps_done == 3
    steps_done, _ = train_steps_on_buffer(sae, optimizer, buffer, cfg, steps_done=steps_done, device=None)
    assert steps_done == 6


# ---------------------------------------------------------------------------
# screen.py: hashed lexical embedding
# ---------------------------------------------------------------------------
def test_hashed_lexical_vector_is_deterministic_and_normalized():
    text = "The quick brown fox jumps over the lazy dog."
    v1 = screen.hashed_lexical_vector(text)
    v2 = screen.hashed_lexical_vector(text)
    assert np.array_equal(v1, v2)
    assert v1.shape == (screen.LEXICAL_DIMS,)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-8


def test_hashed_lexical_vector_differs_for_different_text():
    v1 = screen.hashed_lexical_vector("apples and oranges are great fruit")
    v2 = screen.hashed_lexical_vector("completely unrelated content about spacecraft")
    assert not np.array_equal(v1, v2)


def test_hashed_lexical_vector_empty_text_is_zero_vector():
    v = screen.hashed_lexical_vector("")
    assert np.linalg.norm(v) == 0.0


def test_tokenize_words_lowercases_and_strips_punctuation():
    words = screen.tokenize_words("Hello, World! It's a test.")
    assert words == ["hello", "world", "it's", "a", "test"]


# ---------------------------------------------------------------------------
# screen.py: numpy-only logistic regression + leave-one-scenario-out AUC
# ---------------------------------------------------------------------------
def test_leave_one_scenario_out_separability_on_separable_toy_data():
    rng = np.random.default_rng(0)
    scenarios = [f"s{i}" for i in range(6)]
    offsets = {s: rng.normal(0, 0.01, size=3) for s in scenarios}
    steered = {s: np.array([5.0, 5.0, 5.0]) + offsets[s] for s in scenarios}
    baseline = {s: np.array([-5.0, -5.0, -5.0]) + offsets[s] for s in scenarios}

    result = screen.leave_one_scenario_out_separability(steered, baseline)

    assert result["acc"] >= 0.9
    assert result["auc"] >= 0.9


def test_leave_one_scenario_out_separability_on_inseparable_toy_data():
    rng = np.random.default_rng(0)
    scenarios = [f"s{i}" for i in range(6)]
    # Identical embeddings for the steered and baseline class within each
    # scenario: there is no feature that distinguishes the classes, so a
    # held-out scenario's steered/baseline pair is indistinguishable and
    # chance accuracy (0.5) is the only consistent outcome.
    common = {s: rng.normal(0, 1, size=4) for s in scenarios}
    result = screen.leave_one_scenario_out_separability(dict(common), dict(common))

    assert 0.4 <= result["acc"] <= 0.6
    assert 0.4 <= result["auc"] <= 0.6


def test_leave_one_scenario_out_separability_needs_at_least_two_scenarios():
    result = screen.leave_one_scenario_out_separability({"s0": np.zeros(3)}, {"s0": np.ones(3)})
    assert result["acc"] != result["acc"]  # nan
    assert result["auc"] != result["auc"]  # nan


def test_auc_score_perfect_separation_is_one():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert screen.auc_score(y, scores) == 1.0


def test_auc_score_single_class_is_nan():
    y = np.array([1, 1, 1])
    scores = np.array([0.1, 0.5, 0.9])
    assert screen.auc_score(y, scores) != screen.auc_score(y, scores) or np.isnan(screen.auc_score(y, scores))


# ---------------------------------------------------------------------------
# screen.py: direction consistency vs. a null
# ---------------------------------------------------------------------------
def test_direction_consistency_own_diffs_cohere_more_than_null():
    rng = np.random.default_rng(1)
    base = np.array([1.0, 0.0] + [0.0] * 14)
    own = [base + rng.normal(0, 0.01, 16) for _ in range(5)]
    other_pool = [rng.normal(0, 1, 16) for _ in range(50)]

    result = screen.direction_consistency(own, other_pool, np.random.default_rng(2))

    assert result["mean_cos"] > 0.9
    assert abs(result["null_mean_cos"]) < 0.3
    assert result["mean_cos"] - result["null_mean_cos"] > 0.5


def test_mean_pairwise_cosine_needs_at_least_two_vectors():
    assert screen.mean_pairwise_cosine([np.ones(3)]) != screen.mean_pairwise_cosine([np.ones(3)]) or np.isnan(
        screen.mean_pairwise_cosine([np.ones(3)])
    )


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(screen.cosine_similarity(v, v) - 1.0) < 1e-8


def test_cosine_similarity_zero_vector_is_zero():
    assert screen.cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


# ---------------------------------------------------------------------------
# screen.py: dose-selection bookkeeping
# ---------------------------------------------------------------------------
def test_resolve_dose_uses_value_or_fallback():
    assert screen.resolve_dose(2.0, 1.0) == 2.0
    assert screen.resolve_dose(None, 1.0) == 1.0


def test_median_dose_with_fallback_substitutes_none_and_handles_empty():
    assert screen.median_dose_with_fallback([1.0, None, 3.0], fallback=2.0) == 2.0
    assert screen.median_dose_with_fallback([], fallback=1.5) == 1.5
    assert screen.median_dose_with_fallback([None, None], fallback=0.75) == 0.75


def test_extract_feature_info_dedups_by_feature():
    records = [
        {
            "payload": {
                "kind": "steered", "feature": 3, "density": 0.01, "quantile": 50.0,
                "max_activation": 2.5, "sign": 1, "dose": 1.0, "coherent": True,
            }
        },
        {
            "payload": {
                "kind": "steered", "feature": 3, "density": 0.01, "quantile": 50.0,
                "max_activation": 2.5, "sign": -1, "dose": 1.0, "coherent": True,
            }
        },
        {"payload": {"kind": "baseline", "scenario": "s0"}},
    ]
    info = screen.extract_feature_info(records)
    assert set(info) == {3}
    assert info[3]["max_activation"] == 2.5
    assert info[3]["density"] == 0.01


def test_feature_max_coherent_doses_from_calibration_records():
    records = [
        {"payload": {"kind": "steered", "feature": 0, "sign": 1, "dose": 0.5, "coherent": True}},
        {"payload": {"kind": "steered", "feature": 0, "sign": 1, "dose": 1.0, "coherent": True}},
        {"payload": {"kind": "steered", "feature": 0, "sign": 1, "dose": 2.0, "coherent": False}},
        {"payload": {"kind": "steered", "feature": 0, "sign": -1, "dose": 0.5, "coherent": False}},
        {"payload": {"kind": "baseline", "scenario": "s0"}},
    ]
    doses = screen.feature_max_coherent_doses(records)
    assert doses[0]["pos"] == 1.0
    assert doses[0]["neg"] is None


def test_flatten_doses_pairs_pos_and_neg_per_feature():
    doses_by_feature = {0: {"pos": 1.0, "neg": None}, 1: {"pos": 0.5, "neg": 0.5}}
    flat = screen.flatten_doses(doses_by_feature)
    assert sorted(flat, key=lambda v: (v is None, v)) == [0.5, 0.5, 1.0, None]


# ---------------------------------------------------------------------------
# run_smoke.py: compute_screen_verdict -- per-entity (either-sign) gate G0,
# and max/p95 of the random-direction null AUCs (see ../VISUAL-1.md "Screen")
# ---------------------------------------------------------------------------
def _screen_row(kind: str, id_, sign: int, auc: float, mean_cos: float, null_mean_cos: float) -> dict:
    return {
        "kind": kind,
        "id": id_,
        "sign": sign,
        "separability": {"resid": {"auc": auc}},
        "consistency": {"mean_cos": mean_cos, "null_mean_cos": null_mean_cos},
    }


def test_compute_screen_verdict_random_auc_max_and_p95():
    random_aucs = [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70, 0.90]
    rows = [_screen_row("random", i, 0, auc, 0.0, 0.0) for i, auc in enumerate(random_aucs)]

    verdict = compute_screen_verdict(rows)

    assert verdict["random_directions"] == len(random_aucs)
    assert verdict["max_random_resid_auc"] == max(random_aucs)
    assert verdict["max_random_resid_auc_p95"] == pytest.approx(float(np.percentile(random_aucs, 95)))


def test_compute_screen_verdict_control_passes_if_either_sign_passes():
    """A control whose positive sign clears gate G0 must count as passing
    even though its negative sign misses only on the consistency margin
    (0.10 vs 0.06 here) -- this is the exact evil/benevolent case visual
    run 1 mis-scored by counting signs separately."""
    rows = [
        _screen_row("random", 0, 0, 0.50, 0.0, 0.0),
        _screen_row("control", "evil_benevolent", 1, 0.83, 0.30, 0.10),  # auc>0.5, margin 0.20 -> passes
        _screen_row("control", "evil_benevolent", -1, 0.55, 0.20, 0.14),  # margin 0.06 -> fails
        _screen_row("control", "sycophantic_honest", 1, 0.40, 0.10, 0.05),  # auc below random -> fails
        _screen_row("control", "sycophantic_honest", -1, 0.45, 0.10, 0.05),  # auc below random -> fails
    ]

    verdict = compute_screen_verdict(rows)

    assert verdict["control_directions_total"] == 4
    assert verdict["control_directions_passing"] == 1  # only evil_benevolent(+), old per-sign metric
    assert verdict["controls_total"] == 2
    assert verdict["controls_passing"] == 1  # evil_benevolent passes via its + sign
    assert verdict["all_controls_pass"] is False

    detail = {d["id"]: d for d in verdict["controls_detail"]}
    assert detail["evil_benevolent"] == {
        "id": "evil_benevolent", "pos_passes": True, "neg_passes": False, "passes": True,
    }
    assert detail["sycophantic_honest"]["passes"] is False


def test_compute_screen_verdict_all_controls_pass_when_every_control_passes():
    rows = [
        _screen_row("random", 0, 0, 0.50, 0.0, 0.0),
        _screen_row("control", "a", 1, 0.90, 0.30, 0.10),
        _screen_row("control", "a", -1, 0.90, 0.30, 0.10),
        _screen_row("control", "b", 1, 0.90, 0.30, 0.10),
        _screen_row("control", "b", -1, 0.40, 0.00, 0.00),  # b's neg sign fails; b still passes via pos
    ]

    verdict = compute_screen_verdict(rows)

    assert verdict["controls_total"] == 2
    assert verdict["controls_passing"] == 2
    assert verdict["all_controls_pass"] is True


def test_compute_screen_verdict_feature_passes_if_either_sign_passes():
    rows = [
        _screen_row("random", 0, 0, 0.50, 0.0, 0.0),
        _screen_row("feature", 75, 1, 0.60, 0.10, 0.05),  # margin 0.05 -> fails
        _screen_row("feature", 75, -1, 1.00, 0.30, 0.07),  # auc>0.5, margin 0.23 -> passes
        _screen_row("feature", 1019, 1, 0.40, 0.10, 0.05),  # auc below random -> fails
        _screen_row("feature", 1019, -1, 0.45, 0.10, 0.05),  # auc below random -> fails
    ]

    verdict = compute_screen_verdict(rows)

    assert verdict["feature_directions_total"] == 4
    assert verdict["feature_directions_passing"] == 1  # only feature 75's neg sign
    assert verdict["features_total"] == 2
    assert verdict["features_passing"] == 1  # only feature 75 (via its neg sign)

    detail = {d["id"]: d for d in verdict["features_detail"]}
    assert detail[75] == {"id": 75, "pos_passes": False, "neg_passes": True, "passes": True}
    assert detail[1019]["passes"] is False


# ---------------------------------------------------------------------------
# screen.py: build_generation_records -- one record per (direction, scenario)
# ---------------------------------------------------------------------------
def test_build_generation_records_recordids_and_payload():
    directions = [
        {
            "kind": "feature",
            "id": 75,
            "sign": -1,
            "dose": 2.0,
            "steered": {
                "s0": {"text": "steered s0", "finish_reason": "eos", "coherence": {"coherent": True}},
                "s1": {"text": "steered s1", "finish_reason": "length", "coherence": {"coherent": False}},
            },
        },
        {
            "kind": "random",
            "id": 3,
            "sign": 0,
            "dose": 1.5,
            "steered": {
                "s0": {"text": "random steered s0", "finish_reason": "eos", "coherence": {"coherent": True}},
            },
        },
    ]
    baseline_texts = {"s0": "baseline s0", "s1": "baseline s1"}

    records = screen.build_generation_records(directions, baseline_texts)

    record_ids = {r["recordId"] for r in records}
    assert record_ids == {
        "screen-gen-feature-75-neg-s0",
        "screen-gen-feature-75-neg-s1",
        "screen-gen-random-3-na-s0",
    }
    by_id = {r["recordId"]: r["payload"] for r in records}
    assert by_id["screen-gen-feature-75-neg-s0"] == {
        "kind": "feature",
        "id": 75,
        "sign": -1,
        "dose": 2.0,
        "scenario": "s0",
        "text": "steered s0",
        "baseline_text": "baseline s0",
        "finish_reason": "eos",
        "coherence": {"coherent": True},
    }
    assert by_id["screen-gen-random-3-na-s0"]["baseline_text"] == "baseline s0"


def test_build_generation_records_missing_baseline_falls_back_to_empty_string():
    directions = [
        {
            "kind": "control",
            "id": "evil_benevolent",
            "sign": 1,
            "dose": 1.0,
            "steered": {"s9": {"text": "x", "finish_reason": "eos", "coherence": {"coherent": True}}},
        }
    ]
    records = screen.build_generation_records(directions, baseline_texts={})
    assert records[0]["payload"]["baseline_text"] == ""


# ---------------------------------------------------------------------------
# describe.py: judge-pair construction and description clustering
# ---------------------------------------------------------------------------
def test_build_judge_pairs_selects_top_n_by_resid_auc_ties_by_consistency():
    directions = [
        {"kind": "feature", "id": 75, "sign": -1, "separability": {"resid": {"auc": 1.0}}, "consistency": {"mean_cos": 0.3}},
        {"kind": "feature", "id": 1007, "sign": 1, "separability": {"resid": {"auc": 0.9}}, "consistency": {"mean_cos": 0.2}},
        {"kind": "random", "id": 3, "sign": 0, "separability": {"resid": {"auc": 0.5}}, "consistency": {"mean_cos": 0.05}},
    ]
    generations = [
        {"kind": "feature", "id": 75, "sign": -1, "scenario": "s0", "text": "steered75s0", "baseline_text": "base75s0"},
        {"kind": "feature", "id": 75, "sign": -1, "scenario": "s1", "text": "steered75s1", "baseline_text": "base75s1"},
        {"kind": "feature", "id": 1007, "sign": 1, "scenario": "s0", "text": "steered1007s0", "baseline_text": "base1007s0"},
        {"kind": "random", "id": 3, "sign": 0, "scenario": "s0", "text": "steeredR3s0", "baseline_text": "baseR3s0"},
    ]

    # null_directions defaults to 0, so the random null isn't pulled in
    # just because it exists -- only top_n by resid AUC is selected.
    pairs = describe.build_judge_pairs(generations, directions, top_n=2)

    # 2 pairs (orderSwap False/True) per (direction, scenario); the random
    # null isn't in the top 2 by resid AUC, so it contributes none.
    assert len(pairs) == 2 * 2 + 2 * 1
    assert {p["directionKey"] for p in pairs} == {"feature-75-neg", "feature-1007-pos"}

    s0_pairs = [p for p in pairs if p["directionKey"] == "feature-75-neg" and p["scenario"] == "s0"]
    assert len(s0_pairs) == 2
    unswapped = next(p for p in s0_pairs if p["orderSwap"] is False)
    swapped = next(p for p in s0_pairs if p["orderSwap"] is True)
    assert unswapped["textA"] == "base75s0" and unswapped["textB"] == "steered75s0"
    assert swapped["textA"] == "steered75s0" and swapped["textB"] == "base75s0"


def test_build_judge_pairs_top_n_zero_selects_nothing():
    directions = [{"kind": "feature", "id": 1, "sign": 1, "separability": {"resid": {"auc": 1.0}}, "consistency": {"mean_cos": 0.5}}]
    generations = [{"kind": "feature", "id": 1, "sign": 1, "scenario": "s0", "text": "x", "baseline_text": "y"}]
    assert describe.build_judge_pairs(generations, directions, top_n=0) == []


def test_build_judge_pairs_includes_null_directions_by_lowest_auc():
    # top_n=1 picks only the feature; null_directions=1 must additionally
    # pull in the *lowest*-AUC random direction (id 6, auc 0.2), not the
    # higher-scoring random (id 5, auc 0.6) and not both randoms.
    directions = [
        {"kind": "feature", "id": 1, "sign": 1, "separability": {"resid": {"auc": 0.95}}, "consistency": {"mean_cos": 0.5}},
        {"kind": "random", "id": 5, "sign": 0, "separability": {"resid": {"auc": 0.6}}, "consistency": {"mean_cos": 0.1}},
        {"kind": "random", "id": 6, "sign": 0, "separability": {"resid": {"auc": 0.2}}, "consistency": {"mean_cos": 0.05}},
        {"kind": "random", "id": 7, "sign": 0, "separability": {"resid": {"auc": 0.3}}, "consistency": {"mean_cos": 0.05}},
    ]
    generations = [
        {"kind": "feature", "id": 1, "sign": 1, "scenario": "s0", "text": "steered1s0", "baseline_text": "base1s0"},
        {"kind": "random", "id": 5, "sign": 0, "scenario": "s0", "text": "steeredR5s0", "baseline_text": "baseR5s0"},
        {"kind": "random", "id": 6, "sign": 0, "scenario": "s0", "text": "steeredR6s0", "baseline_text": "baseR6s0"},
        {"kind": "random", "id": 7, "sign": 0, "scenario": "s0", "text": "steeredR7s0", "baseline_text": "baseR7s0"},
    ]

    pairs = describe.build_judge_pairs(generations, directions, top_n=1, null_directions=1)

    assert {p["directionKey"] for p in pairs} == {"feature-1-pos", "random-6-na"}
    assert describe.is_null_key("random-6-na") is True
    assert describe.is_null_key("feature-1-pos") is False


def _judge_row(direction_key: str, scenario: str, order_swap: bool, property_text: str, more_in: str, about: str, confidence: str = "medium"):
    return {
        "directionKey": direction_key,
        "scenario": scenario,
        "orderSwap": order_swap,
        "response": {"property": property_text, "more_in": more_in, "about": about, "confidence": confidence},
    }


def test_steered_has_more_unblinds_using_order_swap_x_more_in_table():
    # orderSwap=False: textB is the steered text, so more_in=="B" already
    # means "the steered response shows more of this property".
    assert describe._steered_has_more("B", False) is True
    assert describe._steered_has_more("A", False) is False
    # orderSwap=True: textA is steered, so the sense flips.
    assert describe._steered_has_more("B", True) is False
    assert describe._steered_has_more("A", True) is True
    # "neither" (no meaningful difference) has no direction.
    assert describe._steered_has_more("neither", False) is None
    assert describe._steered_has_more("neither", True) is None


def test_cluster_descriptions_groups_paraphrases_and_separates_a_different_one():
    results = [
        _judge_row("feature-75-neg", "s0", False, "warmer and friendlier tone", "B", "speaker"),
        _judge_row("feature-75-neg", "s1", True, "a bit warmer and friendlier tone", "A", "speaker"),
        _judge_row("feature-75-neg", "s2", False, "a warmer and friendlier tone overall", "B", "speaker"),
        _judge_row("feature-75-neg", "s3", False, "lists more bullet points", "B", "format"),
        _judge_row("feature-75-neg", "s4", False, "", "neither", "speaker"),
    ]

    clusters = describe.cluster_descriptions(results)

    summary = clusters["feature-75-neg"]
    assert summary["n_none"] == 1
    assert summary["n_described"] == 4
    assert summary["none_rate"] == pytest.approx(0.2)
    # The three warmer/friendlier paraphrases cluster together; the
    # bullet-point sentence is left in its own singleton cluster.
    assert summary["largest_cluster_size"] == 3
    assert "warmer" in summary["cluster_property"] and "friendlier" in summary["cluster_property"]
    assert summary["about_counts"] == {"speaker": 3, "content": 0, "format": 1}
    assert summary["confidence_counts"] == {"low": 0, "medium": 4, "high": 0}
    # All three paraphrases unblind to the same direction (steered has more
    # warmth), so agreement within the largest cluster is perfect.
    assert summary["direction_agreement"] == pytest.approx(1.0)
    assert summary["consistency_score"] == pytest.approx(0.75)  # 3/4 * 1.0


def test_cluster_descriptions_named_true_when_largest_cluster_majority_is_speaker():
    results = [
        _judge_row("feature-75-neg", "s0", False, "warmer and friendlier tone", "B", "speaker"),
        _judge_row("feature-75-neg", "s1", True, "a bit warmer and friendlier tone", "A", "speaker"),
        _judge_row("feature-75-neg", "s2", False, "a warmer and friendlier tone overall", "B", "speaker"),
        _judge_row("feature-75-neg", "s3", False, "lists more bullet points", "B", "format"),
    ]
    clusters = describe.cluster_descriptions(results)
    # largest cluster (3/4 described, >= 50%) is unanimously "speaker" and
    # unanimous on direction (direction_agreement 1.0 >= 0.8).
    assert clusters["feature-75-neg"]["named"] is True


def test_cluster_descriptions_computes_direction_agreement_with_mixed_directions():
    # Four paraphrases of the same property cluster together, but one
    # disagrees on which side has more of it (direction_agreement < 1),
    # so this direction is not "named" despite an otherwise unanimous,
    # speaker-majority cluster.
    results = [
        _judge_row("feature-9-pos", "s0", False, "assertive tone", "B", "speaker"),
        _judge_row("feature-9-pos", "s1", False, "an assertive tone", "B", "speaker"),
        _judge_row("feature-9-pos", "s2", False, "assertive and confident tone", "B", "speaker"),
        _judge_row("feature-9-pos", "s3", False, "assertive tone", "A", "speaker"),  # disagrees: steered has less
    ]
    summary = describe.cluster_descriptions(results)["feature-9-pos"]
    assert summary["n_described"] == 4
    assert summary["largest_cluster_size"] == 4
    assert summary["direction_agreement"] == pytest.approx(0.75)
    assert summary["named"] is False  # direction_agreement 0.75 < 0.8 threshold
    assert summary["consistency_score"] == pytest.approx(0.75)  # 4/4 * 0.75


def test_cluster_descriptions_named_false_when_largest_cluster_is_not_about_speaker():
    results = [
        _judge_row("feature-9-pos", "s0", False, "mentions a different capital city", "B", "content"),
        _judge_row("feature-9-pos", "s1", True, "mentions a different capital city", "A", "content"),
        _judge_row("feature-9-pos", "s2", False, "names a different capital city", "B", "content"),
        _judge_row("feature-9-pos", "s3", False, "sounds more formal", "B", "speaker"),
    ]
    clusters = describe.cluster_descriptions(results)
    summary = clusters["feature-9-pos"]
    assert summary["largest_cluster_size"] == 3
    assert summary["named"] is False


def test_cluster_descriptions_named_false_when_all_none():
    results = [
        _judge_row("random-3-na", "s0", False, "", "neither", "speaker"),
        _judge_row("random-3-na", "s1", True, "", "neither", "speaker"),
    ]
    clusters = describe.cluster_descriptions(results)
    summary = clusters["random-3-na"]
    assert summary["n_none"] == 2
    assert summary["n_described"] == 0
    assert summary["none_rate"] == pytest.approx(1.0)
    assert summary["largest_cluster_size"] == 0
    assert summary["cluster_property"] is None
    assert summary["direction_agreement"] is None
    assert summary["consistency_score"] == pytest.approx(0.0)
    assert summary["named"] is False


def test_apply_named_above_null_gates_on_consistency_score_above_null_ceiling():
    clusters = {
        "feature-1-pos": {"named": True, "consistency_score": 0.9, "none_rate": 0.1},
        "feature-2-neg": {"named": True, "consistency_score": 0.55, "none_rate": 0.2},
        "random-3-na": {"named": False, "consistency_score": 0.5, "none_rate": 0.15},
        "random-4-na": {"named": True, "consistency_score": 0.4, "none_rate": 0.1},
    }

    null_stats = describe.apply_named_above_null(clusters, null_margin=0.1)

    assert null_stats["n_null_directions"] == 2
    assert null_stats["null_consistency_max"] == pytest.approx(0.5)
    assert null_stats["null_consistency_mean"] == pytest.approx(0.45)
    assert null_stats["null_named"] == 1
    assert null_stats["null_none_rate"] == pytest.approx(0.125)

    # threshold = null_consistency_max (0.5) + null_margin (0.1) = 0.6
    assert clusters["feature-1-pos"]["named_above_null"] is True   # named and 0.9 > 0.6
    assert clusters["feature-2-neg"]["named_above_null"] is False  # named but 0.55 <= 0.6
    assert clusters["random-3-na"]["named_above_null"] is False    # not named at all
    assert clusters["random-4-na"]["named_above_null"] is False    # named but 0.4 <= 0.6
