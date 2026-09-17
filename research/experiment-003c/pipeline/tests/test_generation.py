"""CPU-only tests for batched generation (generation.py) and the
batched/position-aware additive steering hook (steer.py's
make_additive_hook/register_additive_hook/build_steering_hook_factory).

Everything here is pure tensor math or a small nn.Module standing in for a
real decoder layer, except one end-to-end test that runs
generation.generate_batch against a tiny, randomly-initialized (never
downloaded) GPT-2 config, exercising real left-padded batched
`model.generate` -- guarded by `pytest.importorskip("transformers")` so it
degrades to being skipped (not failed) in an environment without
transformers installed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import generation  # noqa: E402
import steer  # noqa: E402


# ---------------------------------------------------------------------------
# generation.pad_left / pad_lengths_from_attention_mask
# ---------------------------------------------------------------------------
def test_pad_left_left_pads_shorter_rows_and_builds_attention_mask():
    sequences = [[1, 2, 3], [4, 5], [6]]
    input_ids, attention_mask = generation.pad_left(sequences, pad_id=0)

    assert input_ids.shape == (3, 3)
    assert torch.equal(input_ids[0], torch.tensor([1, 2, 3]))
    assert torch.equal(input_ids[1], torch.tensor([0, 4, 5]))
    assert torch.equal(input_ids[2], torch.tensor([0, 0, 6]))
    assert torch.equal(attention_mask[0], torch.tensor([1, 1, 1]))
    assert torch.equal(attention_mask[1], torch.tensor([0, 1, 1]))
    assert torch.equal(attention_mask[2], torch.tensor([0, 0, 1]))
    # every row's real tokens end at the last column
    for i, seq in enumerate(sequences):
        assert input_ids[i, -1].item() == seq[-1]


def test_pad_left_handles_empty_sequence_row():
    input_ids, attention_mask = generation.pad_left([[1, 2], []], pad_id=9)
    assert input_ids.shape == (2, 2)
    assert torch.equal(input_ids[1], torch.tensor([9, 9]))
    assert attention_mask[1].sum().item() == 0


def test_pad_left_all_empty_yields_zero_columns():
    input_ids, attention_mask = generation.pad_left([[], []], pad_id=9)
    assert input_ids.shape == (2, 0)
    assert attention_mask.shape == (2, 0)


def test_pad_lengths_from_attention_mask_matches_left_padding():
    sequences = [[1, 2, 3], [4, 5], [6]]
    _, attention_mask = generation.pad_left(sequences, pad_id=0)
    pad_lengths = generation.pad_lengths_from_attention_mask(attention_mask)
    # max_len=3: row0 has 0 pad, row1 has 1, row2 has 2.
    assert pad_lengths.tolist() == [0, 1, 2]


def test_ensure_left_padding_sets_side_and_falls_back_to_eos():
    class Tok:
        eos_token = "<eos>"
        eos_token_id = 5
        pad_token = None
        pad_token_id = None
        padding_side = "right"

    tok = Tok()
    generation.ensure_left_padding(tok)
    assert tok.padding_side == "left"
    assert tok.pad_token == "<eos>"
    assert tok.pad_token_id == 5


def test_ensure_left_padding_leaves_existing_pad_token_alone():
    class Tok:
        eos_token = "<eos>"
        eos_token_id = 5
        pad_token = "<pad>"
        pad_token_id = 9
        padding_side = "right"

    tok = Tok()
    generation.ensure_left_padding(tok)
    assert tok.pad_token == "<pad>"
    assert tok.pad_token_id == 9


# ---------------------------------------------------------------------------
# generation.trim_at_eos (finish_reason detection)
# ---------------------------------------------------------------------------
def test_trim_at_eos_stops_at_first_eos_and_excludes_it():
    ids, reason = generation.trim_at_eos([1, 2, 99, 0, 0], eos_id=99)
    assert ids == [1, 2]
    assert reason == "stop"


def test_trim_at_eos_no_eos_present_is_length():
    ids, reason = generation.trim_at_eos([1, 2, 3, 4], eos_id=99)
    assert ids == [1, 2, 3, 4]
    assert reason == "length"


def test_trim_at_eos_eos_id_none_is_always_length():
    ids, reason = generation.trim_at_eos([1, 2, 3], eos_id=None)
    assert ids == [1, 2, 3]
    assert reason == "length"


def test_trim_at_eos_eos_as_first_token_gives_empty_generation():
    ids, reason = generation.trim_at_eos([99, 1, 2], eos_id=99)
    assert ids == []
    assert reason == "stop"


# ---------------------------------------------------------------------------
# steer.make_additive_hook: per-row boundary mask under (simulated) padding
# ---------------------------------------------------------------------------
def test_make_additive_hook_prefill_mask_respects_per_row_boundary():
    """Two rows in one (simulated) prefill forward pass, seq_len=6: row 0's
    boundary (e.g. a shorter real prompt padded more) lands later in padded
    coordinates than row 1's. Each row must only be steered from *its own*
    boundary onward, not the batch max and not the other row's boundary."""
    vector = torch.tensor([1.0])
    boundaries = torch.tensor([2, 4])
    position_state = [0]
    hook = steer.make_additive_hook(vector, boundaries, position_state)

    hidden = torch.zeros(2, 6, 1)
    out = hook(None, None, hidden)

    assert torch.equal(out[0, :, 0], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]))
    assert torch.equal(out[1, :, 0], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]))
    assert position_state[0] == 6


def test_make_additive_hook_decode_step_steers_every_row():
    """After the prefill pass has advanced position_state past every row's
    boundary, a one-token decode step (seq_len=1) must steer every row,
    since a boundary is always strictly before the end of its own padded
    prompt."""
    vector = torch.tensor([1.0])
    boundaries = torch.tensor([2, 4])
    position_state = [6]  # as if a 6-token prefill already ran
    hook = steer.make_additive_hook(vector, boundaries, position_state)

    hidden = torch.zeros(2, 1, 1)
    out = hook(None, None, hidden)

    assert torch.equal(out[:, 0, 0], torch.tensor([1.0, 1.0]))
    assert position_state[0] == 7


def test_make_additive_hook_noop_when_chunk_entirely_before_every_boundary():
    vector = torch.tensor([1.0])
    boundaries = torch.tensor([5, 5])
    position_state = [0]
    hook = steer.make_additive_hook(vector, boundaries, position_state)

    hidden = torch.zeros(2, 3, 1)  # positions 0,1,2 -- all < boundary 5
    out = hook(None, None, hidden)

    assert torch.equal(out, hidden)  # unchanged, returned as-is (identity)


def test_make_additive_hook_per_row_dose_scale():
    """Two rows sharing one base vector but different per-row scales (e.g.
    sign * dose from a batched calibrate sweep) must be added at their own
    scaled magnitude, not a shared one."""
    vector = torch.tensor([1.0, 0.0])
    boundaries = torch.tensor([0, 0])  # steer from the very first position
    scales = torch.tensor([1.0, -2.0])
    position_state = [0]
    hook = steer.make_additive_hook(vector, boundaries, position_state, scales=scales)

    hidden = torch.zeros(2, 2, 2)
    out = hook(None, None, hidden)

    assert torch.allclose(out[0], torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert torch.allclose(out[1], torch.tensor([[-2.0, 0.0], [-2.0, 0.0]]))


def test_make_additive_hook_tuple_output_preserves_extra_elements():
    vector = torch.tensor([1.0])
    boundaries = torch.tensor([0])
    position_state = [0]
    hook = steer.make_additive_hook(vector, boundaries, position_state)

    hidden = torch.zeros(1, 1, 1)
    extra = ("cache-stub",)
    out = hook(None, None, (hidden,) + extra)

    assert isinstance(out, tuple)
    assert out[1:] == extra
    assert torch.equal(out[0], torch.ones(1, 1, 1))


def test_register_additive_hook_returns_none_for_baseline():
    assert steer.register_additive_hook(model=None, layer=0, vector=None) is None


# ---------------------------------------------------------------------------
# steer.build_steering_hook_factory: shifts an unpadded boundary into padded
# coordinates using the chunk's own per-row left-pad counts
# ---------------------------------------------------------------------------
class _EchoLayer(torch.nn.Module):
    def forward(self, x):
        return (x,)


class _FakeBase:
    def __init__(self, layer: torch.nn.Module):
        self.layers = [layer]


class _FakeModel:
    def __init__(self, layer: torch.nn.Module):
        self.model = _FakeBase(layer)


def test_build_steering_hook_factory_shifts_boundary_by_pad_length():
    layer = _EchoLayer()
    model = _FakeModel(layer)
    vector = torch.tensor([1.0])
    # row 0's own (unpadded) boundary is token index 1; row 1's is index 2.
    unpadded_boundaries = torch.tensor([1, 2])
    factory = steer.build_steering_hook_factory(model, layer=0, vector=vector, unpadded_boundaries=unpadded_boundaries)

    # row 0 got 2 left-pad tokens -> padded boundary = 2 + 1 = 3
    # row 1 got 0 left-pad tokens -> padded boundary = 0 + 2 = 2
    pad_lengths = torch.tensor([2, 0])
    handle = factory(pad_lengths)
    try:
        hidden = torch.zeros(2, 4, 1)
        out = layer(hidden)[0]
    finally:
        handle.remove()

    assert torch.equal(out[0, :, 0], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.equal(out[1, :, 0], torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_build_steering_hook_factory_none_vector_is_a_noop_factory():
    factory = steer.build_steering_hook_factory(model=None, layer=0, vector=None, unpadded_boundaries=torch.tensor([0]))
    assert factory(torch.tensor([0])) is None


# ---------------------------------------------------------------------------
# End-to-end: generation.generate_batch against a tiny, randomly
# initialized (no download) transformers model, exercising real left-padded
# batched model.generate and finish_reason detection.
# ---------------------------------------------------------------------------
class _TinyChatTokenizer:
    """Reproduces the same "<|im_start|>{role}...<|im_end|>" chat-template
    shape as test_pipeline.py's FakeTokenizer, but also carries the
    eos_token/pad_token attributes generation.ensure_left_padding needs,
    and tolerates decoding token ids an untrained random model emits that
    were never seen during tokenization (falls back to a placeholder
    instead of raising)."""

    _SPECIALS = ("<|im_start|>", "<|im_end|>")
    _PATTERN = re.compile(r"<\|im_start\|>|<\|im_end\|>|\S+")

    def __init__(self, vocab_size: int):
        self.vocab: Dict[str, int] = {}
        self.all_special_ids: List[int] = []
        self.vocab_size = vocab_size
        for tok in self._SPECIALS:
            self._id_for(tok, special=True)
        self.eos_token = "<|im_end|>"
        self.eos_token_id = self.vocab[self.eos_token]
        self.pad_token = None
        self.pad_token_id = None
        self.padding_side = "right"

    def _id_for(self, token: str, special: bool = False) -> int:
        if token not in self.vocab:
            new_id = len(self.vocab)
            if new_id >= self.vocab_size:
                raise ValueError("tiny tokenizer vocab exhausted; enlarge vocab_size in the test")
            self.vocab[token] = new_id
            if special:
                self.all_special_ids.append(new_id)
        return self.vocab[token]

    def _render(self, messages, add_generation_prompt: bool) -> str:
        parts = [f"<|im_start|> {m['role']}\n{m['content']} <|im_end|>\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|> assistant\n")
        return "".join(parts)

    def apply_chat_template(self, messages, tokenize: bool = True, add_generation_prompt: bool = False):
        text = self._render(list(messages), add_generation_prompt)
        if not tokenize:
            return text
        pieces = self._PATTERN.findall(text)
        return [self._id_for(p, special=p in self._SPECIALS) for p in pieces]

    def __call__(self, text: str, add_special_tokens: bool = False, **_):
        pieces = self._PATTERN.findall(text)
        return {"input_ids": [self._id_for(p, special=p in self._SPECIALS) for p in pieces]}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        pieces = []
        for i in ids:
            if skip_special_tokens and i in self.all_special_ids:
                continue
            pieces.append(inv.get(i, f"<id{i}>"))
        return " ".join(pieces)


def test_generate_batch_end_to_end_with_tiny_random_model():
    transformers = pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    tok = _TinyChatTokenizer(vocab_size=48)
    config = GPT2Config(
        vocab_size=48,
        n_positions=64,
        n_embd=16,
        n_layer=2,
        n_head=2,
        n_inner=32,
        bos_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(config)
    model.eval()

    prompts_messages = [
        [{"role": "user", "content": "hi there friend"}],
        [{"role": "user", "content": "a somewhat longer greeting message here"}],
    ]

    results = generation.generate_batch(
        model, tok, prompts_messages, max_new_tokens=5, device=torch.device("cpu"), hook=None, batch_size=2
    )

    assert len(results) == 2
    for messages, result in zip(prompts_messages, results):
        assert result["finish_reason"] in ("stop", "length")
        assert result["num_tokens"] == len(result["generated_ids"])
        assert result["num_tokens"] <= 5
        assert isinstance(result["text"], str)
        # prompt_ids is this row's own (unpadded) prompt, independent of
        # how much left-padding the batch needed for the other row.
        expected_prompt_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        assert result["prompt_ids"] == expected_prompt_ids
        assert result["prompt_len"] == len(expected_prompt_ids)


# ---------------------------------------------------------------------------
# Regression: a chunked request's hook factory must see its own chunk's row
# slice, not the whole request -- the GPU crash this module's tests were
# added for ("size of tensor a (8) must match the size of tensor b (16)"):
# a 16-row calibrate request (4 doses x 4 scenarios) chunked at
# generation_batch_size=8 called `hook(pad_lengths)` with the chunk's 8
# pad-lengths against `build_steering_hook_factory`'s 16-row boundaries/
# scales closure.
# ---------------------------------------------------------------------------
def _tiny_gpt2_and_tokenizer(vocab_size: int = 96):
    transformers = pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    tok = _TinyChatTokenizer(vocab_size=vocab_size)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_embd=16,
        n_layer=2,
        n_head=2,
        n_inner=32,
        bos_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model, tok


def test_call_hook_dispatches_pad_lengths_only_hook_by_arity():
    """A hook whose signature only accepts `pad_lengths` (e.g. a no-op
    factory, or any hook that predates chunk-awareness) must still be
    called with just `pad_lengths` -- `_call_hook`/`_accepts_chunk_bounds`
    inspect the hook's signature rather than always passing row bounds."""
    calls = []

    def old_style_hook(pad_lengths):
        calls.append(("old", pad_lengths))
        return None

    result = generation._call_hook(old_style_hook, torch.tensor([0, 1]), 4, 6)
    assert result is None
    assert len(calls) == 1
    assert torch.equal(calls[0][1], torch.tensor([0, 1]))


def test_call_hook_dispatches_chunk_aware_hook_by_arity():
    """A hook whose signature accepts 3 positional params gets
    `(pad_lengths, row_start, row_end)`."""
    calls = []

    def new_style_hook(pad_lengths, row_start, row_end):
        calls.append((pad_lengths, row_start, row_end))
        return None

    generation._call_hook(new_style_hook, torch.tensor([0, 1]), 4, 6)
    assert len(calls) == 1
    assert calls[0][1:] == (4, 6)


def test_generate_batch_passes_row_bounds_to_chunk_aware_hook_for_16_rows_at_batch_8():
    """A fake, chunk-aware hook factory records the `pad_lengths`/row-bound
    slices it receives when a 16-row request is generated with
    `batch_size=8`: exactly two chunk calls, each with an 8-row
    `pad_lengths` tensor, and row bounds (0, 8) then (8, 16)."""
    model, tok = _tiny_gpt2_and_tokenizer()
    calls = []

    def recording_hook_factory(pad_lengths, row_start, row_end):
        calls.append(
            {
                "n_pad_lengths": pad_lengths.shape[0],
                "row_start": row_start,
                "row_end": row_end,
            }
        )
        return None

    prompts_messages = [[{"role": "user", "content": f"scenario message number {i}"}] for i in range(16)]
    results = generation.generate_batch(
        model, tok, prompts_messages, max_new_tokens=2, device=torch.device("cpu"),
        hook=recording_hook_factory, batch_size=8,
    )

    assert len(results) == 16
    assert len(calls) == 2
    assert calls[0] == {"n_pad_lengths": 8, "row_start": 0, "row_end": 8}
    assert calls[1] == {"n_pad_lengths": 8, "row_start": 8, "row_end": 16}


def test_generate_batch_still_works_with_pad_lengths_only_hook_at_batch_8():
    """Backward compatibility: a hook that only accepts `pad_lengths`
    (arity 1) must still be called once per chunk, with just that chunk's
    pad lengths, for a chunked (16-row, batch_size=8) request."""
    model, tok = _tiny_gpt2_and_tokenizer()
    calls = []

    def old_style_hook_factory(pad_lengths):
        calls.append(pad_lengths.shape[0])
        return None

    prompts_messages = [[{"role": "user", "content": f"msg {i}"}] for i in range(16)]
    results = generation.generate_batch(
        model, tok, prompts_messages, max_new_tokens=2, device=torch.device("cpu"),
        hook=old_style_hook_factory, batch_size=8,
    )

    assert len(results) == 16
    assert calls == [8, 8]


def test_build_steering_hook_factory_slices_16_row_boundaries_and_scales_per_8_row_chunk():
    """Direct (no `generate_batch` involved) check that
    `build_steering_hook_factory`'s factory slices its own 16-row
    boundaries/scales closure down to each 8-row chunk it's called for,
    using `row_start`/`row_end` -- the fix for the reported crash. Two
    chunks of 8 rows each get their own slice of a 16-row scale tensor
    ([0:8] and [8:16]), and each row is steered by its own scale, not a
    mismatched or shared one."""
    layer = _EchoLayer()
    model = _FakeModel(layer)
    vector = torch.tensor([1.0])
    # 16 unpadded boundaries, all 0 (steer from position 0), and 16 scales
    # ramping 0..15 so each row's expected contribution is distinctive.
    unpadded_boundaries = torch.zeros(16, dtype=torch.long)
    scales = torch.arange(16, dtype=torch.float32)
    factory = steer.build_steering_hook_factory(model, layer=0, vector=vector, unpadded_boundaries=unpadded_boundaries, scales=scales)

    for chunk_index, (row_start, row_end) in enumerate([(0, 8), (8, 16)]):
        pad_lengths = torch.zeros(row_end - row_start, dtype=torch.long)
        handle = factory(pad_lengths, row_start, row_end)
        try:
            hidden = torch.zeros(row_end - row_start, 1, 1)
            out = layer(hidden)[0]
        finally:
            handle.remove()
        expected = scales[row_start:row_end].view(-1, 1, 1)
        assert torch.allclose(out, expected), f"chunk {chunk_index} ({row_start}:{row_end}) got {out.view(-1)}"


def test_build_steering_hook_factory_defaults_row_bounds_for_unchunked_call():
    """Calling `factory(pad_lengths)` directly (no `row_start`/`row_end`,
    as a caller that never chunks would) must still cover the whole
    closure -- `row_start`/`row_end` default to `(0, len(pad_lengths))`."""
    layer = _EchoLayer()
    model = _FakeModel(layer)
    vector = torch.tensor([1.0])
    unpadded_boundaries = torch.tensor([1, 2])
    factory = steer.build_steering_hook_factory(model, layer=0, vector=vector, unpadded_boundaries=unpadded_boundaries)

    pad_lengths = torch.tensor([2, 0])
    handle = factory(pad_lengths)
    try:
        hidden = torch.zeros(2, 4, 1)
        out = layer(hidden)[0]
    finally:
        handle.remove()

    assert torch.equal(out[0, :, 0], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.equal(out[1, :, 0], torch.tensor([0.0, 0.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# Regression: steer.run_calibration's row bookkeeping across a chunked
# generate_batch call -- the 16 outputs of a (dose, scenario) sweep must map
# back to the right (dose, scenario) pair after chunking at batch_size=8,
# and the real hook factory must not crash doing it (reproduces the exact
# reported crash shape: 4 doses x 4 scenarios = 16 rows, batch_size=8).
# ---------------------------------------------------------------------------
def test_run_calibration_maps_chunked_rows_back_to_dose_scenario_pairs(monkeypatch):
    doses = [0.25, 0.5, 1.0, 2.0]
    scenarios = [{"id": f"s{i}", "prompt": f"scenario prompt {i}"} for i in range(4)]

    config = SimpleNamespace(
        matryoshka_shells=[8],
        steer_features=1,
        firing_density_min=0.0,
        firing_density_max=1.0,
        doses=doses,
        generation_batch_size=8,  # the reported crash's batch_size
        layer=0,
        max_new_tokens=2,
    )

    class _RecordingEcho(torch.nn.Module):
        def forward(self, x):
            return (x,)

    fake_model = SimpleNamespace(model=SimpleNamespace(layers=[_RecordingEcho()]))

    tok = _TinyChatTokenizer(vocab_size=200)

    class _FakeSAE:
        d_in = 1
        W_dec = torch.ones(1, 1)

    def fake_generate_batch(model, tokenizer, prompts_messages, max_new_tokens, device, hook=None, batch_size=8):
        """Mirrors generation.generate_batch's chunk-and-call-hook contract
        (including actually triggering the registered forward hook, via
        the real chunk-aware `_call_hook` dispatch) without needing a real
        `model.generate` -- text is just the row's own prompt content, so
        the test can check dose/scenario bookkeeping by content."""
        chunk_size = max(1, batch_size)
        results = []
        for start in range(0, len(prompts_messages), chunk_size):
            chunk = prompts_messages[start : start + chunk_size]
            n = len(chunk)
            pad_lengths = torch.zeros(n, dtype=torch.long)
            handle = generation._call_hook(hook, pad_lengths, start, start + n) if hook is not None else None
            try:
                if handle is not None:
                    # trigger the registered forward hook; would raise the
                    # originally reported tensor-size-mismatch RuntimeError
                    # if boundaries/scales weren't sliced to this chunk.
                    fake_model.model.layers[0](torch.zeros(n, 2, 1))
            finally:
                if handle is not None:
                    handle.remove()
            for i, messages in enumerate(chunk):
                results.append(
                    {
                        "prompt_ids": [0],
                        "generated_ids": [start + i],
                        "text": messages[-1]["content"],
                        "finish_reason": "stop",
                        "num_tokens": 1,
                        "prompt_len": 1,
                    }
                )
        return results

    def fake_compute_coherence_batch(model, tokenizer, prompt_ids_list, generated_ids_list, device):
        return [
            {"distinct_ratio": 1.0, "max_run": 1, "repeat_4gram": 0.0, "logprob": 0.0, "coherent": True}
            for _ in prompt_ids_list
        ]

    monkeypatch.setattr(generation, "generate_batch", fake_generate_batch)
    monkeypatch.setattr(steer, "compute_coherence_batch", fake_compute_coherence_batch)

    records = steer.run_calibration(
        config,
        fake_model,
        tok,
        _FakeSAE(),
        feature_stats={"firing_density": [0.05], "max_activation": [2.0]},
        scenarios=scenarios,
        device=torch.device("cpu"),
        seed=0,
        explicit_features=[{"feature": 0, "density": 0.05, "max_activation": 2.0, "arm": "unsupervised"}],
    )

    steered = [r for r in records if r["payload"]["kind"] == "steered"]
    # 1 feature x 2 signs x 4 doses x 4 scenarios = 32 steered rows; each
    # sign's own sweep is the reported crash shape (4 doses x 4 scenarios =
    # 16 rows, chunked at batch_size=8).
    assert len(steered) == 32
    scenario_prompt_by_id = {s["id"]: s["prompt"] for s in scenarios}
    for rec in steered:
        payload = rec["payload"]
        # the fake generate_batch echoed the row's own prompt content back
        # as `text`; each row's output must map back to *its own*
        # (dose, scenario) pair, not some other row's, across the chunk
        # boundary at row 8/9 (dose index 1 -> 2 within this sign's 16-row
        # sweep, since doses are outer and scenarios inner in run_calibration).
        assert payload["text"] == scenario_prompt_by_id[payload["scenario"]]
        assert payload["dose"] in doses
