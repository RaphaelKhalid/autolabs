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
