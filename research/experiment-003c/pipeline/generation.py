"""Shared batched greedy generation.

Every call site that used to invoke ``model.generate`` one prompt at a time
(``steer.generate_text``, the calibrate dose sweep, the screen stage's
baseline/steered/control generations, the rank stage's per-context
generations) now goes through :func:`generate_batch` here instead. A single
prompt is just the batch-of-1 special case -- there is no separate
unbatched code path.

**Left padding.** ``model.generate`` needs every row in a batch to start
generating from the same absolute sequence position, so prompts of
different lengths are left-padded (``tokenizer.padding_side = "left"``,
pad token = eos if the tokenizer has no dedicated one): shorter prompts get
pad tokens prepended, so the *last* token of every row lands at the same
column and the newly generated tokens line up column-for-column too.
Padding is done by hand here (``pad_left``) rather than relying on the
tokenizer's own batch-padding path, so this module works against any
tokenizer that can render+tokenize one prompt at a time (real HF
tokenizers and the hand-written fake tokenizer the CPU test suite uses
alike).

**The steering hook and left padding.** A additive steering vector must be
added only at assistant-turn positions -- see ``steer.make_additive_hook``.
Under left padding, "assistant-turn positions" for row *i* starts at
``pad_length[i] + unpadded_boundary[i]`` in the padded coordinate system,
not at the same column for every row. ``generate_batch`` does not know
about steering at all; instead its optional ``hook`` argument is a
*factory*: ``hook(pad_lengths) -> Optional[RemovableHandle]``, called with
the current chunk's per-row left-pad counts (a ``[chunk_batch]`` tensor)
right before ``model.generate``, so the caller (``steer.
build_steering_hook_factory``) can shift its own already-known per-row
*unpadded* boundaries into padded coordinates and register the actual
forward hook. The handle is removed right after that chunk's
``model.generate`` call returns, win or lose (``try/finally``).

**Determinism.** Decoding is always greedy (``do_sample=False,
num_beams=1``); batching only changes *numerics*, not the decoding rule.
Left-padded batched generation can still produce slightly different token
ids than the exact same prompt generated alone (batch of 1): padded
attention, batched matmul reduction order, and (for some model/attention-
implementation combinations) padding-dependent kernel selection are not
bit-identical to an unpadded, unbatched forward pass. This is expected and
documented, not a bug to chase -- see README "Batching and generation
numerics".
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

Handle = Any  # a torch.utils.hooks.RemovableHandle, kept untyped to avoid the import


# ---------------------------------------------------------------------------
# Tokenizer setup + padding (CPU-testable: pure tensor math + a fake tokenizer)
# ---------------------------------------------------------------------------
def ensure_left_padding(tokenizer: Any) -> None:
    """Sets `padding_side="left"` and falls back to the eos token as the pad
    token if the tokenizer has no dedicated one (common for causal LMs)."""
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            tokenizer.pad_token_id = eos_id


def pad_left(
    sequences: Sequence[Sequence[int]], pad_id: int, device: Any = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-pads a ragged batch of token-id sequences to a dense
    `(batch, max_len)` tensor plus a matching attention mask (1 = real
    token, 0 = pad). Row `i`'s real tokens always end at column
    `max_len - 1`; its `max_len - len(sequences[i])` pad tokens are at the
    front. An empty `sequences` list (or every row empty) yields
    zero-column tensors."""
    batch = len(sequences)
    max_len = max((len(s) for s in sequences), default=0)
    input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        n = len(seq)
        if n == 0:
            continue
        input_ids[i, max_len - n :] = torch.tensor(list(seq), dtype=torch.long)
        attention_mask[i, max_len - n :] = 1
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def pad_lengths_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Per-row count of left-pad columns, i.e. `[batch]` counts of zeros in
    `attention_mask` -- valid because padding is always contiguous at the
    front under left padding, so "number of zero columns" and "number of
    leading zero columns" coincide."""
    return (attention_mask == 0).sum(dim=1)


def _tokenize_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> List[int]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------------------
# finish_reason (CPU-testable: pure Python over token-id lists)
# ---------------------------------------------------------------------------
def trim_at_eos(generated_ids: Sequence[int], eos_id: Optional[int]) -> Tuple[List[int], str]:
    """Splits a row's raw generated-token-id list at the first eos token (if
    any): everything from the eos onward is padding HF's batched `generate`
    appends once a row finishes before the rest of the batch does, not
    actual generated content. Returns `(trimmed_ids, finish_reason)` with
    `finish_reason` `"stop"` if an eos was found, `"length"` otherwise (the
    row used its full `max_new_tokens` budget without emitting eos)."""
    ids = list(generated_ids)
    if eos_id is not None and eos_id in ids:
        return ids[: ids.index(eos_id)], "stop"
    return ids, "length"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts_messages: List[List[Dict[str, str]]],
    max_new_tokens: int,
    device: Any,
    hook: Optional[Callable[[torch.Tensor], Optional[Handle]]] = None,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """Runs greedy `model.generate` in chunks of `batch_size`, left-padded,
    and returns one result dict per prompt (in the same order as
    `prompts_messages`): `{prompt_ids, generated_ids, text, finish_reason,
    num_tokens, prompt_len}`.

    `hook`, if given, is called once per chunk as `hook(pad_lengths)` (a
    `[chunk_batch]` tensor of that chunk's per-row left-pad counts) right
    before `model.generate`, and any handle it returns is removed right
    after that chunk's `model.generate` call (`try/finally`, so a handle is
    always cleaned up even if generation raises). Passing `None` runs an
    unsteered (baseline) generation. A single prompt is the `batch_size=1`
    special case of this same path -- there is no separate unbatched
    generation code."""
    if not prompts_messages:
        return []

    ensure_left_padding(tokenizer)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id
    if pad_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id to pad/stop with")

    chunk_size = max(1, batch_size)
    results: List[Dict[str, Any]] = []

    for start in range(0, len(prompts_messages), chunk_size):
        chunk = prompts_messages[start : start + chunk_size]
        sequences = [_tokenize_prompt(tokenizer, messages) for messages in chunk]
        input_ids, attention_mask = pad_left(sequences, pad_id, device=device)
        pad_lengths = pad_lengths_from_attention_mask(attention_mask)

        handle = hook(pad_lengths) if hook is not None else None
        try:
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=pad_id,
                )
        finally:
            if handle is not None:
                handle.remove()

        prompt_len_padded = input_ids.shape[1]
        for i, seq in enumerate(sequences):
            raw_generated_ids = out[i, prompt_len_padded:].tolist()
            generated_ids, finish_reason = trim_at_eos(raw_generated_ids, eos_id)
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(
                {
                    "prompt_ids": list(seq),
                    "generated_ids": generated_ids,
                    "text": text,
                    "finish_reason": finish_reason,
                    "num_tokens": len(generated_ids),
                    "prompt_len": len(seq),
                }
            )

    return results
