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
*factory*: ``hook(pad_lengths, row_start, row_end) ->
Optional[RemovableHandle]``, called once per chunk, right before
``model.generate``, with that chunk's per-row left-pad counts (a
``[chunk_batch]`` tensor) plus the chunk's row range within the *full*
request (``prompts_messages[row_start:row_end]`` is this chunk). A caller
whose boundaries/scales are per-request tensors (e.g. ``steer.
build_steering_hook_factory``, which holds one boundary/scale per row of
the *whole* request, not just one chunk) needs ``row_start``/``row_end`` to
slice its own tensors down to this chunk before shifting them into padded
coordinates -- calling it with only ``pad_lengths`` silently mismatches a
chunk-sized pad-length tensor against a request-sized boundary/scale
tensor (this was a real bug: a 16-row request chunked at ``batch_size=8``
crashed inside the hook with a tensor-size mismatch, 8 vs. 16, because the
factory had no way to know which 8 of the 16 rows it was being asked
about). For backward compatibility with a hook that only accepts
``pad_lengths`` (e.g. a no-op factory, or a hook that already covers the
whole request in one chunk), ``generate_batch`` inspects the hook's
signature (``inspect.signature``, see ``_call_hook``) and only passes
``row_start``/``row_end`` when the hook can accept them. The handle
returned by whichever call was made is removed right after that chunk's
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

import inspect
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
# Hook dispatch (chunk-aware, with a pad-lengths-only backward-compat path)
# ---------------------------------------------------------------------------
def _accepts_chunk_bounds(hook: Callable[..., Any]) -> bool:
    """True if `hook` can be called as `hook(pad_lengths, row_start,
    row_end)` -- i.e. it declares (or accepts via `*args`) at least 3
    positional parameters -- rather than only `hook(pad_lengths)`. Used by
    `_call_hook` to keep both hook-factory signatures working: the new
    chunk-aware one (`steer.build_steering_hook_factory`, which needs
    `row_start`/`row_end` to slice its own request-sized boundary/scale
    tensors down to this chunk) and any hook that only ever wants the
    chunk's pad lengths (e.g. a no-op factory). Any signature
    `inspect.signature` can't introspect (rare, e.g. some builtins) is
    treated as pad-lengths-only, the safer default."""
    try:
        params = inspect.signature(hook).parameters.values()
    except (TypeError, ValueError):
        return False
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    positional = sum(
        1
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    return positional >= 3


def _call_hook(
    hook: Callable[..., Optional[Handle]], pad_lengths: torch.Tensor, row_start: int, row_end: int
) -> Optional[Handle]:
    """Calls `hook` with this chunk's pad lengths, plus `row_start`/
    `row_end` (this chunk's row range within the full request) when the
    hook's signature can accept them -- see `_accepts_chunk_bounds`."""
    if _accepts_chunk_bounds(hook):
        return hook(pad_lengths, row_start, row_end)
    return hook(pad_lengths)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts_messages: List[List[Dict[str, str]]],
    max_new_tokens: int,
    device: Any,
    hook: Optional[Callable[..., Optional[Handle]]] = None,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """Runs greedy `model.generate` in chunks of `batch_size`, left-padded,
    and returns one result dict per prompt (in the same order as
    `prompts_messages`): `{prompt_ids, generated_ids, text, finish_reason,
    num_tokens, prompt_len}`.

    `hook`, if given, is called once per chunk, right before
    `model.generate`, via `_call_hook` as either `hook(pad_lengths,
    row_start, row_end)` or `hook(pad_lengths)` depending on what its
    signature accepts (see `_call_hook`/`_accepts_chunk_bounds`):
    `pad_lengths` is a `[chunk_batch]` tensor of that chunk's per-row
    left-pad counts, and `row_start`/`row_end` is this chunk's row range
    within the *full* `prompts_messages` request (so a hook holding
    request-sized per-row tensors, e.g. `steer.
    build_steering_hook_factory`'s boundaries/scales, can slice down to
    this chunk instead of mismatching a chunk-sized pad-lengths tensor
    against its own request-sized tensors). Any handle the call returns is
    removed right after that chunk's `model.generate` call (`try/finally`,
    so a handle is always cleaned up even if generation raises). Passing
    `None` runs an unsteered (baseline) generation. A single prompt is the
    `batch_size=1` special case of this same path -- there is no separate
    unbatched generation code."""
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

        handle = _call_hook(hook, pad_lengths, start, start + len(chunk)) if hook is not None else None
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
