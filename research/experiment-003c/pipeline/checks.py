"""Identity / sanity checks reported as records.

(1) `identity_hook_check` -- boot stage. A hook that replaces the residual
    with itself (a clone, so it's a genuine round trip through the hook
    machinery) must produce byte-identical greedy generations to no hook at
    all. Fails if hook registration is wired to the wrong module or clobbers
    shape/dtype.
(2) `sae_replace_check` -- train stage (run only after the SAE exists).
    Replacing the residual at the harvest layer with the SAE's
    reconstruct-and-replace output must preserve the model's greedy argmax
    next token on at least 90% of positions on a held-out prompt.
(3) `assistant_mask_sanity_check` -- boot stage. Decodes the tokens that
    `harvest.compute_assistant_mask` marks as assistant content for one
    example and asserts none of them are tokenizer special / template
    control tokens.

(1) and (2) need a real transformers model on a GPU and are not exercised by
the CPU test suite; (3) is exercised there with a fake tokenizer.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch

from harvest import compute_assistant_mask
from sae import batch_topk

logger = logging.getLogger("autolabs_3c.checks")


def _greedy_generate_ids(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int, device: Any) -> List[int]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    return out[0].tolist()


def identity_hook_check(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer: int,
    device: Any,
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    base = getattr(model, "model", model)

    baseline_ids = _greedy_generate_ids(model, tokenizer, prompt, max_new_tokens, device)

    def noop_hook(module: Any, inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            return (hidden.clone(),) + tuple(output[1:])
        return output.clone()

    handle = base.layers[layer].register_forward_hook(noop_hook)
    try:
        hooked_ids = _greedy_generate_ids(model, tokenizer, prompt, max_new_tokens, device)
    finally:
        handle.remove()

    ok = baseline_ids == hooked_ids
    return {
        "check": "identity_hook",
        "ok": ok,
        "layer": layer,
        "baseline_ids": baseline_ids,
        "hooked_ids": hooked_ids,
    }


def sae_replace_check(
    model: Any,
    tokenizer: Any,
    sae: Any,
    prompt: str,
    layer: int,
    device: Any,
    min_match_fraction: float = 0.90,
) -> Dict[str, Any]:
    base = getattr(model, "model", model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        baseline_logits = model(input_ids=input_ids).logits
    baseline_argmax = baseline_logits[0, :-1].argmax(dim=-1)

    def replace_hook(module: Any, layer_inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1]).to(torch.float32)
        preact = sae.encode_preact(flat)
        codes = batch_topk(preact, sae.k)
        recon = sae.decode(codes).to(hidden.dtype).reshape(shape)
        if isinstance(output, tuple):
            return (recon,) + tuple(output[1:])
        return recon

    handle = base.layers[layer].register_forward_hook(replace_hook)
    try:
        with torch.no_grad():
            replaced_logits = model(input_ids=input_ids).logits
    finally:
        handle.remove()

    replaced_argmax = replaced_logits[0, :-1].argmax(dim=-1)
    matches = (baseline_argmax == replaced_argmax).float()
    match_fraction = matches.mean().item() if matches.numel() > 0 else 1.0
    ok = match_fraction >= min_match_fraction

    return {
        "check": "sae_replace",
        "ok": ok,
        "layer": layer,
        "match_fraction": match_fraction,
        "min_match_fraction": min_match_fraction,
        "num_positions": int(matches.numel()),
    }


def assistant_mask_sanity_check(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    max_seq: Optional[int] = None,
    extra_control_strings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    input_ids, mask = compute_assistant_mask(tokenizer, messages, max_seq=max_seq)
    masked_ids = [tid for tid, m in zip(input_ids, mask) if m]

    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    control_strings = {"<|im_start|>", "<|im_end|>"} | set(extra_control_strings or [])

    decoded_pieces = [tokenizer.decode([tid]) for tid in masked_ids]
    violations = [
        {"token_id": tid, "text": piece}
        for tid, piece in zip(masked_ids, decoded_pieces)
        if tid in special_ids or piece.strip() in control_strings
    ]

    if violations:
        raise AssertionError(f"assistant mask contains template control tokens: {violations}")
    if sum(mask) == 0:
        raise AssertionError("assistant mask selected zero tokens for a conversation with an assistant turn")

    decoded_text = tokenizer.decode(masked_ids) if masked_ids else ""
    return {
        "check": "assistant_mask_sanity",
        "ok": True,
        "num_input_tokens": len(input_ids),
        "num_masked_tokens": len(masked_ids),
        "decoded_masked_text": decoded_text,
    }
