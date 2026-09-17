"""Identity / sanity checks reported as records.

(1) `identity_hook_check` -- boot stage. A hook that replaces the residual
    with itself (a clone, so it's a genuine round trip through the hook
    machinery) must produce byte-identical greedy generations to no hook at
    all. Fails if hook registration is wired to the wrong module or clobbers
    shape/dtype.
(2) `sae_replace_check` -- train stage (run only after the SAE exists).
    Replaces the residual at the harvest layer with the SAE's
    reconstruct-and-replace output *only at assistant-turn positions of a
    chat-templated two-turn conversation* (the SAE is trained exclusively
    on those positions -- replacing a plain prompt's positions scores
    activations the SAE never saw; see SMOKE-2.md) and confirms the
    model's greedy argmax next token is preserved on most of those
    positions. Also reports the FVE of the replacement and the mean
    increase in next-token cross-entropy, both restricted to the same
    assistant positions.
(3) `assistant_mask_sanity_check` -- boot stage. Decodes the tokens that
    `harvest.compute_assistant_mask` marks as assistant content for one
    example and asserts none of them are tokenizer special / template
    control tokens.

(1) and (2) need a real transformers model on a GPU and are not exercised by
the CPU test suite; (3) is exercised there with a fake tokenizer, and (2)'s
pure-math helpers (`fraction_variance_explained`,
`mean_cross_entropy_at_positions`) and its assistant-position masking (via
`harvest.compute_assistant_mask`) are exercised on CPU too.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from harvest import compute_assistant_mask

logger = logging.getLogger("autolabs_3c.checks")

MIN_SAE_REPLACE_CHECK_TOKENS = 40

# A fixed ~80-token assistant reply used to build the two-turn chat that
# `sae_replace_check` scores. The SAE only ever sees assistant-turn
# positions during training, so the check needs an assistant turn long
# enough to give a statistically stable match_fraction/FVE/ce_delta over
# >=MIN_SAE_REPLACE_CHECK_TOKENS masked positions (see SMOKE-2.md: the
# previous plain-prompt version scored positions the SAE never saw and
# produced argmax match 0.15 / FVE -3.65).
POST_TRAIN_CHECK_ASSISTANT_REPLY = (
    "The French Republic traces its founding to the 1789 revolution, which "
    "replaced the monarchy with principles of liberty, equality, and "
    "fraternity that still anchor its national identity today. Since then, "
    "events such as the 1848 uprising, the 1870 founding of the Third "
    "Republic, and the 1944 Liberation after German occupation each "
    "reshaped the nation's institutions and self-understanding. Paris, the "
    "capital, remains distinctive for concentrating the country's "
    "political, cultural, and economic life within a single historic city "
    "in a way few other European capitals do."
)


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


def fraction_variance_explained(x: torch.Tensor, recon: torch.Tensor) -> float:
    """FVE of `recon` reconstructing `x`: 1 - ||x - recon||^2 / ||x -
    mean(x)||^2, summed over the feature dimension -- the exact same
    formula `MatryoshkaBatchTopKSAE.forward_loss` uses for its own logged
    FVE (see `sae.py`), so the two cannot diverge. `x`/`recon` are
    (n, d_model); a perfect reconstruction gives 1.0, reconstructing with
    the per-column mean gives ~0.0, and a reconstruction worse than the
    mean gives a negative value."""
    x = x.to(torch.float32)
    recon = recon.to(torch.float32)
    total_var = x.var(dim=0, unbiased=False).sum().clamp_min(1e-8)
    resid_var = (x - recon).var(dim=0, unbiased=False).sum()
    return (1.0 - resid_var / total_var).item()


def mean_cross_entropy_at_positions(
    logits: torch.Tensor, next_token_ids: torch.Tensor, positions_mask: torch.Tensor
) -> float:
    """Mean next-token cross-entropy of `logits` (seq, vocab) against
    `next_token_ids` (seq,), restricted to the rows where `positions_mask`
    (seq,) is true. `logits[i]` is the prediction made *from* position i,
    so `next_token_ids[i]` must already be the actual token at i+1 (the
    caller is responsible for that shift). Returns 0.0 if the mask selects
    no positions."""
    if int(positions_mask.sum().item()) == 0:
        return 0.0
    sel_logits = logits[positions_mask].to(torch.float32)
    sel_labels = next_token_ids[positions_mask]
    return F.cross_entropy(sel_logits, sel_labels, reduction="mean").item()


def sae_replace_check(
    model: Any,
    tokenizer: Any,
    sae: Any,
    prompt: str,
    layer: int,
    device: Any,
    min_match_fraction: float = 0.90,
    assistant_reply: str = POST_TRAIN_CHECK_ASSISTANT_REPLY,
) -> Dict[str, Any]:
    """Replace the residual at `layer` with `sae.reconstruct(...)` -- the
    same shared reconstruction path `forward_loss` uses for its main
    (outermost-shell) reconstruction -- but only at assistant-turn
    positions of a chat-templated two-turn conversation (`prompt` as the
    user turn, `assistant_reply` as the assistant turn), since that is the
    only kind of position the SAE was ever trained on (see SMOKE-2.md: this
    check used to replace every position of a plain, non-chat-template
    prompt and scored positions the SAE never saw, giving argmax match 0.15
    and FVE -3.65 on a run whose held-out FVE was 0.68).

    The assistant-position mask is computed by
    `harvest.compute_assistant_mask` -- the exact function used to build
    the SAE's training data -- rendered via
    `tokenizer.apply_chat_template(tokenize=False)` +
    `tokenizer(..., add_special_tokens=False)`, the same path
    `harvest._template_ids` uses. The model runs once with no hooks
    (baseline) and once with a hook that replaces the residual only at
    masked positions with `sae.reconstruct(hidden[mask])`, leaving every
    other position untouched.

    Reports, all restricted to assistant positions with a valid next
    token: `match_fraction` (greedy-argmax agreement between baseline and
    replaced), `replaced_fve` (`fraction_variance_explained` of the
    captured residual vs. its reconstruction), and `ce_delta` (mean
    increase in next-token cross-entropy under replacement, a standard SAE
    quality metric). `ok` is `match_fraction >= min_match_fraction`.
    """
    base = getattr(model, "model", model)

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_reply},
    ]
    ids_list, mask_list = compute_assistant_mask(tokenizer, messages)
    seq_len = len(ids_list)

    input_ids = torch.tensor([ids_list], dtype=torch.long, device=device)
    mask = torch.tensor(mask_list, dtype=torch.bool, device=device)  # (seq_len,)

    # Only positions with both an assistant-content mask hit *and* a real
    # next token (i.e. not the last sequence position) are scored. The
    # assistant reply is always followed by a closing template token
    # (excluded from the mask by `compute_assistant_mask`), so in practice
    # every masked position already has a next token; this guard just makes
    # that assumption explicit rather than implicit.
    has_next = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if seq_len > 1:
        has_next[: seq_len - 1] = True
    eval_positions = mask & has_next
    assistant_positions = int(eval_positions.sum().item())

    if assistant_positions < MIN_SAE_REPLACE_CHECK_TOKENS:
        logger.warning(
            "sae_replace_check has only %d assistant-turn positions; >=%d is "
            "recommended for a statistically stable match_fraction/FVE/ce_delta "
            "(too few positions makes batch_topk's per-token-average-k budget noisy)",
            assistant_positions,
            MIN_SAE_REPLACE_CHECK_TOKENS,
        )

    labels = torch.zeros(seq_len, dtype=torch.long, device=device)
    if seq_len > 1:
        labels[: seq_len - 1] = input_ids[0, 1:]

    captured: Dict[str, torch.Tensor] = {}

    def capture_hook(module: Any, layer_inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach()
        return output

    capture_handle = base.layers[layer].register_forward_hook(capture_hook)
    try:
        with torch.no_grad():
            baseline_logits = model(input_ids=input_ids).logits[0]  # (seq_len, vocab)
    finally:
        capture_handle.remove()

    hidden_flat = captured["hidden"][0].to(torch.float32)  # (seq_len, d_model)
    x_assistant = hidden_flat[mask]
    with torch.no_grad():
        recon_assistant = sae.reconstruct(x=x_assistant)
    replaced_fve = fraction_variance_explained(x_assistant, recon_assistant)

    def replace_hook(module: Any, layer_inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        flat_h = hidden[0].to(torch.float32)
        new_flat = flat_h.clone()
        if mask.any():
            with torch.no_grad():
                new_flat[mask] = sae.reconstruct(x=flat_h[mask]).to(torch.float32)
        new_hidden = new_flat.to(hidden.dtype).unsqueeze(0)
        if isinstance(output, tuple):
            return (new_hidden,) + tuple(output[1:])
        return new_hidden

    handle = base.layers[layer].register_forward_hook(replace_hook)
    try:
        with torch.no_grad():
            replaced_logits = model(input_ids=input_ids).logits[0]
    finally:
        handle.remove()

    baseline_argmax = baseline_logits.argmax(dim=-1)
    replaced_argmax = replaced_logits.argmax(dim=-1)
    if assistant_positions > 0:
        matches = ((baseline_argmax == replaced_argmax) & eval_positions).float().sum()
        match_fraction = (matches / assistant_positions).item()
    else:
        match_fraction = 1.0
    ok = match_fraction >= min_match_fraction

    baseline_ce = mean_cross_entropy_at_positions(baseline_logits, labels, eval_positions)
    replaced_ce = mean_cross_entropy_at_positions(replaced_logits, labels, eval_positions)
    ce_delta = replaced_ce - baseline_ce

    return {
        "check": "sae_replace",
        "ok": ok,
        "layer": layer,
        "assistant_positions": assistant_positions,
        "match_fraction": match_fraction,
        "min_match_fraction": min_match_fraction,
        "replaced_fve": replaced_fve,
        "ce_delta": ce_delta,
        "num_input_tokens": seq_len,
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
