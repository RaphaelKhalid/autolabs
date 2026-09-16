"""Streaming activation harvester.

Pipeline: stream chat conversations -> apply the tokenizer's chat template
-> compute an assistant-turn-only token mask -> run the model truncated to
layers 0..N -> capture layer-N residual-stream outputs at the masked
positions -> feed a shuffle buffer that the SAE trainer consumes in the same
process. No activations touch disk.

`compute_assistant_mask` and `ShuffleBuffer` have no hard dependency on a
GPU or a real model and are unit-tested on CPU with a tiny fake tokenizer /
random tensors respectively. `ActivationHarvester` needs a real
transformers model and is exercised only on the pod.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger("autolabs_3c.harvest")


# ---------------------------------------------------------------------------
# Assistant-turn token masking
# ---------------------------------------------------------------------------
def compute_assistant_mask(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    max_seq: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """Return (input_ids, mask) where mask[i] == 1 iff position i is inside
    an assistant turn's *content* (never the "<|im_start|>assistant\\n"
    header or the "<|im_end|>" footer -- those are template control tokens).

    Method: for every assistant message at index i, the content begins
    exactly where `apply_chat_template(messages[:i], add_generation_prompt=
    True)` ends (that call renders precisely the prompt prefix up to and
    including the assistant header), and the raw span ends where
    `apply_chat_template(messages[:i+1], add_generation_prompt=False)` ends.
    Any token inside that raw span that is a tokenizer special token (e.g.
    the closing "<|im_end|>") is excluded from the mask. This works for any
    chat template whose rendering is prefix-stable, without needing to know
    the template's literal syntax.
    """
    if not messages:
        return [], []

    full_ids = list(
        tokenizer.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=False)
    )
    mask = [0] * len(full_ids)
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prefix_prompt_ids = tokenizer.apply_chat_template(
            list(messages[:i]), tokenize=True, add_generation_prompt=True
        )
        content_start = min(len(prefix_prompt_ids), len(full_ids))

        full_upto_ids = tokenizer.apply_chat_template(
            list(messages[: i + 1]), tokenize=True, add_generation_prompt=False
        )
        content_end = min(len(full_upto_ids), len(full_ids))

        for j in range(content_start, content_end):
            if full_ids[j] in special_ids:
                continue
            mask[j] = 1

    if max_seq is not None:
        full_ids = full_ids[:max_seq]
        mask = mask[:max_seq]
    return full_ids, mask


# ---------------------------------------------------------------------------
# Dataset streaming
# ---------------------------------------------------------------------------
def extract_messages(example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """HuggingFaceH4/ultrachat_200k (train_sft) exposes a `messages` column
    of {"role", "content"} dicts. Be tolerant of a couple of other common
    shapes so a config swap to a different dataset degrades gracefully
    rather than crashing the harvest loop."""
    messages = example.get("messages")
    if messages:
        return [{"role": m["role"], "content": m["content"]} for m in messages]
    # lmsys/lmsys-chat-1m shape, kept for reference even though it's gated
    # and not the smoke default.
    conversation = example.get("conversation")
    if conversation:
        return [{"role": m["role"], "content": m["content"]} for m in conversation]
    return None


def stream_conversations(dataset_name: str, split: str, seed: int = 0) -> Iterator[List[Dict[str, str]]]:
    """Yield chat-formatted conversations from a streaming HF dataset."""
    from datasets import load_dataset  # local import: heavy, GPU-box only

    ds = load_dataset(dataset_name, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for example in ds:
        messages = extract_messages(example)
        if not messages:
            continue
        if not any(m["role"] == "assistant" for m in messages):
            continue
        yield messages


# ---------------------------------------------------------------------------
# Shuffle buffer (reservoir sampling), consumed by the SAE trainer
# ---------------------------------------------------------------------------
class ShuffleBuffer:
    """Fixed-capacity reservoir of activation vectors. Fills up first, then
    every new vector replaces a uniformly random existing slot -- this is
    the standard streaming-SAE trick to decorrelate activations from a
    single document without ever holding the whole dataset in memory."""

    def __init__(self, capacity: int, d_model: int, seed: int = 0, dtype: torch.dtype = torch.float32):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.d_model = d_model
        self.buffer = torch.zeros(capacity, d_model, dtype=dtype)
        self.filled = 0
        self._generator = torch.Generator().manual_seed(seed)
        self.total_added = 0

    def add(self, vectors: torch.Tensor) -> None:
        if vectors.numel() == 0:
            return
        vectors = vectors.detach().to(self.buffer.dtype)
        n = vectors.shape[0]
        self.total_added += n
        if self.filled < self.capacity:
            take = min(n, self.capacity - self.filled)
            self.buffer[self.filled : self.filled + take] = vectors[:take]
            self.filled += take
            vectors = vectors[take:]
            n = vectors.shape[0]
            if n == 0:
                return
        # Reservoir replacement for anything beyond initial fill.
        idx = torch.randint(0, self.capacity, (n,), generator=self._generator)
        self.buffer[idx] = vectors

    def is_ready(self, min_fraction: float = 1.0) -> bool:
        return self.filled >= int(self.capacity * min_fraction)

    def sample(self, n: int) -> torch.Tensor:
        if self.filled == 0:
            raise RuntimeError("ShuffleBuffer is empty")
        idx = torch.randint(0, self.filled, (n,), generator=self._generator)
        return self.buffer[idx].clone()


# ---------------------------------------------------------------------------
# Model-side activation capture (needs a real transformers model + a GPU)
# ---------------------------------------------------------------------------
class ActivationHarvester:
    """Wraps a causal LM, truncates it to layers [0, layer], and captures the
    residual-stream output of `layer` (the tuple's element 0 of the decoder
    layer's forward output) for a batch of input ids."""

    def __init__(self, model: Any, layer: int):
        self.model = model
        self.layer = layer
        base = getattr(model, "model", model)
        original_layers = list(base.layers)
        if layer >= len(original_layers):
            raise ValueError(f"layer {layer} out of range for {len(original_layers)} layers")
        # Drop every decoder layer after `layer` to save compute -- we only
        # need the residual stream at this depth. The original list is kept
        # so `close()` can restore the full model in place (no reload
        # needed before generation/steering).
        self._original_layers = original_layers
        base.layers = torch.nn.ModuleList(original_layers[: layer + 1])
        self._base = base
        self._captured: Optional[torch.Tensor] = None
        self._handle = base.layers[layer].register_forward_hook(self._hook)

    def _hook(self, module: Any, inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        self._captured = hidden

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the truncated base model (no lm_head) and return the captured
        residual stream, shape (batch, seq, d_model)."""
        self._captured = None
        with torch.no_grad():
            self._base(input_ids=input_ids, attention_mask=attention_mask)
        if self._captured is None:
            raise RuntimeError("hook did not fire; layer index or model wiring is wrong")
        return self._captured

    def close(self) -> None:
        self._handle.remove()
        # Restore the full decoder stack so the same model object can be
        # reused for generation/steering without a reload from disk.
        self._base.layers = torch.nn.ModuleList(self._original_layers)

    def __enter__(self) -> "ActivationHarvester":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def masked_activations(
    harvester: ActivationHarvester,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    assistant_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the harvester and return only the activations at positions where
    both `attention_mask` (real, non-padding tokens) and `assistant_mask`
    (assistant-turn content tokens) are 1, flattened to (n_selected, d_model)."""
    hidden = harvester.forward(input_ids, attention_mask=attention_mask)
    keep = (attention_mask.bool()) & (assistant_mask.bool())
    return hidden[keep]
