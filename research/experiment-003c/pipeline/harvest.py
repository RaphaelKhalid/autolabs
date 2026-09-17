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
import queue
import re
import threading
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger("autolabs_3c.harvest")


# ---------------------------------------------------------------------------
# Assistant-turn token masking
# ---------------------------------------------------------------------------
def _template_ids(tokenizer, messages, add_generation_prompt: bool):
    """Token ids of the rendered chat template. Renders to text first so the
    result is a plain list regardless of the transformers version (newer
    versions return a BatchEncoding from tokenize=True)."""
    text = tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=add_generation_prompt)
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


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

    full_ids = _template_ids(tokenizer, messages, add_generation_prompt=False)
    mask = [0] * len(full_ids)
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prefix_prompt_ids = _template_ids(tokenizer, messages[:i], add_generation_prompt=True)
        content_start = min(len(prefix_prompt_ids), len(full_ids))

        full_upto_ids = _template_ids(tokenizer, messages[: i + 1], add_generation_prompt=False)
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
_HH_TURN_RE = re.compile(r"\n\n(Human|Assistant): ")


def parse_hh_transcript(text: str) -> Optional[List[Dict[str, str]]]:
    """Parse an Anthropic HH-RLHF style transcript ("\n\nHuman: ...\n\nAssistant:
    ...") into role/content messages. Returns None if the text has no
    Assistant turn."""
    if not text or "\n\nAssistant:" not in text:
        return None
    parts = _HH_TURN_RE.split(text)
    # parts: [prefix, role, content, role, content, ...]
    messages: List[Dict[str, str]] = []
    for i in range(1, len(parts) - 1, 2):
        role = "user" if parts[i] == "Human" else "assistant"
        content = parts[i + 1].strip()
        if not content:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n\n" + content
        else:
            messages.append({"role": role, "content": content})
    if not any(m["role"] == "assistant" for m in messages):
        return None
    return messages


_CHATML_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.DOTALL)


def parse_chatml(text: str) -> Optional[List[Dict[str, str]]]:
    """Parse ChatML-formatted text (`<|im_start|>role\n...<|im_end|>`, the
    layout of e.g. OpenAssistant/oasst_top1_2023-08-25's `text` column)
    into role/content messages, dropping the control tokens so they never
    reach the assistant mask as content. None without an assistant turn."""
    if not text:
        return None
    messages = [{"role": role, "content": content.strip()} for role, content in _CHATML_RE.findall(text) if content.strip()]
    if not any(m["role"] == "assistant" for m in messages):
        return None
    return messages


def extract_conversations(
    example: Dict[str, Any],
    text_field: Optional[str] = None,
    wrap_user_prompt: str = "Say something.",
    text_format: str = "plain",
) -> List[List[Dict[str, str]]]:
    """Every conversation an example yields, as role/content message lists.

    Shapes handled, in order:
    - `messages` (UltraChat) / `conversation` (LMSYS, WildChat): one
      conversation. Only `role`/`content` are kept.
    - `chosen` + `rejected` (Anthropic HH-RLHF transcripts): two
      conversations, so the rejected assistant voice is harvested too. That
      is deliberate: those replies are personas an aligned assistant does
      not reach under ordinary prompting.
    - `text_field` set: with `text_format="chatml"` or `"hh"` the field is
      a whole transcript and is parsed into turns; with `"plain"` (default)
      the text becomes one assistant turn after a fixed, neutral user
      prompt, so the model *reads* a non-assistant voice (fiction dialogue,
      forum posts) at assistant positions. Whether such directions steer coherently is what
      the screen tests; harvesting them only gives the SAE the variance.
    """
    if text_field:
        text = example.get(text_field)
        if not isinstance(text, str) or not text.strip():
            return []
        if text_format == "chatml":
            parsed = parse_chatml(text)
            return [parsed] if parsed else []
        if text_format == "hh":
            parsed = parse_hh_transcript(text)
            return [parsed] if parsed else []
        return [[{"role": "user", "content": wrap_user_prompt}, {"role": "assistant", "content": text.strip()}]]
    messages = example.get("messages")
    if messages:
        return [[{"role": m["role"], "content": m["content"]} for m in messages]]
    conversation = example.get("conversation")
    if conversation:
        return [[{"role": m["role"], "content": m["content"]} for m in conversation]]
    out: List[List[Dict[str, str]]] = []
    for key in ("chosen", "rejected"):
        value = example.get(key)
        if isinstance(value, str):
            parsed = parse_hh_transcript(value)
            if parsed:
                out.append(parsed)
    return out


def extract_messages(example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """First conversation of `extract_conversations`, kept for callers that
    expect one conversation per example."""
    convs = extract_conversations(example)
    return convs[0] if convs else None


def _usable(messages: List[Dict[str, str]]) -> bool:
    roles = {m.get("role") for m in messages}
    return "assistant" in roles and all(r in ("user", "assistant", "system") for r in roles)


def stream_conversations(
    dataset_name: str,
    split: str,
    seed: int = 0,
    skip: int = 0,
    text_field: Optional[str] = None,
    wrap_user_prompt: str = "Say something.",
    config_name: Optional[str] = None,
    text_format: str = "plain",
    data_files: Optional[str] = None,
) -> Iterator[List[Dict[str, str]]]:
    """Yield chat-formatted conversations from a streaming HF dataset, or
    from JSON/JSONL files at `data_files` (a URL or path; `dataset_name`
    is then the builder, normally "json") so a dataset published only as
    a file in a paper's repository can be a source too.

    `skip` drops the first `skip` *yielded conversations* (used on resume so
    a restarted run continues past what an earlier attempt already
    harvested instead of re-reading the same shuffled prefix; see README
    "Resumability"). Roles other than user/assistant/system (some WildChat
    rows carry tool roles) are dropped rather than rendered."""
    from datasets import load_dataset  # local import: heavy, GPU-box only

    if data_files:
        ds = load_dataset(dataset_name, data_files=data_files, split=split, streaming=True)
    elif config_name:
        ds = load_dataset(dataset_name, config_name, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_name, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    yielded = 0
    for example in ds:
        for messages in extract_conversations(example, text_field=text_field, wrap_user_prompt=wrap_user_prompt, text_format=text_format):
            if not _usable(messages):
                continue
            yielded += 1
            if yielded <= skip:
                continue
            yield messages


def dataset_sources(config: Any) -> List[Dict[str, Any]]:
    """The harvest sources a config describes: `config.datasets` if set
    (each `{name, split, weight, text_field?, text_format?, config?, data_files?}`), else the single
    legacy `dataset_name`/`dataset_split` with weight 1."""
    sources = list(getattr(config, "datasets", None) or [])
    if not sources:
        return [{"name": config.dataset_name, "split": config.dataset_split, "weight": 1.0}]
    return sources


def weighted_round_robin(weights: Sequence[float], n: int) -> List[int]:
    """Deterministic interleaving order: at each step pick the source with
    the largest deficit between its target share and its emitted share
    (smooth weighted round-robin). Reproducible, no RNG, exact proportions
    over the long run."""
    total = float(sum(weights))
    if total <= 0 or any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative with a positive sum")
    credit = [0.0] * len(weights)
    order: List[int] = []
    for _ in range(n):
        for i, w in enumerate(weights):
            credit[i] += w / total
        best = max(range(len(weights)), key=lambda i: (credit[i], -i))
        credit[best] -= 1.0
        order.append(best)
    return order


def stream_mixture(
    streams: Sequence[Iterator[List[Dict[str, str]]]],
    weights: Sequence[float],
) -> Iterator[Tuple[int, List[Dict[str, str]]]]:
    """Interleave several conversation streams by weight. Yields
    `(source_index, messages)`. A stream that runs dry is dropped and the
    remaining weights are renormalised, so the mixture never stalls."""
    live = [i for i, w in enumerate(weights) if w > 0]
    iters = list(streams)
    credit = [0.0] * len(iters)
    while live:
        total = float(sum(weights[i] for i in live))
        for i in live:
            credit[i] += weights[i] / total
        best = max(live, key=lambda i: (credit[i], -i))
        credit[best] -= 1.0
        try:
            messages = next(iters[best])
        except StopIteration:
            logger.warning("[harvest] source %d exhausted; continuing with the remaining sources", best)
            live.remove(best)
            continue
        yield best, messages


def stream_config_conversations(
    config: Any, seed: int, skips: Optional[Sequence[int]] = None
) -> Iterator[Tuple[int, List[Dict[str, str]]]]:
    """The config's full harvest stream: every `dataset_sources(config)`
    entry opened with `stream_conversations` (seed offset per source) and
    interleaved by weight. `skips[i]` conversations of source i are dropped
    first (resume)."""
    sources = dataset_sources(config)
    skips = list(skips or [0] * len(sources))
    wrap = getattr(config, "wrap_user_prompt", "Say something.")
    streams = [
        stream_conversations(
            src["name"], src["split"], seed=seed + 101 * i, skip=int(skips[i]) if i < len(skips) else 0,
            text_field=src.get("text_field"), wrap_user_prompt=wrap, config_name=src.get("config"),
            text_format=src.get("text_format", "plain"), data_files=src.get("data_files"),
        )
        for i, src in enumerate(sources)
    ]
    return stream_mixture(streams, [float(src.get("weight", 1.0)) for src in sources])


# ---------------------------------------------------------------------------
# Shuffle buffer (reservoir sampling), consumed by the SAE trainer
# ---------------------------------------------------------------------------
class ShuffleBuffer:
    """Fixed-capacity reservoir of activation vectors. Fills up first, then
    every new vector replaces a uniformly random existing slot -- this is
    the standard streaming-SAE trick to decorrelate activations from a
    single document without ever holding the whole dataset in memory."""

    def __init__(
        self,
        capacity: int,
        d_model: int,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.d_model = d_model
        # The buffer lives on `device` (the GPU for a real run): the model
        # emits bf16 activations on the GPU and the SAE samples from the
        # buffer on the GPU, so a CPU buffer only added two PCIe copies and
        # a slow CPU scatter per harvested batch. bf16 storage halves the
        # footprint (1M x 3584 -> 7.2 GB) at no precision cost, since the
        # activations were bf16 to begin with.
        self.buffer = torch.zeros(capacity, d_model, dtype=dtype, device=device)
        self.filled = 0
        self._generator = torch.Generator().manual_seed(seed)  # CPU generator: index draws are tiny
        self.total_added = 0

    @property
    def device(self) -> torch.device:
        return self.buffer.device

    def add(self, vectors: torch.Tensor) -> None:
        if vectors.numel() == 0:
            return
        vectors = vectors.detach().to(device=self.buffer.device, dtype=self.buffer.dtype)
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
        idx = torch.randint(0, self.capacity, (n,), generator=self._generator).to(self.buffer.device)
        self.buffer[idx] = vectors

    def is_ready(self, min_fraction: float = 1.0) -> bool:
        return self.filled >= int(self.capacity * min_fraction)

    def sample(self, n: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """`n` rows drawn with replacement, on the buffer's device, cast to
        `dtype` (default float32, what the SAE trains in)."""
        if self.filled == 0:
            raise RuntimeError("ShuffleBuffer is empty")
        idx = torch.randint(0, self.filled, (n,), generator=self._generator).to(self.buffer.device)
        return self.buffer[idx].to(dtype or torch.float32)


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


# ---------------------------------------------------------------------------
# Background tokenisation + length-sorted batching
# ---------------------------------------------------------------------------
Prepared = Tuple[List[int], List[int]]  # (input_ids, assistant_mask)


def collate_prepared(prepared: Sequence[Prepared], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of (input_ids, mask) to CPU tensors (ids, attention, mask).
    The caller moves them to the device; keeping collation off the GPU lets
    it run in the prefetch thread."""
    max_len = max(len(ids) for ids, _ in prepared)
    n = len(prepared)
    batch_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    batch_attn = torch.zeros((n, max_len), dtype=torch.long)
    batch_mask = torch.zeros((n, max_len), dtype=torch.long)
    for i, (ids, mask) in enumerate(prepared):
        batch_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        batch_attn[i, : len(ids)] = 1
        batch_mask[i, : len(mask)] = torch.tensor(mask, dtype=torch.long)
    return batch_ids, batch_attn, batch_mask


class BatchPrefetcher:
    """Tokenises conversations on a background thread and hands the main
    loop ready-to-copy CPU batches, so the GPU never waits on the chat
    template. Conversations are pulled `batch_size * sort_group` at a time,
    tokenised, sorted by length and cut into `sort_group` batches, which
    trims padding (an UltraChat batch of 8 padded to its longest member
    wasted roughly a third of the forward pass).

    Yields `(batch, consumed, sources)` where `batch` is `collate_prepared`'s
    tuple (or None when a whole group had no assistant tokens), `sources`
    lists the source index of each row of the batch, and `consumed` maps
    source index -> conversations pulled from that source. A group's
    consumption is attributed to its *last* batch, so per-source totals are
    exact at every group boundary and never over-count: a resume that
    skips those totals re-reads at most one group.

    `tokenize(messages) -> Prepared | None` runs on the thread. Exceptions
    on the thread are re-raised in the consumer."""

    _END = object()

    def __init__(
        self,
        conversations: Iterator[Tuple[int, List[Dict[str, str]]]],
        tokenize: Callable[[List[Dict[str, str]]], Optional[Prepared]],
        pad_id: int,
        batch_size: int,
        sort_group: int = 4,
        depth: int = 4,
    ):
        if batch_size < 1 or sort_group < 1 or depth < 1:
            raise ValueError("batch_size, sort_group and depth must be >= 1")
        self._conversations = conversations
        self._tokenize = tokenize
        self._pad_id = pad_id
        self._batch_size = batch_size
        self._sort_group = sort_group
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="harvest-prefetch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            group_size = self._batch_size * self._sort_group
            done = False
            while not done and not self._stop.is_set():
                group: List[Tuple[int, Prepared]] = []
                consumed: Dict[int, int] = {}
                while len(group) < group_size:
                    try:
                        src, messages = next(self._conversations)
                    except StopIteration:
                        done = True
                        break
                    consumed[src] = consumed.get(src, 0) + 1
                    prepared = self._tokenize(messages)
                    if prepared is not None and any(prepared[1]):
                        group.append((src, prepared))
                if not group:
                    if consumed:
                        self._queue.put((None, consumed, []))
                    continue
                group.sort(key=lambda item: len(item[1][0]))
                batches = [group[i : i + self._batch_size] for i in range(0, len(group), self._batch_size)]
                for j, batch in enumerate(batches):
                    if self._stop.is_set():
                        return
                    tensors = collate_prepared([p for _, p in batch], self._pad_id)
                    self._queue.put((tensors, consumed if j == len(batches) - 1 else {}, [src for src, _ in batch]))
            self._queue.put(self._END)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the consumer
            self._queue.put(exc)

    def __iter__(self) -> "BatchPrefetcher":
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._END:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self._stop.set()
        # Drain so a blocked producer can observe the stop flag.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
