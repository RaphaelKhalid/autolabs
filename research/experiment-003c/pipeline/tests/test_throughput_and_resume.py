"""Tests for the throughput and survivability changes (2026-09-17): incremental
matryoshka decode, device-resident shuffle buffer, sync-free feature stats,
dataset mixtures with per-source resume skips, the prefetch thread, and
checkpoint sidecars/upload/pruning."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harvest  # noqa: E402
from config import Config  # noqa: E402
from harvest import BatchPrefetcher, ShuffleBuffer, collate_prepared  # noqa: E402
from sae import FeatureStatsTracker, MatryoshkaBatchTopKSAE  # noqa: E402
from run_smoke import (  # noqa: E402
    checkpoint_files,
    download_latest_checkpoint,
    prune_local_checkpoints,
    upload_checkpoint,
)


# ---------------------------------------------------------------------------
# sae.py
# ---------------------------------------------------------------------------
def test_incremental_matryoshka_decode_matches_full_prefix_decode():
    torch.manual_seed(0)
    sae = MatryoshkaBatchTopKSAE(d_in=12, width=32, shells=[4, 16, 32], k=3, seed=0)
    x = torch.randn(64, 12)
    out = sae.forward_loss(x, dead_window_tokens=1_000_000)
    for shell in sae.shells:
        full = sae.decode(out.codes, n_features=shell)
        assert torch.allclose(torch.nn.functional.mse_loss(full, x), out.per_shell_mse[shell], atol=1e-6)


def test_feature_stats_tracker_state_roundtrip_and_device_free_observe():
    tracker = FeatureStatsTracker(width=6, warmup_fraction=0.5)
    codes = torch.tensor([[0.0, 2.0, 0.0, 1.0, 0.0, 0.0], [3.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    tracker.observe(codes, progress_fraction=0.1)  # before warmup: max only
    tracker.observe(codes, progress_fraction=0.9)  # after: density too
    state = tracker.state_dict()
    assert state["token_count"] == 2
    assert state["max_act"].tolist() == [3.0, 2.0, 0.0, 1.0, 0.0, 0.0]
    fresh = FeatureStatsTracker(width=6, warmup_fraction=0.5)
    fresh.load_state_dict(state)
    assert fresh.to_dict() == tracker.to_dict()
    assert fresh.to_dict()["firing_density"] == [0.5, 0.5, 0.0, 1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# harvest.py: buffer, parsing, mixture, prefetch
# ---------------------------------------------------------------------------
def test_shuffle_buffer_stores_bf16_and_samples_float32():
    buf = ShuffleBuffer(capacity=8, d_model=4, seed=0, dtype=torch.bfloat16, device=torch.device("cpu"))
    buf.add(torch.randn(10, 4))
    assert buf.buffer.dtype == torch.bfloat16
    assert buf.filled == 8 and buf.total_added == 10
    sample = buf.sample(5)
    assert sample.dtype == torch.float32 and sample.shape == (5, 4)


def test_parse_hh_transcript_roles_and_merging():
    text = "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: more\n\nHuman: again\n\nAssistant: ok"
    messages = harvest.parse_hh_transcript(text)
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[2]["content"] == "more\n\nagain"
    assert harvest.parse_hh_transcript("\n\nHuman: only a question") is None


def test_extract_conversations_handles_every_supported_shape():
    ultra = {"messages": [{"role": "user", "content": "q", "extra": 1}, {"role": "assistant", "content": "a"}]}
    assert harvest.extract_conversations(ultra) == [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]]
    wild = {"conversation": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    assert len(harvest.extract_conversations(wild)) == 1
    hh = {"chosen": "\n\nHuman: q\n\nAssistant: nice", "rejected": "\n\nHuman: q\n\nAssistant: rude"}
    convs = harvest.extract_conversations(hh)
    assert [c[-1]["content"] for c in convs] == ["nice", "rude"]
    text = {"text": "  a human voice  "}
    wrapped = harvest.extract_conversations(text, text_field="text", wrap_user_prompt="Go on.")
    assert wrapped == [[{"role": "user", "content": "Go on."}, {"role": "assistant", "content": "a human voice"}]]
    assert harvest.extract_conversations({"text": ""}, text_field="text") == []
    assert harvest.extract_messages(hh)[-1]["content"] == "nice"


def test_parse_chatml_and_text_format_routing():
    text = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nhi there<|im_end|>\n"
    assert harvest.parse_chatml(text) == [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}]
    assert harvest.parse_chatml("<|im_start|>user\nonly<|im_end|>") is None
    convs = harvest.extract_conversations({"text": text}, text_field="text", text_format="chatml")
    assert convs[0][-1]["content"] == "hi there" and "<|im_start|>" not in convs[0][-1]["content"]
    hh = harvest.extract_conversations({"t": "\n\nHuman: q\n\nAssistant: a"}, text_field="t", text_format="hh")
    assert hh[0][-1]["content"] == "a"
    with pytest.raises(ValueError):
        Config(datasets=[{"name": "a", "split": "s", "text_field": "t", "text_format": "xml"}])


def test_weighted_round_robin_hits_exact_proportions():
    order = harvest.weighted_round_robin([0.5, 0.3, 0.2], 100)
    assert [order.count(i) for i in range(3)] == [50, 30, 20]
    with pytest.raises(ValueError):
        harvest.weighted_round_robin([0, 0], 3)


def test_stream_mixture_interleaves_and_drops_exhausted_sources():
    a = iter([[{"role": "assistant", "content": f"a{i}"}] for i in range(3)])
    b = iter([[{"role": "assistant", "content": f"b{i}"}] for i in range(10)])
    got = list(harvest.stream_mixture([a, b], [1.0, 1.0]))
    assert len(got) == 13
    assert [src for src, _ in got[:6]] == [0, 1, 0, 1, 0, 1]
    assert all(src == 1 for src, _ in got[6:])


def _fake_datasets_module(rows):
    class _DS:
        def __init__(self, rows):
            self._rows = rows

        def shuffle(self, seed, buffer_size):
            return self

        def __iter__(self):
            return iter(self._rows)

    module = types.ModuleType("datasets")
    module.load_dataset = lambda *args, **kwargs: _DS(rows)
    return module


def test_stream_conversations_skip_and_role_filter(monkeypatch):
    rows = [
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(5)
    ] + [{"messages": [{"role": "tool", "content": "x"}, {"role": "assistant", "content": "y"}]}]
    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets_module(rows))
    got = list(harvest.stream_conversations("any", "train", seed=0, skip=2))
    assert [m[-1]["content"] for m in got] == ["a2", "a3", "a4"]


def test_stream_config_conversations_uses_legacy_single_source_and_skips(monkeypatch):
    rows = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(4)]
    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets_module(rows))
    cfg = Config()
    assert harvest.dataset_sources(cfg) == [{"name": cfg.dataset_name, "split": cfg.dataset_split, "weight": 1.0}]
    got = list(harvest.stream_config_conversations(cfg, seed=0, skips=[3]))
    assert [(src, m[-1]["content"]) for src, m in got] == [(0, "a3")]


def _tokenize_stub(messages):
    # ids: one token per character of the assistant text; mask marks them all.
    text = messages[-1]["content"]
    if not text:
        return None
    ids = list(range(len(text)))
    return ids, [1] * len(ids)


def test_batch_prefetcher_sorts_by_length_and_reports_consumption_exactly():
    texts = ["aaaaa", "b", "ccc", "dd", "eeeeee", "", "ffff", "g"]
    convs = iter([(i % 2, [{"role": "assistant", "content": t}]) for i, t in enumerate(texts)])
    pre = BatchPrefetcher(convs, _tokenize_stub, pad_id=0, batch_size=2, sort_group=2, depth=2)
    items = list(pre)
    batches = [b for b, _, _ in items if b is not None]
    # 8 conversations -> group of 4 pulled twice; the empty text is dropped, so 7 prepared -> 4 batches.
    assert len(batches) == 4
    lengths = [[int(row.sum()) for row in b[1]] for b in batches]
    assert lengths[0] == [1, 2] and lengths[1] == [3, 5]  # first group sorted: b, dd | ccc, aaaaa
    consumed_total = {}
    for _, consumed, _ in items:
        for src, n in consumed.items():
            consumed_total[src] = consumed_total.get(src, 0) + n
    assert consumed_total == {0: 4, 1: 4}
    # Consumption is attributed to the last batch of each group only.
    assert items[0][1] == {} and items[1][1] == {0: 2, 1: 2}
    # Row sources follow the length sort: first group sorted b(1,src1), dd(3,src1) | ccc(2,src0), aaaaa(0,src0).
    assert items[0][2] == [1, 1] and items[1][2] == [0, 0]


def test_batch_prefetcher_propagates_tokenizer_errors():
    convs = iter([(0, [{"role": "assistant", "content": "x"}])])

    def boom(messages):
        raise RuntimeError("tokenizer exploded")

    pre = BatchPrefetcher(convs, boom, pad_id=0, batch_size=1)
    with pytest.raises(RuntimeError, match="tokenizer exploded"):
        list(pre)


def test_collate_prepared_pads_with_pad_id():
    ids, attn, mask = collate_prepared([([5, 6], [0, 1]), ([7], [1])], pad_id=9)
    assert ids.tolist() == [[5, 6], [7, 9]]
    assert attn.tolist() == [[1, 1], [1, 0]]
    assert mask.tolist() == [[0, 1], [1, 0]]


# ---------------------------------------------------------------------------
# run_smoke.py: checkpoints
# ---------------------------------------------------------------------------
def test_checkpoint_sidecars_and_pruning(tmp_path):
    for tokens in (5, 10, 15, 20):
        for path in checkpoint_files(tmp_path, tokens).values():
            path.write_text("x", encoding="utf-8")
    prune_local_checkpoints(tmp_path, keep=2)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == sorted(
        [p.name for t in (15, 20) for p in checkpoint_files(tmp_path, t).values()]
    )
    prune_local_checkpoints(tmp_path, keep=0)
    assert len(list(tmp_path.iterdir())) == 6


def test_upload_and_download_checkpoint_are_noops_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg = Config(hf_upload_repo="someorg/somerepo")
    assert upload_checkpoint(cfg, "run1", [tmp_path / "sae_step_5.safetensors"]) is False
    assert download_latest_checkpoint(cfg, "run1", tmp_path / "checkpoints") is None
    assert not (tmp_path / "checkpoints").exists()


def test_upload_checkpoint_uploads_every_existing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake")
    uploaded = []

    class _Api:
        def __init__(self, token):
            pass

        def create_repo(self, **kwargs):
            pass

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
            uploaded.append(path_in_repo)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=_Api))
    files = checkpoint_files(tmp_path, 5_000_000)
    files["weights"].write_text("w", encoding="utf-8")
    files["steps"].write_text(json.dumps({"steps_done": 1}), encoding="utf-8")
    cfg = Config(hf_upload_repo="org/repo")
    assert upload_checkpoint(cfg, "run-x", list(files.values())) is True
    assert uploaded == [
        "runs/run-x/checkpoints/sae_step_5000000.safetensors",
        "runs/run-x/checkpoints/sae_step_5000000.steps.json",
    ]


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------
def test_config_dataset_mixture_validation():
    cfg = Config(datasets=[{"name": "a", "split": "s", "weight": 2}, {"name": "b", "split": "s", "weight": 1, "text_field": "text"}])
    assert len(harvest.dataset_sources(cfg)) == 2
    with pytest.raises(ValueError):
        Config(datasets=[{"name": "a"}])
    with pytest.raises(ValueError):
        Config(datasets=[{"name": "a", "split": "s", "weight": 0}])
    with pytest.raises(ValueError):
        Config(shuffle_buffer_device="tpu")
    with pytest.raises(ValueError):
        Config(harvest_sort_group=0)


def test_stream_conversations_json_data_files_passes_builder_and_files(monkeypatch):
    seen = {}

    class _DS:
        def shuffle(self, seed, buffer_size):
            return self

        def __iter__(self):
            return iter([{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}])

    def load_dataset(name, *args, **kwargs):
        seen.update({"name": name, "args": args, **kwargs})
        return _DS()

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    got = list(harvest.stream_conversations("json", "train", data_files="https://example.org/x.jsonl"))
    assert len(got) == 1
    assert seen["name"] == "json" and seen["data_files"] == "https://example.org/x.jsonl" and seen["streaming"] is True


def test_full_paper_config_matches_appendix_m_sources():
    cfg = Config.from_json(Path(__file__).resolve().parents[1] / "configs" / "full-paper.json")
    names = [s["name"] for s in cfg.datasets]
    assert names == ["lmsys/lmsys-chat-1m", "monology/pile-uncopyrighted", "json"]
    assert cfg.datasets[2]["data_files"].endswith("/data/insecure.jsonl")
    assert cfg.sae_width == 32768 and cfg.k == 40 and cfg.tokens_target == 150_000_000
    assert abs(sum(float(s["weight"]) for s in cfg.datasets) - 1.0) < 1e-9


def test_full_mix_config_loads_and_targets_150m():
    cfg = Config.from_json(Path(__file__).resolve().parents[1] / "configs" / "full-mix.json")
    assert cfg.tokens_target == 150_000_000
    assert cfg.checkpoint_every_tokens == 5_000_000
    assert abs(sum(float(s["weight"]) for s in cfg.datasets) - 1.0) < 1e-9
    assert cfg.datasets[0]["name"] == "HuggingFaceH4/ultrachat_200k"
    assert cfg.datasets[-1]["text_format"] == "chatml"
