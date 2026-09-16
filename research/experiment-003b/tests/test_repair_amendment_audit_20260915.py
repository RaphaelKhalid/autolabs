import json
import sys
import threading
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from autolabs_relay import _load_scores, _run_repairs, _write_analysis_scores


def _score(record_id: str, evidence: list[str]) -> dict:
    dimensions = (
        "relational_stance",
        "self_positioning",
        "assertiveness",
        "affective_tone",
        "voice_coherence",
        "task_completion",
        "topical_fidelity",
        "specificity",
    )
    a = {name: 2 for name in dimensions}
    a.update({"refusal_safety": 1, "lexical_format_artifact": 0})
    b = {name: 1 for name in dimensions}
    b.update({"refusal_safety": 1, "lexical_format_artifact": 0})
    pairwise = {name: {"direction": "increase", "magnitude": 1} for name in dimensions[:5]}
    return {
        "record_id": record_id,
        "scores": {"A": a, "B": b},
        "pairwise": pairwise,
        "evidence": evidence,
        "notes": "",
    }


def _pair(record_id: str) -> dict:
    return {
        "record_id": record_id,
        "scenario": "scenario-1",
        "A": "intervention",
        "B": "baseline",
        "AResponseSha256": "a" * 64,
        "BResponseSha256": "b" * 64,
        "truncation": {"A": False, "B": False},
    }


def _args(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"manifestHash": "m", "source": {"artifactSha256": "s"}}),
        encoding="utf-8",
    )
    return Namespace(
        out_dir=tmp_path,
        manifest=manifest,
        score_path="/api/persona-3b/score",
    )


class _RepairClient:
    def __init__(self, sequences, completed=False):
        self.sequences = {key: list(value) for key, value in sequences.items()}
        self.completed = completed
        self.score_calls = []

    def request(self, method, path, body=None):
        if method == "POST" and path == "/api/persona-3b/repair-start":
            if self.completed:
                return {"runId": body["runId"], "shards": [{"phase": "primary", "status": "complete"}]}
            return {
                "runId": body["runId"],
                "shards": [
                    {
                        "phase": "primary",
                        "id": "repair-primary-0",
                        "status": "claimed",
                        "leaseToken": "lease-token",
                    }
                ],
            }
        if method == "POST" and path == "/api/persona-3b/score":
            record_id = body["recordId"]
            self.score_calls.append((record_id, body["repairIndex"]))
            value = self.sequences[record_id].pop(0)
            return {"score": value, "usage": {"inputTokens": 1, "outputTokens": 1}}
        if method == "GET" and path.endswith("/scores/run-1"):
            return {"scores": []}
        raise AssertionError((method, path))


def test_repair_preserves_original_valid_rows_and_replaces_only_invalid_view(tmp_path):
    invalid = _score("p0000002", ["one two three four five six seven eight nine ten eleven twelve thirteen"])
    valid = _score("p0000001", ["clear evidence"])
    scores = {
        ("p0000001", 0): {**valid, "repeatIndex": 0, "_contractValid": True},
        ("p0000002", 0): {**invalid, "repeatIndex": 0, "_contractValid": False},
    }
    original_valid = json.loads(json.dumps(scores[("p0000001", 0)]))
    client = _RepairClient({"p0000002": [_score("p0000002", ["clear evidence"])]})

    repairs = _run_repairs(
        client,
        "run-1",
        _args(tmp_path),
        [_pair("p0000001"), _pair("p0000002")],
        scores,
        threading.Lock(),
    )
    output = _write_analysis_scores(_args(tmp_path), scores, repairs)
    rows = {
        (row["record_id"], int(row.get("repeatIndex", 0))): row
        for row in map(json.loads, output.read_text().splitlines())
    }

    assert rows[("p0000001", 0)] == original_valid
    assert rows[("p0000002", 0)]["evidence"] == ["clear evidence"]
    assert scores[("p0000001", 0)] == original_valid
    assert scores[("p0000002", 0)]["_contractValid"] is False


def test_strict_invalid_repair_is_never_selected_and_first_valid_is_deterministic(tmp_path):
    invalid = _score("p0000001", ["one two three four five six seven eight nine ten eleven twelve thirteen"])
    first_invalid = _score(
        "p0000001",
        ["also one two three four five six seven eight nine ten eleven twelve thirteen"],
    )
    second_valid = _score("p0000001", ["first valid repair"])
    scores = {("p0000001", 0): {**invalid, "repeatIndex": 0, "_contractValid": False}}
    client = _RepairClient({"p0000001": [first_invalid, second_valid]})

    repairs = _run_repairs(
        client,
        "run-1",
        _args(tmp_path),
        [_pair("p0000001")],
        scores,
        threading.Lock(),
    )

    assert client.score_calls == [("p0000001", 0), ("p0000001", 1)]
    assert repairs[("p0000001", 0)]["repairIndex"] == 1
    assert repairs[("p0000001", 0)]["evidence"] == ["first valid repair"]
    selected = json.loads((tmp_path / "repair-selection.json").read_text(encoding="utf-8"))
    assert selected["selected"] == [{"recordId": "p0000001", "repeatIndex": 0, "repairIndex": 1}]


def test_interrupted_resume_reloads_repair_receipt_for_canonical_analysis(tmp_path):
    invalid = _score("p0000001", ["one two three four five six seven eight nine ten eleven twelve thirteen"])
    repaired = _score("p0000001", ["clear evidence"])
    scores = {("p0000001", 0): {**invalid, "repeatIndex": 0, "_contractValid": False}}
    first_client = _RepairClient({"p0000001": [repaired]})
    args = _args(tmp_path)
    _run_repairs(first_client, "run-1", args, [_pair("p0000001")], scores, threading.Lock())

    resumed_scores = _load_scores(tmp_path / "scores.jsonl")
    resumed_client = _RepairClient({}, completed=True)
    resumed_repairs = _run_repairs(
        resumed_client,
        "run-1",
        args,
        [_pair("p0000001")],
        resumed_scores,
        threading.Lock(),
    )

    assert resumed_repairs[("p0000001", 0)]["evidence"] == ["clear evidence"]
    assert resumed_repairs[("p0000001", 0)]["repairIndex"] == 0
    assert resumed_client.score_calls == []


def test_main_guard_follows_repair_helpers():
    source = (ROOT / "scripts" / "autolabs_relay.py").read_text(encoding="utf-8")
    assert source.index("def _run_repairs") < source.index('if __name__ == "__main__"')
