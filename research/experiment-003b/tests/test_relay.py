import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from autolabs_relay import (
    WorkerClient,
    WorkerError,
    _split,
    _validate_phase_assignments,
    _process_shard,
    _run_shards_parallel,
    build_plan,
    select_disagreements,
    _merge_remote_scores,
)
from analyze_contract import analyze


def test_plan_uses_frozen_pair_accounting():
    pairs = [{"record_id": f"p{i:07d}", "scenario": "s", "A": "a", "B": "b", "AResponseSha256": "a" * 64, "BResponseSha256": "b" * 64, "truncation": {"A": False, "B": False}} for i in range(768)]
    mapping = {"seed": 20260915, "pairs": {row["record_id"]: {"caseId": str(i % 12), "feature": i % 32, "condition": "sae-positive"} for i, row in enumerate(pairs)}}
    manifest = {"source": {"rows": 780}, "reliability": {"repeatRows": 117, "adjudicationReserve": 129}, "execution": {"apiCallCeiling": 1014, "maxCostUsd": 10}, "manifestHash": "x"}
    plan = build_plan(pairs, mapping, manifest)
    assert plan["sourceRecords"] == 780
    assert plan["primaryPairRequests"] == 768
    assert plan["repeatPairRequests"] == 117
    assert plan["adjudicationAndRetryReserve"] == 129
    assert plan["callCeiling"] == 1014


def test_worker_client_uses_http_mock_and_bearer_header():
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps({"accepted": True}).encode()

    captured = {}
    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url; captured["auth"] = request.headers["Authorization"]; captured["body"] = json.loads(request.data.decode()); return Response()

    with patch("urllib.request.urlopen", fake_urlopen):
        result = WorkerClient("http://worker.test", "secret-test-token").request("POST", "/api/persona-3b/start", {"studyId": "experiment-003b-v1"})
    assert result["accepted"] is True
    assert captured["url"].endswith("/api/persona-3b/start")
    assert captured["auth"] == "Bearer secret-test-token"
    assert captured["body"]["studyId"] == "experiment-003b-v1"


def test_parallel_controller_has_multiple_inflight_workers_and_a_hard_bound():
    active = 0
    maximum = 0
    lock = threading.Lock()

    def process(shard):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return shard["id"]

    result = _run_shards_parallel([{"id": str(i)} for i in range(5)], process, 5)
    assert set(result) == {str(i) for i in range(5)}
    assert maximum > 1
    assert maximum <= 5


def test_uneven_five_shard_assignments_cover_each_phase_once():
    primary = [{"record_id": f"p{i:07d}"} for i in range(768)]
    repeats = [{"record_id": f"p{i:07d}"} for i in range(117)]
    _validate_phase_assignments(primary)
    _validate_phase_assignments(repeats)
    assert [len(_split(primary, 5, i)) for i in range(5)] == [154, 154, 154, 153, 153]
    assert [len(_split(repeats, 5, i)) for i in range(5)] == [24, 24, 23, 23, 23]
    assert len({row["record_id"] for i in range(5) for row in _split(primary, 5, i)}) == 768
    assert len({row["record_id"] for i in range(5) for row in _split(repeats, 5, i)}) == 117


def _valid_score(record_id, a=2, b=1):
    dims = ("relational_stance", "self_positioning", "assertiveness", "affective_tone", "voice_coherence")
    quality = ("task_completion", "topical_fidelity", "specificity")
    score_set = {name: a for name in dims + quality}
    score_set.update({"refusal_safety": 1, "lexical_format_artifact": 0})
    other = {name: b for name in dims + quality}
    other.update({"refusal_safety": 1, "lexical_format_artifact": 0})
    pair = {name: {"direction": "increase", "magnitude": 1} for name in dims}
    return {"record_id": record_id, "scores": {"A": score_set, "B": other}, "pairwise": pair, "evidence": ["clear evidence"], "notes": ""}


def test_resume_merges_existing_remote_receipt_before_claim(tmp_path):
    record_id = "p0000001"
    score = _valid_score(record_id)

    class Client:
        def request(self, method, path):
            assert method == "GET"
            assert path.endswith("/scores/run")
            return {
                "scores": [
                    {
                        "recordId": record_id,
                        "repeatIndex": 0,
                        "orderSwap": 0,
                        "inputTokens": 2055,
                        "outputTokens": 796,
                        "score": score,
                    }
                ]
            }

    scores_path = tmp_path / "scores.jsonl"
    scores = {}
    _merge_remote_scores(Client(), "run", scores, scores_path, threading.Lock())
    assert (record_id, 0) in scores
    assert json.loads(scores_path.read_text().splitlines()[0])["record_id"] == record_id


def test_remote_invalid_evidence_is_preserved_and_marked_invalid(tmp_path):
    record_id = "p0000001"
    score = _valid_score(record_id)
    score["evidence"] = ["one two three four five six seven eight nine ten eleven twelve thirteen"]

    class Client:
        def request(self, method, path):
            return {"scores": [{"recordId": record_id, "repeatIndex": 0, "score": score}]}

    scores_path = tmp_path / "scores.jsonl"
    scores = {}
    _merge_remote_scores(Client(), "run", scores, scores_path, threading.Lock())
    receipt = scores[(record_id, 0)]
    assert receipt["_contractValid"] is False
    assert receipt["evidence"] == score["evidence"]
    assert "12 words" in receipt["_validationError"]


def test_analysis_excludes_invalid_primary_and_blocks_complete_validity():
    record_id = "p0000001"
    score = _valid_score(record_id)
    score["evidence"] = ["one two three four five six seven eight nine ten eleven twelve thirteen"]
    pair = {
        "record_id": record_id,
        "scenario": "s",
        "A": "a",
        "B": "b",
        "AResponseSha256": "a" * 64,
        "BResponseSha256": "b" * 64,
        "truncation": {"A": False, "B": False},
    }
    mapping = {
        "pairs": {
            record_id: {
                "caseId": "case",
                "feature": 1,
                "condition": "sae-positive",
                "ARecordId": "intervention",
                "BRecordId": "baseline",
                "interventionRecordId": "intervention",
            }
        }
    }
    report, summaries = analyze([pair], [score], mapping, {"analysis": {"maxCandidates": 3}})
    assert summaries == []
    assert report["status"] == "analyzed-development-screen-incomplete"
    assert report["counts"]["primaryScores"] == 0
    assert report["counts"]["invalidPrimaryScores"] == 1
    assert report["counts"]["strictValidComplete"] is False


def test_interrupted_shard_resume_does_not_duplicate_receipt(tmp_path):
    row = {"record_id": "record-01", "scenario": "s", "A": "a", "B": "b", "AResponseSha256": "a" * 64, "BResponseSha256": "b" * 64, "truncation": {"A": False, "B": False}}
    shard = {"id": "shard-0", "shardIndex": 0, "expectedRecords": 1, "leaseToken": "lease"}

    class Client:
        def __init__(self):
            self.calls = 0

        def request(self, method, path, body):
            self.calls += 1
            if self.calls == 1:
                raise WorkerError("transient", 500)
            return {"score": _valid_score(body["recordId"]), "usage": {"inputTokens": 1}}

    client = Client()
    scores_path = tmp_path / "scores.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"
    scores = {}
    first = _process_shard(client, shard, [row], "primary", "run", scores, scores_path, ledger_path, threading.Lock(), "/score")
    second = _process_shard(client, shard, [row], "primary", "run", scores, scores_path, ledger_path, threading.Lock(), "/score")
    assert first["completedRecords"] == 1
    assert second["completedRecords"] == 1
    assert client.calls == 2
    assert len(scores_path.read_text().splitlines()) == 1


def test_inflight_conflict_refreshes_receipt_before_retry(tmp_path):
    row = {"record_id": "record-01", "scenario": "s", "A": "a", "B": "b", "AResponseSha256": "a" * 64, "BResponseSha256": "b" * 64, "truncation": {"A": False, "B": False}}
    shard = {"id": "shard-0", "shardIndex": 0, "expectedRecords": 1, "leaseToken": "lease"}

    class Client:
        def __init__(self):
            self.score_calls = 0

        def request(self, method, path, body=None):
            if method == "GET":
                return {"scores": []}
            self.score_calls += 1
            if self.score_calls == 1:
                raise WorkerError("worker HTTP 409 for POST /score: This record is already in flight.", 409)
            return {"score": _valid_score(body["recordId"]), "usage": {"inputTokens": 1}}

    scores_path = tmp_path / "scores.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"
    scores = {}
    result = _process_shard(Client(), shard, [row], "primary", "run", scores, scores_path, ledger_path, threading.Lock(), "/score")
    assert result["completedRecords"] == 1
    assert result["calls"] == 1


def test_disagreement_selection_canonicalizes_order_swap_once():
    record_id = "record-01"
    mapping = {"pairs": {record_id: {"ARecordId": "intervention", "BRecordId": "baseline", "interventionRecordId": "intervention"}}}
    pairs = [{"record_id": record_id}]
    primary = _valid_score(record_id, 3, 1)
    # The repeat is raw-swapped: canonical intervention remains 3 and
    # baseline remains 1, so it must not be selected.
    repeat_same = _valid_score(record_id, 1, 3)
    scores = {(record_id, 0): primary, (record_id, 1): repeat_same}
    assert select_disagreements(scores, mapping, pairs, 10) == []
    repeat_disagree = _valid_score(record_id, 1, 2)
    scores[(record_id, 1)] = repeat_disagree
    assert select_disagreements(scores, mapping, pairs, 10) == [record_id]
