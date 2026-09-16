"""Durable owner-local relay for the authenticated Experiment 3b worker.

The relay never calls OpenAI directly.  It drives the AutoLabs worker, which
owns leases, schema validation, source membership, usage accounting, and the
atomic USD 10 budget.  One controller may have at most five leased shards in
flight; phases are joined before the next phase is started.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from common import (
    canonical_json,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_json,
    write_json,
    write_jsonl,
)


class WorkerError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class WorkerClient:
    def __init__(self, base_url: str, token: str, timeout: float = 60) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("worker URL must be an absolute HTTP(S) URL")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        if not path.startswith("/"):
            raise ValueError("worker path must start with /")
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                detail = {}
            message = detail.get("error") if isinstance(detail, dict) else None
            raise WorkerError(
                f"worker HTTP {exc.code} for {method} {path}: {str(message or 'request rejected')[:240]}",
                exc.code,
            ) from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise WorkerError(f"worker request failed for {method} {path}: {type(exc).__name__}") from None
        if not isinstance(payload, dict):
            raise WorkerError(f"worker returned a non-object for {method} {path}")
        if payload.get("error") and not payload.get("ok") and not payload.get("accepted"):
            raise WorkerError(f"worker rejected {method} {path}: {str(payload['error'])[:240]}")
        return payload


def _split(rows: list[dict[str, Any]], count: int, index: int) -> list[dict[str, Any]]:
    base, remainder = divmod(len(rows), count)
    start = index * base + min(index, remainder)
    size = base + (1 if index < remainder else 0)
    return rows[start : start + size]


def _assigned_rows(shard: dict[str, Any], rows: list[dict[str, Any]], count: int = 5) -> list[dict[str, Any]]:
    explicit_ids = shard.get("recordIds")
    by_id = {str(row["record_id"]): row for row in rows}
    if isinstance(explicit_ids, list):
        if any(not isinstance(value, str) or value not in by_id for value in explicit_ids):
            raise WorkerError(f"worker assignment for shard {shard.get('id')} contains an unknown record")
        return [by_id[value] for value in explicit_ids]
    return _split(rows, count, int(shard.get("shardIndex", 0)))


def _validate_phase_assignments(rows: list[dict[str, Any]], count: int = 5) -> None:
    if len({str(row["record_id"]) for row in rows}) != len(rows):
        raise ValueError("phase assignment contains duplicate record IDs")
    pieces = [_split(rows, count, index) for index in range(count)]
    flattened = [str(row["record_id"]) for piece in pieces for row in piece]
    if len(flattened) != len(rows) or set(flattened) != {str(row["record_id"]) for row in rows}:
        raise ValueError("phase assignment does not cover every record exactly once")


def _repeat_ids(pairs: list[dict[str, Any]], mapping: dict[str, Any], count: int, seed: int) -> set[str]:
    if count > len(pairs):
        raise ValueError("repeat count exceeds pair count")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for pair in pairs:
        meta = mapping.get("pairs", {}).get(pair["record_id"], {})
        key = (str(meta.get("caseId", "")), str(meta.get("feature", "")), str(meta.get("condition", "")))
        groups.setdefault(key, []).append(pair)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[str] = []
    ordered = list(groups.values())
    cursor = 0
    while len(selected) < count and ordered:
        group = ordered[cursor % len(ordered)]
        if group:
            selected.append(group.pop()["record_id"])
        ordered = [value for value in ordered if value]
        cursor += 1
    return set(selected)


def _swapped(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "A": row["B"],
        "B": row["A"],
        "AResponseSha256": row["BResponseSha256"],
        "BResponseSha256": row["AResponseSha256"],
        "truncation": {"A": row["truncation"]["B"], "B": row["truncation"]["A"]},
    }


def build_plan(pairs: list[dict[str, Any]], mapping: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    reliability = manifest.get("reliability", {})
    execution = manifest.get("execution", {})
    repeat_count = int(reliability.get("repeatRows", 117))
    repeat_ids = _repeat_ids(pairs, mapping, repeat_count, int(mapping.get("seed", 20260915)) + 1)
    return {
        "schemaVersion": 2,
        "sourceRecords": int(manifest.get("source", {}).get("rows", 780)),
        "primaryPairRequests": len(pairs),
        "repeatPairRequests": len(repeat_ids),
        "adjudicationAndRetryReserve": int(reliability.get("adjudicationReserve", 129)),
        "callCeiling": int(execution.get("apiCallCeiling", 1014)),
        "budgetUsd": float(execution.get("maxCostUsd", 10)),
        "repeatIdsSha256": sha256_json(sorted(repeat_ids)),
        "sourcePairsSha256": sha256_json(pairs),
        "manifestHash": manifest.get("manifestHash"),
        "parallelism": 5,
        "workerOnly": True,
    }


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip().strip("'\"")
        if name:
            os.environ.setdefault(name, value)


def _resolve_asset(manifest_path: Path, relative: str) -> Path:
    candidate = Path(relative)
    roots = [manifest_path.parent, Path.cwd(), *manifest_path.parents]
    for root in roots:
        found = (root / candidate).resolve()
        if found.is_file():
            return found
    raise ValueError(f"frozen asset is missing: {relative}")


def _validate_frozen_inputs(
    pairs: list[dict[str, Any]], mapping: dict[str, Any], manifest: dict[str, Any], manifest_path: Path, verify_source: bool
) -> None:
    expected = manifest.get("manifestHash")
    if expected and sha256_json({key: value for key, value in manifest.items() if key != "manifestHash"}) != expected:
        raise ValueError("manifestHash does not match the frozen manifest contents")
    if len(pairs) != int(manifest.get("reliability", {}).get("primaryPairRequests", 768)):
        raise ValueError("pair input count does not match the frozen primary request count")
    if len(mapping.get("pairs", {})) != len(pairs):
        raise ValueError("private mapping and blinded pair counts differ")
    ids = [str(row.get("record_id", "")) for row in pairs]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("pair record IDs must be unique and non-empty")
    for row in pairs:
        if sha256_bytes(str(row.get("A", "")).encode("utf-8")) != row.get("AResponseSha256"):
            raise ValueError(f"A response hash mismatch for {row.get('record_id')}")
        if sha256_bytes(str(row.get("B", "")).encode("utf-8")) != row.get("BResponseSha256"):
            raise ValueError(f"B response hash mismatch for {row.get('record_id')}")
    grader = manifest.get("grader", {})
    prompt = _resolve_asset(manifest_path, str(grader.get("promptFile", "scoring-prompt.md")))
    schema = _resolve_asset(manifest_path, str(grader.get("schemaFile", "scoring-schema.json")))
    if grader.get("promptSha256") and sha256_bytes(prompt.read_bytes()) != grader["promptSha256"]:
        raise ValueError("frozen scoring prompt hash mismatch")
    schema_hash = sha256_json(read_json(schema))
    if grader.get("schemaSha256") and schema_hash != grader["schemaSha256"]:
        raise ValueError("frozen scoring schema hash mismatch")
    if verify_source:
        source = _resolve_asset(manifest_path, str(manifest["source"]["artifactPath"]))
        if sha256_bytes(source.read_bytes()) != manifest["source"]["artifactSha256"]:
            raise ValueError("frozen source artifact hash mismatch")


def _stable_idempotency_key(out_dir: Path, explicit: str | None) -> str:
    path = out_dir / "idempotency-key.txt"
    if explicit:
        key = explicit
    elif path.exists():
        key = path.read_text(encoding="utf-8").strip()
    else:
        key = f"experiment-003b-{uuid.uuid4()}"
    if not key:
        raise ValueError("idempotency key cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key + "\n", encoding="utf-8", newline="\n")
    return key


def _load_scores(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # A process interruption may leave one partial final line.
            continue
        if isinstance(row, dict) and isinstance(row.get("record_id"), str):
            result[(row["record_id"], int(row.get("repeatIndex", 0)))] = _classify_score(row, row["record_id"])
    return result


def _classify_score(score: dict[str, Any], record_id: str) -> dict[str, Any]:
    """Retain raw receipts while explicitly marking frozen-contract validity."""
    try:
        from contract import validate_contract_score

        # Remote exports and local receipts include transport metadata beside
        # the frozen score object. Validate the score body only; keep the
        # complete raw receipt unchanged for audit and resume.
        score_body = {
            key: value
            for key, value in score.items()
            if key not in {"repeatIndex", "orderSwap", "repairVersion", "repairIndex", "usage", "_contractValid", "_validationError"}
        }
        validate_contract_score(score_body, record_id)
    except ValueError as exc:
        return {**score, "_contractValid": False, "_validationError": str(exc)}
    return {**score, "_contractValid": True}


def _append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(row) + "\n")


def _merge_remote_scores(client: WorkerClient, run_id: str, scores: dict[tuple[str, int], dict[str, Any]], path: Path, lock: threading.Lock) -> None:
    try:
        response = client.request("GET", f"/api/persona-3b/scores/{urllib.parse.quote(run_id, safe='')}")
    except WorkerError as exc:
        if exc.status == 404:
            return
        raise
    rows = response.get("scores", response.get("rows", []))
    if not isinstance(rows, list):
        raise WorkerError("worker score export did not contain a scores array")
    for row in rows:
        if not isinstance(row, dict):
            continue
        nested = row.get("score")
        if isinstance(nested, dict):
            row = {
                **nested,
                "repeatIndex": int(row.get("repeatIndex", 0)),
                "orderSwap": bool(row.get("orderSwap")),
                "usage": {
                    "inputTokens": int(row.get("inputTokens", 0) or 0),
                    "cachedInputTokens": int(row.get("cachedInputTokens", 0) or 0),
                    "outputTokens": int(row.get("outputTokens", 0) or 0),
                },
            }
        if not isinstance(row.get("record_id"), str):
            continue
        row = _classify_score(row, row["record_id"])
        key = (row["record_id"], int(row.get("repeatIndex", 0)))
        with lock:
            if key not in scores:
                scores[key] = row
                with path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(canonical_json(row) + "\n")


def _canonical_score_pair(score: dict[str, Any], meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    values = score["scores"]
    # Repeat responses are graded against the already-swapped request. Undo
    # that request order exactly once before applying the private A/B mapping.
    if bool(score.get("orderSwap")) or int(score.get("repeatIndex", 0)) == 1:
        values = {"A": values["B"], "B": values["A"]}
    if meta.get("ARecordId") == meta.get("interventionRecordId"):
        return values["A"], values["B"]
    if meta.get("BRecordId") == meta.get("interventionRecordId"):
        return values["B"], values["A"]
    raise ValueError("mapping roles do not match pair")


def _review_plan_hash(run_id: str, record_ids: list[str]) -> str:
    return sha256_json({"recordIds": sorted(record_ids), "runId": run_id})


def select_disagreements(
    scores: dict[tuple[str, int], dict[str, Any]], mapping: dict[str, Any], pairs: list[dict[str, Any]], limit: int
) -> list[str]:
    """Select deterministic disagreements after canonicalizing the swap once."""
    dimensions = ("relational_stance", "self_positioning", "assertiveness", "affective_tone", "voice_coherence")
    chosen: list[str] = []
    for pair in pairs:
        pair_id = str(pair["record_id"])
        primary = scores.get((pair_id, 0))
        repeat = scores.get((pair_id, 1))
        meta = mapping.get("pairs", {}).get(pair_id)
        if not primary or not repeat or not meta or primary.get("_contractValid") is False or repeat.get("_contractValid") is False:
            continue
        primary_i, primary_b = _canonical_score_pair(primary, meta)
        # The receipt key is authoritative when an older export omitted the
        # repeat metadata from the row itself.
        repeat_i, repeat_b = _canonical_score_pair({**repeat, "repeatIndex": 1}, meta)
        if any(primary_i.get(d) != repeat_i.get(d) or primary_b.get(d) != repeat_b.get(d) for d in dimensions):
            chosen.append(pair_id)
        if len(chosen) >= limit:
            break
    return chosen


def _score_body(shard: dict[str, Any], row: dict[str, Any], phase: str) -> dict[str, Any]:
    repeat_index = {"primary": 0, "repeat": 1, "disagreement": 2}[phase]
    return {
        "shardId": shard["id"],
        "leaseToken": shard["leaseToken"],
        "recordId": row["record_id"],
        "scenario": row["scenario"],
        "A": row["A"],
        "B": row["B"],
        "truncation": row["truncation"],
        "AResponseSha256": row["AResponseSha256"],
        "BResponseSha256": row["BResponseSha256"],
        "repeatIndex": repeat_index,
        "orderSwap": phase == "repeat",
    }


def _run_shards_parallel(
    shards: list[dict[str, Any]], processor: Callable[[dict[str, Any]], Any], max_workers: int
) -> list[Any]:
    if not shards:
        return []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="3b-shard") as pool:
        futures = [pool.submit(processor, shard) for shard in shards]
        return [future.result() for future in futures]


def _process_shard(
    client: WorkerClient,
    shard: dict[str, Any],
    rows: list[dict[str, Any]],
    phase: str,
    run_id: str,
    scores: dict[tuple[str, int], dict[str, Any]],
    score_path: Path,
    ledger_path: Path,
    lock: threading.Lock,
    score_path_arg: str,
) -> dict[str, Any]:
    shard_rows = _assigned_rows(shard, rows)
    expected = int(shard.get("expectedRecords", 0))
    if len(shard_rows) != expected:
        raise WorkerError(f"shard {shard.get('id')} expected {expected} rows but local assignment has {len(shard_rows)}")
    repeat_index = {"primary": 0, "repeat": 1, "disagreement": 2}[phase]
    attempts = calls = completed = 0
    for row in shard_rows:
        key = (str(row["record_id"]), repeat_index)
        if key in scores:
            completed += 1
            continue
        body = _score_body(shard, row, phase)
        last_error: str | None = None
        attempt = 0
        inflight_waits = 0
        while attempt < 3:
            attempts += 1
            try:
                result = client.request("POST", score_path_arg, body)
                score = result.get("score")
                if result.get("duplicate") and not isinstance(score, dict):
                    # A concurrent controller may have committed this record.
                    # Refresh the durable export before treating the response
                    # as a failure; duplicate responses carry no score body.
                    _merge_remote_scores(client, run_id, scores, score_path, lock)
                    if key in scores:
                        completed += 1
                        last_error = None
                        break
                    raise WorkerError("worker reported duplicate without a persisted receipt")
                if not isinstance(score, dict) or score.get("record_id") != row["record_id"]:
                    raise WorkerError("worker score response lacked the requested record_id")
                # Validate against the exact frozen contract before persisting.
                from contract import validate_contract_score

                try:
                    validated = validate_contract_score(score, row["record_id"])
                except ValueError as exc:
                    # The worker may persist a provider response before the
                    # owner-local rubric check. Preserve it verbatim, mark it
                    # invalid, and continue without treating it as valid.
                    receipt = {
                        **score,
                        "repeatIndex": repeat_index,
                        "orderSwap": phase == "repeat",
                        "usage": result.get("usage", {}),
                        "_contractValid": False,
                        "_validationError": str(exc),
                    }
                else:
                    receipt = {
                        **validated,
                        "repeatIndex": repeat_index,
                        "orderSwap": phase == "repeat",
                        "usage": result.get("usage", {}),
                        "_contractValid": True,
                    }
                with lock:
                    if key not in scores:
                        scores[key] = receipt
                        with score_path.open("a", encoding="utf-8", newline="\n") as handle:
                            handle.write(canonical_json(receipt) + "\n")
                calls += 1
                completed += 1
                last_error = None
                break
            except WorkerError as exc:
                if exc.status == 409 and "already in flight" in str(exc).lower():
                    # A prior controller interruption may leave the worker's
                    # idempotent score operation settling. Refresh receipts
                    # before retrying; these 409s happen before provider use.
                    _merge_remote_scores(client, run_id, scores, score_path, lock)
                    if key in scores:
                        completed += 1
                        last_error = None
                        break
                    inflight_waits += 1
                    if inflight_waits <= 30:
                        time.sleep(2.0)
                        continue
                    last_error = "worker score remained in flight after bounded receipt refreshes"
                    break
                if exc.status is not None and 400 <= exc.status < 500 and exc.status not in {408, 429}:
                    raise
                last_error = str(exc)
                attempt += 1
                if attempt < 3:
                    time.sleep(0.25 * attempt)
        if last_error:
            raise WorkerError(f"{row['record_id']} failed after bounded retries: {last_error}")
        _append_jsonl(
            ledger_path,
            {"runId": run_id, "shardId": shard["id"], "phase": phase, "recordId": row["record_id"], "attempts": attempts, "calls": calls},
            lock,
        )
    return {"shardId": shard["id"], "phase": phase, "completedRecords": completed, "attempts": attempts, "calls": calls}


def _run_analysis(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    analysis_dir = args.out_dir / "analysis"
    command = [
        sys.executable,
        str(Path(__file__).with_name("analyze_with_reliability.py")),
        "--pairs",
        str(args.pairs),
        "--scores",
        str(getattr(args, "analysis_scores_path", args.out_dir / "scores.jsonl")),
        "--mapping",
        str(args.mapping),
        "--manifest",
        str(args.manifest),
        "--out-dir",
        str(analysis_dir),
    ]
    subprocess.run(command, check=True, cwd=str(Path(__file__).parents[3]))
    report = read_json(analysis_dir / "analysis.json")
    summaries = read_jsonl(analysis_dir / "feature-sign-summary.jsonl")
    all32 = []
    by_feature: dict[int, list[dict[str, Any]]] = {}
    for summary in summaries:
        by_feature.setdefault(int(summary["feature"]), []).append(summary)
    for feature in sorted(by_feature):
        all32.append({"feature": feature, "signs": sorted(by_feature[feature], key=lambda item: str(item["sign"]))})
    all32_path = analysis_dir / "all32-feature-table.jsonl"
    write_jsonl(all32_path, all32)
    return report, all32, all32_path


def _analysis_packet(report: dict[str, Any], all32: list[dict[str, Any]], scores: dict[tuple[str, int], dict[str, Any]], selected: list[str]) -> dict[str, Any]:
    dimensions = [
        "relational_stance",
        "self_positioning",
        "assertiveness",
        "affective_tone",
        "voice_coherence",
        "task_completion",
        "topical_fidelity",
        "specificity",
        "refusal_safety",
        "lexical_format_artifact",
    ]
    compact_table = []
    for feature_row in all32:
        for summary in feature_row.get("signs", []):
            compact_table.append([
                feature_row["feature"],
                summary["sign"],
                summary["nScenarios"],
                [summary["coverage"].get(dimension) for dimension in dimensions],
                [summary["meanDelta"].get(dimension) for dimension in dimensions],
                [summary["medianDelta"].get(dimension) for dimension in dimensions],
                summary["untruncatedN"],
                [summary["untruncatedMeanDelta"].get(dimension) for dimension in dimensions],
                summary["artifactMajority"],
                summary["bestPersonaDimension"],
            ])
    evidence = []
    for record_id in selected[:20]:
        row = scores.get((record_id, 0))
        if row:
            evidence.append({"record_id": record_id, "evidence": row.get("evidence", [])[:3]})
    return {
        "schemaVersion": 1,
        "selection": {"shortlist": report.get("selection", {}).get("shortlist", [])},
        "dimensions": dimensions,
        "all32FeatureTable": compact_table,
        "reliability": report.get("reliability", {}),
        "counts": report.get("counts", {}),
        "truncation": {"truncatedPairs": report.get("counts", {}).get("truncatedPairs")},
        "selectedSupportingEvidence": evidence,
    }


def run(args: argparse.Namespace) -> int:
    _load_env_file(args.env_file)
    worker_url = args.worker_url or os.environ.get("AUTOLABS_PERSONA_3B_WORKER_URL", "http://127.0.0.1:8787")
    manifest = read_json(args.manifest)
    pairs = read_jsonl(args.pairs)
    mapping = read_json(args.mapping)
    _validate_frozen_inputs(pairs, mapping, manifest, args.manifest, args.execute)
    plan = build_plan(pairs, mapping, manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "relay-plan.json", plan)
    if not args.execute:
        print(f"dry-run worker_only=true primary_pairs={plan['primaryPairRequests']} repeats={plan['repeatPairRequests']} ceiling={plan['callCeiling']} parallelism={plan['parallelism']}")
        return 0
    token = os.environ.get(args.token_env)
    if not token:
        raise ValueError(f"required worker credential environment variable is absent: {args.token_env}")
    client = WorkerClient(worker_url, token)
    key = _stable_idempotency_key(args.out_dir, args.idempotency_key)
    started = client.request(
        "POST",
        "/api/persona-3b/start",
        {
            "studyId": manifest["studyId"],
            "phase": manifest["phase"],
            "manifestHash": manifest["manifestHash"],
            "sourceArtifactHash": manifest["source"]["artifactSha256"],
            "callCeiling": plan["callCeiling"],
            "budgetUsd": plan["budgetUsd"],
            "idempotencyKey": key,
        },
    )
    run_id = started.get("run", {}).get("id")
    if not isinstance(run_id, str):
        raise WorkerError("worker start response did not include a run id")
    write_json(args.out_dir / "start-receipt.json", {k: v for k, v in started.items() if k not in {"leaseToken"}})
    prior_receipt_path = args.out_dir / "final-receipt.json"
    if prior_receipt_path.exists():
        prior = read_json(prior_receipt_path)
        if prior.get("manifestHash") != manifest.get("manifestHash") or prior.get("sourceArtifactHash") != manifest.get("source", {}).get("artifactSha256"):
            raise ValueError("existing final receipt is bound to different frozen inputs")
        prior_status = client.request("GET", f"/api/persona-3b/status/{urllib.parse.quote(run_id, safe='')}")
        if prior_status.get("run", {}).get("status") == "complete":
            write_json(args.out_dir / "final-status.json", prior_status)
            print(f"worker_run_id={run_id} status=complete scores_path={args.out_dir / 'scores.jsonl'} parallelism={args.parallelism}")
            return 0
    ledger_path = args.out_dir / "relay-ledger.jsonl"
    score_path = args.out_dir / "scores.jsonl"
    scores = _load_scores(score_path)
    lock = threading.Lock()
    # Import remote receipts before claiming shards so a recovered controller
    # never retries a score that was already committed by the worker.
    _merge_remote_scores(client, run_id, scores, score_path, lock)
    repairs: dict[tuple[str, int], dict[str, Any]] = {}
    amendment_path = args.out_dir / "repair-amendment.json"
    if amendment_path.exists():
        repairs = _run_repairs(client, run_id, args, pairs, scores, lock)
        args.analysis_scores_path = _write_analysis_scores(args, scores, repairs)
    repeat_ids = _repeat_ids(pairs, mapping, int(manifest["reliability"]["repeatRows"]), int(mapping.get("seed", 20260915)) + 1)
    phases: dict[str, list[dict[str, Any]]] = {
        "primary": pairs,
        "repeat": [_swapped(pair) for pair in pairs if pair["record_id"] in repeat_ids],
    }
    for phase_rows in phases.values():
        _validate_phase_assignments(phase_rows)
    review_ids: list[str] = []
    review_done = False
    synthesis_claimed = False
    for _ in range(10_000):
        claimed: list[dict[str, Any]] = []
        for _claim in range(args.parallelism):
            item = client.request("GET", "/api/persona-3b/next")
            shard = item.get("shard")
            if not shard:
                break
            if str(shard.get("phase")) == "synthesis":
                # The worker may queue the zero-record synthesis lease in the
                # same request that completes adjudication. Consume it as the
                # phase transition signal; analysis and /synthesize remain
                # owner-local operations below.
                synthesis_claimed = True
                break
            claimed.append(shard)
        if synthesis_claimed:
            break
        if claimed:
            phase = str(claimed[0].get("phase"))
            if any(str(item.get("phase")) != phase for item in claimed):
                raise WorkerError("worker leased mixed phases in one parallel batch")
            if phase not in phases:
                raise WorkerError(f"worker leased unsupported phase {phase}")
            rows = phases[phase]
            for shard in claimed:
                assigned = _assigned_rows(shard, rows)
                if len(assigned) != int(shard.get("expectedRecords", 0)):
                    raise WorkerError(f"shard {shard.get('id')} assignment count does not match expectedRecords")
            results = _run_shards_parallel(
                claimed,
                lambda shard: _process_shard(client, shard, rows, phase, run_id, scores, score_path, ledger_path, lock, args.score_path),
                args.parallelism,
            )
            for result in results:
                write_json(args.out_dir / f"checkpoint-{phase}-{result['shardId']}.json", {"runId": run_id, **result})
            continue
        _merge_remote_scores(client, run_id, scores, score_path, lock)
        status = client.request("GET", f"/api/persona-3b/status/{urllib.parse.quote(run_id, safe='')}")
        write_json(args.out_dir / "last-status.json", status)
        current = status.get("run", {}).get("status")
        if current in {"complete", "failed", "cancelled"}:
            if current != "complete":
                raise WorkerError(f"worker run ended in {current}")
            break
        if current == "awaiting_adjudication" and not review_done:
            # One attempt from the shared reserve is held for the bounded
            # analyst call below; retries/third-judge reviews retain the
            # remaining 128 attempts.
            reserve = max(0, int(manifest["reliability"]["adjudicationReserve"]) - 1)
            review_path = args.out_dir / "review-plan.json"
            if review_path.exists():
                saved = read_json(review_path)
                review_ids = sorted(str(value) for value in saved.get("recordIds", []))
                if saved.get("planHash") != _review_plan_hash(run_id, review_ids):
                    raise ValueError("persisted review plan hash mismatch")
            else:
                review_ids = sorted(select_disagreements(scores, mapping, pairs, reserve))
                write_json(review_path, {"runId": run_id, "recordIds": review_ids, "planHash": _review_plan_hash(run_id, review_ids), "criterion": "primary/repeat canonical score disagreement"})
            client.request("POST", "/api/persona-3b/review-plan", {"runId": run_id, "recordIds": review_ids, "planHash": _review_plan_hash(run_id, review_ids)})
            phases["disagreement"] = [pair for pair in pairs if pair["record_id"] in set(review_ids)]
            review_done = True
            continue
        if current == "synthesizing" or (current == "running" and any(str(item.get("phase")) == "synthesis" and str(item.get("status")) in {"claimed", "running"} for item in status.get("shards", []))):
            # Synthesis is an API analyst phase; the relay performs it below
            # only after the complete numeric score corpus is locally present.
            break
        time.sleep(args.poll_seconds)
    _merge_remote_scores(client, run_id, scores, score_path, lock)
    primary_count = sum(1 for key in scores if key[1] == 0)
    repeat_count = sum(1 for key in scores if key[1] == 1)
    adjudication_count = sum(1 for key in scores if key[1] == 2)
    if primary_count != len(pairs) or repeat_count != len(repeat_ids):
        raise WorkerError(f"cannot analyze incomplete corpus: primary={primary_count}/{len(pairs)} repeat={repeat_count}/{len(repeat_ids)}")
    report, all32, all32_path = _run_analysis(args)
    strict_counts = report.get("counts", {})
    if strict_counts.get("strictValidComplete") is not True:
        write_json(
            args.out_dir / "strict-incomplete-receipt.json",
            {
                "status": "strict-validity-incomplete",
                "runId": run_id,
                "manifestHash": manifest["manifestHash"],
                "sourceArtifactHash": manifest["source"]["artifactSha256"],
                "counts": strict_counts,
                "invalidScoreReasons": report.get("invalidScoreReasons", {}),
            },
        )
        raise WorkerError(
            "strict-valid analysis incomplete; refusing synthesis/finalize "
            f"({strict_counts.get('primaryScores', 0)}/{strict_counts.get('pairs', len(pairs))} valid primary receipts)"
        )
    analysis_path = args.out_dir / "analysis" / "analysis.json"
    analysis_digest = sha256_json(report)
    packet_object = _analysis_packet(report, all32, scores, review_ids)
    packet = canonical_json(packet_object)
    if len(packet.encode("utf-8")) > 20_000:
        raise WorkerError("compact analyst packet exceeds the worker 20,000-character cap")
    packet_path = args.out_dir / "analysis" / "analyst-packet.json"
    packet_path.write_text(packet + "\n", encoding="utf-8", newline="\n")
    packet_digest = sha256_bytes(packet.encode("utf-8"))
    analyst_status = "pending"
    try:
        analyst = client.request("POST", "/api/persona-3b/synthesize", {"runId": run_id, "analysisDigest": analysis_digest, "packetDigest": packet_digest, "packet": packet})
        write_json(args.out_dir / "analysis" / "analyst-receipt.json", {**analyst, "analysisDigest": analysis_digest, "packetDigest": packet_digest})
        analyst_status = "complete"
    except WorkerError as exc:
        if exc.status not in {404, 409, 429}:
            raise
        write_json(args.out_dir / "analysis" / "analyst-receipt.json", {"status": "pending", "reason": str(exc), "analysisDigest": analysis_digest, "packetDigest": packet_digest})
    artifact = {"analysis": report, "all32FeatureTable": all32, "analysisDigest": analysis_digest, "packetDigest": packet_digest}
    artifact_path = args.out_dir / "final-artifact.json"
    write_json(artifact_path, artifact)
    artifact_hash = sha256_bytes(artifact_path.read_bytes())
    finalize = client.request(
        "POST",
        "/api/persona-3b/finalize",
        {
            "runId": run_id,
            "analysisDigest": analysis_digest,
            "artifactHash": artifact_hash,
            "manifestHash": manifest["manifestHash"],
            "sourceArtifactHash": manifest["source"]["artifactSha256"],
            "counts": {"sourceRecords": int(manifest["source"]["rows"]), "primary": len(pairs), "repeat": len(repeat_ids), "disagreement": adjudication_count, "synthesis": 1 if analyst_status == "complete" else 0},
            "shortlistSummary": report.get("selection", {}).get("shortlist", []),
            "confirmationStatus": "pending",
        },
    )
    write_json(args.out_dir / "final-receipt.json", {"status": "scored-development-screen-final-receipt", "runId": run_id, "manifestHash": manifest["manifestHash"], "sourceArtifactHash": manifest["source"]["artifactSha256"], "analysisDigest": analysis_digest, "artifactHash": artifact_hash, "all32TableDigest": sha256_bytes(all32_path.read_bytes()), "counts": {"sourceRecords": int(manifest["source"]["rows"]), "primaryPairRequests": len(pairs), "repeatPairRequests": len(repeat_ids), "adjudicationRecords": adjudication_count}, "shortlistSummary": report.get("selection", {}).get("shortlist", []), "confirmationStatus": "pending", "analystStatus": analyst_status, "workerFinalize": finalize})
    final = client.request("GET", f"/api/persona-3b/status/{urllib.parse.quote(run_id, safe='')}")
    write_json(args.out_dir / "final-status.json", final)
    if final.get("run", {}).get("status") != "complete":
        raise WorkerError(f"finalize did not produce complete status: {final.get('run', {}).get('status')}")
    print(f"worker_run_id={run_id} status=complete scores_path={score_path} parallelism={args.parallelism}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--worker-url")
    parser.add_argument("--env-file", type=Path, default=Path("research/experiment-003b/private-run/relay.env"))
    parser.add_argument("--token-env", default="AUTOLABS_PERSONA_3B_RELAY_TOKEN")
    parser.add_argument("--score-path", default="/api/persona-3b/score")
    parser.add_argument("--idempotency-key")
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--parallelism", type=int, default=5)
    parser.add_argument("--execute", action="store_true", help="start and drive the worker run; omitted means plan only")
    args = parser.parse_args()
    if not 1 <= args.parallelism <= 5:
        parser.error("--parallelism must be between 1 and 5")
    try:
        return run(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError, WorkerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


REPAIR_AMENDMENT_HASH = "51cb83b6325867a4480b456be35db26cfba72002ca79f5b663bb8d377d97edc8"
REPAIR_CALL_CEILING = 1400

def _repair_targets(scores: dict[tuple[str, int], dict[str, Any]]) -> dict[int, list[str]]:
    return {idx: sorted(record_id for (record_id, repeat_index), row in scores.items() if repeat_index == idx and row.get("_contractValid") is False) for idx in (0, 1, 2)}

def _repair_phase_body(shard: dict[str, Any], pair: dict[str, Any], repeat_index: int, repair_index: int) -> dict[str, Any]:
    row = _swapped(pair) if repeat_index == 1 else pair
    return {"shardId": shard["id"], "leaseToken": shard["leaseToken"], "recordId": pair["record_id"], "scenario": row["scenario"], "A": row["A"], "B": row["B"], "truncation": row["truncation"], "AResponseSha256": row["AResponseSha256"], "BResponseSha256": row["BResponseSha256"], "repeatIndex": repeat_index, "orderSwap": repeat_index == 1, "repairVersion": "repair-v1", "repairIndex": repair_index}

def _run_repairs(client: WorkerClient, run_id: str, args: argparse.Namespace, pairs: list[dict[str, Any]], scores: dict[tuple[str, int], dict[str, Any]], lock: threading.Lock) -> dict[tuple[str, int], dict[str, Any]]:
    targets = _repair_targets(scores)
    prior_plan = args.out_dir / "repair-target-plan.json"
    if not targets[0] and prior_plan.exists():
        saved_targets = read_json(prior_plan).get("targets", {})
        targets = {idx: sorted(str(value) for value in saved_targets.get(("primary", "repeat", "disagreement")[idx], [])) for idx in (0, 1, 2)}
    if not targets[0]:
        return {}
    pair_by_id = {str(row["record_id"]): row for row in pairs}
    repair_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    exhausted: list[dict[str, Any]] = []
    repair_path = args.out_dir / "repair-receipts.jsonl"
    try:
        remote = client.request("GET", f"/api/persona-3b/scores/{urllib.parse.quote(run_id, safe='' )}")
        for item in remote.get("scores", []):
            if isinstance(item, dict) and item.get("repairVersion") == "repair-v1" and isinstance(item.get("score"), dict):
                raw = {**item["score"], "repeatIndex": int(item.get("repeatIndex", 0)), "orderSwap": bool(item.get("orderSwap")), "repairVersion": "repair-v1", "repairIndex": int(item.get("repairIndex", 0)), "usage": {}}
                repair_rows.setdefault((str(raw.get("record_id")), int(raw["repeatIndex"])), []).append(_classify_score(raw, str(raw.get("record_id"))))
    except WorkerError as exc:
        if exc.status != 404: raise
    owner_path = args.out_dir / "repair-owner-token.txt"
    owner_token = owner_path.read_text(encoding="utf-8").strip() if owner_path.exists() else "repair-owner-" + str(uuid.uuid4())
    owner_path.write_text(owner_token + "\n", encoding="utf-8", newline="\n")
    target_hashes = {str(idx): sha256_json(targets[idx]) for idx in (0, 1, 2)}
    write_json(args.out_dir / "repair-target-plan.json", {"runId": run_id, "rule": "first subsequent strictlyvalid attempt for invalidrecord", "targets": {"primary": targets[0], "repeat": targets[1], "disagreement": targets[2]}, "targetHashes": {"primary": target_hashes["0"], "repeat": target_hashes["1"], "disagreement": target_hashes["2"]}})
    start = client.request("POST", "/api/persona-3b/repair-start", {"runId": run_id, "amendmentHash": REPAIR_AMENDMENT_HASH, "manifestHash": read_json(args.manifest)["manifestHash"], "sourceArtifactHash": read_json(args.manifest)["source"]["artifactSha256"], "originalCallCeiling": 1014, "effectiveCallCeiling": REPAIR_CALL_CEILING, "hardBudgetUsd": 10, "ownerToken": owner_token, "targetHashes": {"primary": target_hashes["0"], "repeat": target_hashes["1"], "disagreement": target_hashes["2"]}, "primaryRecordIds": targets[0], "repeatRecordIds": targets[1], "disagreementRecordIds": targets[2]})
    write_json(args.out_dir / "repair-start-receipt.json", start)
    shards = {str(item.get("phase")): item for item in start.get("shards", []) if isinstance(item, dict)}
    if repair_path.exists():
        for line in repair_path.read_text(encoding="utf-8").splitlines():
            try: item = json.loads(line)
            except json.JSONDecodeError: continue
            receipt = item.get("receipt") if isinstance(item, dict) else None
            if isinstance(receipt, dict) and receipt.get("repairVersion") == "repair-v1": repair_rows.setdefault((str(receipt.get("record_id")), int(receipt.get("repeatIndex", 0))), []).append(receipt)

    for repeat_index in (0, 1, 2):
        shard = shards.get(("primary", "repeat", "disagreement")[repeat_index])
        if not shard or shard.get("status") == "complete":
            continue
        for record_id in targets[repeat_index]:
            key = (record_id, repeat_index)
            if any(row.get("_contractValid") is True for row in repair_rows.get(key, [])): continue
            pair = pair_by_id.get(record_id)
            if pair is None:
                raise WorkerError(f"repair target {record_id} is absent from frozen pairs")
            chosen = None
            existing_rows = repair_rows.get(key, [])
            start_index = max((int(row.get("repairIndex", -1)) for row in existing_rows), default=-1) + 1
            for repair_index in range(start_index, REPAIR_CALL_CEILING - 955):
                body = _repair_phase_body(shard, pair, repeat_index, repair_index)
                result = client.request("POST", args.score_path, body)
                score = result.get("score")
                if not isinstance(score, dict):
                    if result.get("duplicate"):
                        raise WorkerError(f"repair receipt missing for duplicate {record_id}:{repeat_index}:{repair_index}")
                    raise WorkerError(f"repair response lacked score for {record_id}")
                try:
                    from contract import validate_contract_score
                    validated = validate_contract_score(score, record_id)
                    valid = True
                    error = None
                except ValueError as exc:
                    validated = score
                    valid = False
                    error = str(exc)
                receipt = {**validated, "repeatIndex": repeat_index, "orderSwap": repeat_index == 1, "repairVersion": "repair-v1", "repairIndex": repair_index, "usage": result.get("usage", {}), "_contractValid": valid}
                if error:
                    receipt["_validationError"] = error
                _append_jsonl(repair_path, {"runId": run_id, "recordId": record_id, "repeatIndex": repeat_index, "repairIndex": repair_index, "repairVersion": "repair-v1", "receipt": receipt}, lock)
                repair_rows.setdefault((record_id, repeat_index), []).append(receipt)
                if valid:
                    chosen = receipt
                    break
            if chosen is None:
                exhausted.append({"recordId": record_id, "repeatIndex": repeat_index, "attempts": len(repair_rows.get(key, []))})
    strict_complete = all(any(row.get("_contractValid") is True for row in repair_rows.get((record_id, idx), [])) for idx, names in enumerate((targets[0], targets[1], targets[2])) for record_id in names)
    if not strict_complete:
        raise WorkerError("strict-valid repair coverage incomplete; refusing repair finalization")
    client.request("POST", "/api/persona-3b/repair-start", {"runId": run_id, "strictValidComplete": strict_complete, "amendmentHash": REPAIR_AMENDMENT_HASH, "manifestHash": read_json(args.manifest)["manifestHash"], "sourceArtifactHash": read_json(args.manifest)["source"]["artifactSha256"], "originalCallCeiling": 1014, "effectiveCallCeiling": REPAIR_CALL_CEILING, "hardBudgetUsd": 10, "ownerToken": owner_token, "targetHashes": {"primary": target_hashes["0"], "repeat": target_hashes["1"], "disagreement": target_hashes["2"]}, "primaryRecordIds": targets[0], "repeatRecordIds": targets[1], "disagreementRecordIds": targets[2], "finalize": True})
    write_json(args.out_dir / "repair-selection.json", {"runId": run_id, "rule": "first subsequent strictlyvalid attempt for invalidrecord", "selected": [{"recordId": k[0], "repeatIndex": k[1], "repairIndex": v[-1].get("repairIndex")} for k, v in sorted(repair_rows.items()) if any(row.get("_contractValid") is True for row in v)], "exhausted": exhausted})
    return {key: next(row for row in rows if row.get("_contractValid") is True) for key, rows in repair_rows.items() if any(row.get("_contractValid") is True for row in rows)}


def _write_analysis_scores(args: argparse.Namespace, scores: dict[tuple[str, int], dict[str, Any]], repairs: dict[tuple[str, int], dict[str, Any]]) -> Path:
    view = dict(scores)
    view.update(repairs)
    path = args.out_dir / "analysis-scores.jsonl"
    write_jsonl(path, [view[key] for key in sorted(view)])
    return path
if __name__ == "__main__":
    raise SystemExit(main())
