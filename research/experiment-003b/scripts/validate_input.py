"""Validate and normalize the completed Experiment 3a development screen.

This command is deliberately offline. It never calls a model or provider. The
output is the evidence ledger consumed by blinding and analysis.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import manifest_features, read_json, read_jsonl, sha256_json, sha256_text, write_json, write_jsonl


EXPECTED_CONDITIONS = {"baseline": 12, "sae-positive": 384, "sae-negative": 384}


def _features(manifest_path: Path | None, selection_path: Path | None) -> tuple[list[int], str]:
    if manifest_path:
        return manifest_features(read_json(manifest_path)), str(manifest_path)
    if selection_path:
        metadata = read_json(selection_path)
        values = metadata.get("features")
        if isinstance(values, list):
            return [int(x) for x in values], str(selection_path)
    raise ValueError("provide --manifest or --selection metadata to identify the 32 selected features")


def validate(rows: list[dict[str, Any]], cases: list[dict[str, Any]], features: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    errors: list[str] = []
    if len(rows) != 780:
        errors.append(f"expected exactly 780 response rows, found {len(rows)}")
    if len(features) != 32:
        errors.append(f"expected exactly 32 feature ids, found {len(features)}")

    cases_by_id = {str(case.get("id")): case for case in cases if case.get("split") == "development-screen"}
    if len(cases_by_id) != 12:
        errors.append(f"expected 12 development-screen cases, found {len(cases_by_id)}")

    condition_counts = Counter(str(row.get("condition")) for row in rows)
    for condition, expected in EXPECTED_CONDITIONS.items():
        if condition_counts[condition] != expected:
            errors.append(f"condition {condition!r}: expected {expected}, found {condition_counts[condition]}")
    unknown_conditions = set(condition_counts) - set(EXPECTED_CONDITIONS)
    if unknown_conditions:
        errors.append(f"unknown conditions: {sorted(unknown_conditions)}")

    normalized: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    by_case_condition: defaultdict[str, Counter[str]] = defaultdict(Counter)
    feature_counts: Counter[int] = Counter()
    for index, row in enumerate(rows):
        case_id = str(row.get("caseId", ""))
        condition = str(row.get("condition", ""))
        feature_value = row.get("feature")
        feature = None if feature_value is None else int(feature_value)
        key = (case_id, condition, "baseline" if feature is None else str(feature))
        if key in seen_keys:
            errors.append(f"duplicate response key at row {index}: {key}")
        seen_keys.add(key)
        if case_id not in cases_by_id:
            errors.append(f"row {index}: caseId is not one of the 12 development cases: {case_id}")
        else:
            expected_prompt = str(cases_by_id[case_id].get("prompt", ""))
            if row.get("prompt") != expected_prompt:
                errors.append(f"row {index}: prompt does not match case bank for {case_id}")
        if condition == "baseline" and feature is not None:
            errors.append(f"row {index}: baseline must have null feature")
        if condition != "baseline" and feature not in features:
            errors.append(f"row {index}: intervention feature {feature!r} is not selected")
        if condition != "baseline" and not isinstance(row.get("amplitude"), (int, float)):
            errors.append(f"row {index}: intervention amplitude is missing")
        if not isinstance(row.get("output"), str):
            errors.append(f"row {index}: output must be a string")
        if not isinstance(row.get("truncated"), bool):
            errors.append(f"row {index}: truncated must be boolean")
        if row.get("generatedTokens") != len(row.get("generatedTokenIds", [])):
            errors.append(f"row {index}: generatedTokens does not match generatedTokenIds length")
        by_case_condition[case_id][condition] += 1
        if feature is not None:
            feature_counts[feature] += 1
        normalized.append({
            "sourceIndex": index,
            "caseId": case_id,
            "prompt": row.get("prompt", ""),
            "output": row.get("output", ""),
            "outputSha256": sha256_text(str(row.get("output", ""))),
            "condition": condition,
            "feature": feature,
            "amplitude": row.get("amplitude"),
            "truncated": row.get("truncated"),
            "generatedTokens": row.get("generatedTokens"),
            "maxNewTokens": row.get("maxNewTokens"),
            "seed": row.get("seed"),
            "temperature": row.get("temperature"),
            "topP": row.get("topP"),
            "phase": row.get("phase"),
        })

    missing_features = sorted(set(features) - set(feature_counts))
    if missing_features:
        errors.append(f"selected features absent from responses: {missing_features}")
    for case_id in sorted(cases_by_id):
        counts = by_case_condition[case_id]
        if counts["baseline"] != 1 or counts["sae-positive"] != 32 or counts["sae-negative"] != 32:
            errors.append(f"case {case_id}: expected 1/32/32 rows, found {dict(counts)}")
    if any(row.get("phase") != "development-screen" for row in rows):
        errors.append("all rows must have phase=development-screen")

    report = {
        "schemaVersion": 1,
        "valid": not errors,
        "errors": errors,
        "counts": {
            "rows": len(rows),
            "cases": len(cases_by_id),
            "baseline": condition_counts["baseline"],
            "saePositive": condition_counts["sae-positive"],
            "saeNegative": condition_counts["sae-negative"],
            "truncated": sum(1 for row in rows if row.get("truncated") is True),
        },
        "featureIds": features,
        "featureCounts": {str(feature): feature_counts[feature] for feature in features},
        "sourceResponseSha256": sha256_json(rows),
        "caseBankSha256": sha256_json(cases),
    }
    return normalized, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        features, feature_source = _features(args.manifest, args.selection)
        rows, report = validate(read_jsonl(args.responses), read_json(args.cases), features)
        report["featureSource"] = feature_source
        write_json(args.out_dir / "validation.json", report)
        if report["valid"]:
            write_jsonl(args.out_dir / "normalized-responses.jsonl", rows)
        print(f"valid={report['valid']} rows={report['counts']['rows']} truncated={report['counts']['truncated']}")
        if report["errors"]:
            for error in report["errors"]:
                print(f"ERROR: {error}", file=sys.stderr)
        return 0 if report["valid"] else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
