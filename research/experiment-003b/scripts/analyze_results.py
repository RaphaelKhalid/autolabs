"""Unblind Luna scores and summarize matched 3b effects.

This is a development analysis. It reports ranking and frozen-rule status; it
does not call candidates personas and does not pool tokens as observations.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import read_json, read_jsonl, sha256_json, write_json, write_jsonl


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.normalized)
    mapping = read_json(args.mapping)
    score_rows = read_jsonl(args.scores)
    scores: dict[tuple[str, int], dict[str, Any]] = {}
    for score in score_rows:
        scores[(str(score["recordId"]), int(score.get("repeatIndex", 0)))] = score
    primary = {(record_id): score for (record_id, repeat), score in scores.items() if repeat == 0}
    if len(primary) != 780:
        raise ValueError(f"analysis requires primary score for all 780 outputs; found {len(primary)}")
    normalized_by_index = {int(row["sourceIndex"]): row for row in rows}
    records = mapping.get("records", {})
    if len(records) != 780:
        raise ValueError(f"mapping requires 780 records; found {len(records)}")
    for record_id, meta in records.items():
        if record_id not in primary:
            raise ValueError(f"missing score for {record_id}")
        row = normalized_by_index.get(int(meta["sourceIndex"]))
        if not row or primary[record_id].get("responseSha256") != row.get("outputSha256"):
            raise ValueError(f"response hash mismatch for {record_id}")

    by_case: defaultdict[str, dict[str, tuple[str, dict[str, Any]]]] = defaultdict(dict)
    for record_id, meta in records.items():
        key = "baseline" if meta["condition"] == "baseline" else f"{meta['condition']}:{meta['feature']}"
        by_case[str(meta["caseId"])][key] = (record_id, meta)
    feature_ids = sorted({int(meta["feature"]) for meta in records.values() if meta["feature"] is not None})
    dimensions = list(primary[next(iter(primary))]["scores"])
    records_out: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for feature in feature_ids:
        feature_summary: dict[str, Any] = {"feature": feature, "signs": {}, "pairedComparisons": 0}
        all_effects: dict[str, list[float]] = defaultdict(list)
        for condition, sign_name in (("sae-positive", "positive"), ("sae-negative", "negative")):
            effects: dict[str, list[float]] = defaultdict(list)
            scenario_effects: list[dict[str, Any]] = []
            for case_id, values in sorted(by_case.items()):
                intervention_key = f"{condition}:{feature}"
                if intervention_key not in values or "baseline" not in values:
                    continue
                intervention_id, intervention_meta = values[intervention_key]
                baseline_id, baseline_meta = values["baseline"]
                intervention_score = primary[intervention_id]["scores"]
                baseline_score = primary[baseline_id]["scores"]
                deltas = {dimension: intervention_score[dimension] - baseline_score[dimension] for dimension in dimensions}
                for dimension, delta in deltas.items():
                    effects[dimension].append(delta)
                    all_effects[f"{sign_name}:{dimension}"].append(delta)
                records_out.append({
                    "caseId": case_id,
                    "feature": feature,
                    "sign": sign_name,
                    "baselineRecordId": baseline_id,
                    "interventionRecordId": intervention_id,
                    "baselineTruncated": baseline_meta["truncated"],
                    "interventionTruncated": intervention_meta["truncated"],
                    "deltas": deltas,
                })
                scenario_effects.append({"caseId": case_id, "deltas": deltas, "truncated": bool(baseline_meta["truncated"] or intervention_meta["truncated"])})
            feature_summary["pairedComparisons"] += len(scenario_effects)
            feature_summary["signs"][sign_name] = {
                "nScenarios": len(scenario_effects),
                "meanDelta": {dimension: mean(values) for dimension, values in effects.items()},
                "medianDelta": {dimension: statistics.median(values) if values else None for dimension, values in effects.items()},
                "scenarioConsistency": {dimension: (sum(1 for value in values if value > 0) / len(values) if values else None) for dimension, values in effects.items()},
                "truncationExcludedMeanDelta": {
                    dimension: mean([entry["deltas"][dimension] for entry in scenario_effects if not entry["truncated"]])
                    for dimension in dimensions
                },
                "scenarioEffects": scenario_effects,
            }
        positive = feature_summary["signs"].get("positive", {})
        negative = feature_summary["signs"].get("negative", {})
        feature_summary["signSensitivity"] = {
            dimension: (positive.get("meanDelta", {}).get(dimension) - negative.get("meanDelta", {}).get(dimension))
            if positive.get("meanDelta", {}).get(dimension) is not None and negative.get("meanDelta", {}).get(dimension) is not None else None
            for dimension in dimensions
        }
        summaries.append(feature_summary)

    # Repeated grader agreement is reported separately from experimental effects.
    agreement: dict[str, Any] = {"repeatRows": 0, "meanAbsoluteDifference": {dimension: None for dimension in dimensions}}
    diffs: defaultdict[str, list[float]] = defaultdict(list)
    for record_id in primary:
        repeat = scores.get((record_id, 1))
        if repeat:
            agreement["repeatRows"] += 1
            for dimension in dimensions:
                diffs[dimension].append(abs(primary[record_id]["scores"][dimension] - repeat["scores"][dimension]))
    agreement["meanAbsoluteDifference"] = {dimension: mean(values) for dimension, values in diffs.items()}

    rules = read_json(args.manifest).get("selectionRules", {})
    frozen = bool(rules.get("frozen", False))
    shortlist: list[int] = []
    rule_results: dict[str, Any] = {}
    if frozen:
        for summary in summaries:
            failures: list[str] = []
            min_consistency = float(rules.get("minScenarioConsistency", 0))
            persona_dimensions = rules.get("personaDimensions", ["style_voice_shift"])
            for dimension in persona_dimensions:
                for sign in ("positive", "negative"):
                    consistency = summary["signs"][sign]["scenarioConsistency"].get(dimension)
                    if consistency is None or consistency < min_consistency:
                        failures.append(f"{sign}:{dimension}:consistency")
            result = {"eligible": not failures, "failures": failures}
            rule_results[str(summary["feature"])] = result
            if result["eligible"]:
                shortlist.append(summary["feature"])
        shortlist = shortlist[: int(rules.get("maximumCandidates", 3))]

    report = {
        "schemaVersion": 1,
        "status": "scored-development-screen",
        "interpretation": "Effects are matched within scenario; truncated outputs are reported as a separate endpoint and are not treated as complete behavior.",
        "counts": {"sourceRows": len(rows), "primaryScores": len(primary), "pairedComparisons": len(records_out), "features": len(summaries), "truncatedRows": sum(1 for row in rows if row["truncated"])},
        "sourceHashes": {"normalized": sha256_json(rows), "mapping": sha256_json(mapping), "scores": sha256_json(score_rows)},
        "graderAgreement": agreement,
        "selection": {"rulesFrozen": frozen, "ruleResults": rule_results, "shortlist": shortlist, "maximumCandidates": int(rules.get("maximumCandidates", 3))},
    }
    write_json(args.out_dir / "analysis.json", report)
    write_jsonl(args.out_dir / "feature-summary.jsonl", summaries)
    write_jsonl(args.out_dir / "matched-differences.jsonl", records_out)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = summarize(args)
        print(f"analyzed comparisons={report['counts']['pairedComparisons']} features={report['counts']['features']} shortlist={report['selection']['shortlist']}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
