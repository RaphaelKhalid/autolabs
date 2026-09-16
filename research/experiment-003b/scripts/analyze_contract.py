"""Unblind contract-format A/B scores and apply the frozen 3b rubric."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import read_json, read_jsonl, sha256_json, write_json, write_jsonl
from contract import PERSONA_DIMENSIONS, QUALITY_DIMENSIONS, validate_contract_score


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def _composite(score: dict[str, Any]) -> float | None:
    values = [score[name] for name in PERSONA_DIMENSIONS if isinstance(score.get(name), int)]
    return statistics.fmean(values) if values else None


def filter_strict_scores(scores: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return frozen-contract-valid rows and preserve invalid rows for reporting."""
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for row in scores:
        if not isinstance(row, dict) or not isinstance(row.get("record_id"), str):
            invalid.append({"reason": "receipt lacks record_id", "row": row})
            continue
        body = {key: value for key, value in row.items() if key not in {"repeatIndex", "orderSwap", "repairVersion", "repairIndex", "usage", "_contractValid", "_validationError"}}
        try:
            validate_contract_score(body, row["record_id"])
        except ValueError as exc:
            invalid.append({"record_id": row["record_id"], "repeatIndex": int(row.get("repeatIndex", 0)), "reason": str(exc)})
        else:
            valid.append(row)
    return valid, invalid


def analyze(pairs: list[dict[str, Any]], scores: list[dict[str, Any]], mapping: dict[str, Any], manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    strict_scores, invalid_rows = filter_strict_scores(scores)
    primary = {str(row["record_id"]): row for row in strict_scores if int(row.get("repeatIndex", 0)) == 0}
    raw_primary = {str(row["record_id"]): row for row in scores if isinstance(row, dict) and isinstance(row.get("record_id"), str) and int(row.get("repeatIndex", 0)) == 0}
    invalid_primary = [row for row in invalid_rows if int(row.get("repeatIndex", 0)) == 0]
    mapped_pairs = mapping.get("pairs", {})
    if len(mapped_pairs) != len(pairs):
        raise ValueError("mapping and blinded pair counts differ")
    dimensions = list(PERSONA_DIMENSIONS + QUALITY_DIMENSIONS + ("refusal_safety", "lexical_format_artifact"))
    effects: list[dict[str, Any]] = []
    grouped: defaultdict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        pair_id = pair["record_id"]
        score = primary.get(pair_id)
        meta = mapped_pairs.get(pair_id)
        if score is None:
            continue
        if meta is None:
            raise ValueError(f"missing score or mapping for {pair_id}")
        validated_a = score["scores"]["A"]; validated_b = score["scores"]["B"]
        if meta["ARecordId"] == meta["interventionRecordId"]:
            intervention, baseline = validated_a, validated_b
        elif meta["BRecordId"] == meta["interventionRecordId"]:
            intervention, baseline = validated_b, validated_a
        else:
            raise ValueError(f"mapping roles do not match pair {pair_id}")
        delta: dict[str, float] = {}
        for dimension in dimensions:
            if isinstance(intervention.get(dimension), int) and isinstance(baseline.get(dimension), int):
                delta[dimension] = intervention[dimension] - baseline[dimension]
        baseline_composite = _composite(baseline); intervention_composite = _composite(intervention)
        composite_delta = intervention_composite - baseline_composite if baseline_composite is not None and intervention_composite is not None else None
        truncation = pair.get("truncation", {})
        row = {"pairId": pair_id, "caseId": meta["caseId"], "feature": int(meta["feature"]), "sign": meta["condition"], "delta": delta, "personaCompositeDelta": composite_delta, "truncated": bool(truncation.get("A") or truncation.get("B")), "pairwise": score.get("pairwise")}
        effects.append(row); grouped[(row["feature"], row["sign"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (feature, sign), rows in sorted(grouped.items()):
        dimension_values = {dimension: [r["delta"][dimension] for r in rows if dimension in r["delta"]] for dimension in dimensions}
        untruncated = [r for r in rows if not r["truncated"]]
        untruncated_values = {dimension: [r["delta"][dimension] for r in untruncated if dimension in r["delta"]] for dimension in dimensions}
        persona = {dimension: dimension_values[dimension] for dimension in PERSONA_DIMENSIONS}
        best_key = max(PERSONA_DIMENSIONS, key=lambda d: abs(statistics.median(persona[d])) if persona[d] else -1)
        summaries.append({"feature": feature, "sign": sign, "nScenarios": len(rows), "coverage": {d: len(dimension_values[d]) for d in dimensions}, "meanDelta": {d: _mean(v) for d, v in dimension_values.items()}, "medianDelta": {d: statistics.median(v) if v else None for d, v in dimension_values.items()}, "untruncatedN": len(untruncated), "untruncatedMeanDelta": {d: _mean(v) for d, v in untruncated_values.items()}, "artifactMajority": sum(1 for r in rows if (r["delta"].get("lexical_format_artifact", 0) >= 2)) > len(rows) / 2, "bestPersonaDimension": best_key})

    rules = manifest.get("analysis", {})
    threshold_results: dict[str, Any] = {}
    provisional: list[dict[str, Any]] = []
    for summary in summaries:
        feature = summary["feature"]; sign = summary["sign"]; opposite = next((s for s in summaries if s["feature"] == feature and s["sign"] != sign), None)
        checks: dict[str, Any] = {"coverage": False, "consistency": False, "signContrast": False, "quality": False, "untruncatedRobustness": "insufficient-observation"}
        passing_keys: list[str] = []
        for dimension in PERSONA_DIMENSIONS:
            values = [r["delta"].get(dimension) for r in grouped[(feature, sign)] if r["delta"].get(dimension) is not None]
            if len(values) >= 8 and sum(1 for v in values if v > 0) >= 7 or len(values) >= 8 and sum(1 for v in values if v < 0) >= 7:
                if abs(statistics.median(values)) >= 0.4: passing_keys.append(dimension)
        checks["coverage"] = all(summary["coverage"][d] >= 8 for d in PERSONA_DIMENSIONS)
        checks["consistency"] = bool(passing_keys)
        if opposite:
            contrasts = [abs(summary["medianDelta"][d] - opposite["medianDelta"][d]) for d in PERSONA_DIMENSIONS if summary["medianDelta"].get(d) is not None and opposite["medianDelta"].get(d) is not None]
            checks["signContrast"] = bool(contrasts and max(contrasts) >= 0.5)
        checks["quality"] = (summary["meanDelta"].get("task_completion") is None or summary["meanDelta"]["task_completion"] >= -0.75) and (summary["meanDelta"].get("topical_fidelity") is None or summary["meanDelta"]["topical_fidelity"] >= -0.75) and not summary["artifactMajority"]
        if summary["untruncatedN"] >= 4:
            checks["untruncatedRobustness"] = any(abs(summary["untruncatedMeanDelta"].get(d) or 0) >= 0.4 for d in PERSONA_DIMENSIONS)
        eligible = all(value is True for value in checks.values())
        threshold_results[f"{feature}:{sign}"] = {"eligible": eligible, "checks": checks, "passingPersonaDimensions": passing_keys}
        if eligible: provisional.append({"feature": feature, "sign": sign, "score": max(abs(summary["medianDelta"].get(d) or 0) for d in PERSONA_DIMENSIONS)})
    provisional.sort(key=lambda x: x["score"], reverse=True)
    missing_primary = len(pairs) - len(primary)
    report = {"schemaVersion": 1, "status": "analyzed-development-screen" if missing_primary == 0 else "analyzed-development-screen-incomplete", "counts": {"pairs": len(pairs), "rawPrimaryScores": len(raw_primary), "primaryScores": len(primary), "invalidPrimaryScores": len(invalid_primary), "missingPrimaryScores": missing_primary, "strictValidComplete": missing_primary == 0, "features": len({s['feature'] for s in summaries}), "truncatedPairs": sum(1 for r in effects if r["truncated"])}, "invalidScoreReasons": dict(Counter(str(row.get("reason", "unknown")) for row in invalid_primary)), "sourceHashes": {"pairs": sha256_json(pairs), "scores": sha256_json(scores), "mapping": sha256_json(mapping)}, "selection": {"rulesFrozen": True, "decisions": threshold_results, "shortlist": provisional[:int(rules.get("maxCandidates", 3))], "allowZeroCandidates": True}, "reliability": {"repeatScores": sum(1 for row in strict_scores if int(row.get("repeatIndex", 0)) == 1), "humanAudit": "pending"}, "interpretation": "Matched scenario-level deltas from frozen-contract-valid receipts only; invalid receipts remain preserved in the raw corpus and are reported separately. Truncation is censoring metadata. This is provisional development evidence, not persona discovery or confirmation."}
    return report, summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True); parser.add_argument("--scores", type=Path, required=True); parser.add_argument("--mapping", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report, summaries = analyze(read_jsonl(args.pairs), read_jsonl(args.scores), read_json(args.mapping), read_json(args.manifest)); args.out_dir.mkdir(parents=True, exist_ok=True); write_json(args.out_dir / "analysis.json", report); write_jsonl(args.out_dir / "feature-sign-summary.jsonl", summaries); print(f"analyzed pairs={report['counts']['pairs']} shortlist={report['selection']['shortlist']}"); return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
