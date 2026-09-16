"""Build contract-aligned blinded outputs and A/B matched pairs for 3b."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import forbidden_leakage, read_jsonl, sha256_json, write_json, write_jsonl


def prepare(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_case: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        key = "baseline" if row["condition"] == "baseline" else f"{row['condition']}:{row['feature']}"
        by_case[row["caseId"]][key] = row
    if len(by_case) != 12:
        raise ValueError(f"expected 12 cases, found {len(by_case)}")
    rng = random.Random(seed)
    outputs: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    mapping: dict[str, Any] = {"schemaVersion": 1, "seed": seed, "records": {}, "pairs": {}}

    def add_output(row: dict[str, Any]) -> str:
        record_id = f"r{len(outputs) + 1:07d}"
        outputs.append({
            "recordId": record_id,
            "scenario": row["prompt"],
            "response": row["output"],
            "responseSha256": row["outputSha256"],
            "truncated": bool(row["truncated"]),
            "generatedTokens": row["generatedTokens"],
        })
        mapping["records"][record_id] = {
            "sourceIndex": row["sourceIndex"], "caseId": row["caseId"],
            "condition": row["condition"], "feature": row["feature"],
            "amplitude": row["amplitude"], "responseSha256": row["outputSha256"],
            "truncated": bool(row["truncated"]),
        }
        return record_id

    for case_id in sorted(by_case):
        values = by_case[case_id]
        baseline = values.get("baseline")
        if baseline is None:
            raise ValueError(f"case {case_id} has no baseline")
        baseline_id = add_output(baseline)
        for condition in ("sae-positive", "sae-negative"):
            for key in sorted(k for k in values if k.startswith(condition + ":")):
                intervention = values[key]
                intervention_id = add_output(intervention)
                pair_id = f"p{len(pairs) + 1:07d}"
                roles = [("baseline", baseline_id), ("intervention", intervention_id)]
                rng.shuffle(roles)
                output_by_id = {item["recordId"]: item for item in outputs}
                a_id, b_id = roles[0][1], roles[1][1]
                pairs.append({
                    "record_id": pair_id,
                    "scenario": intervention["prompt"],
                    "truncation": {"A": output_by_id[a_id]["truncated"], "B": output_by_id[b_id]["truncated"]},
                    "A": output_by_id[a_id]["response"], "B": output_by_id[b_id]["response"],
                    "AResponseSha256": output_by_id[a_id]["responseSha256"],
                    "BResponseSha256": output_by_id[b_id]["responseSha256"],
                })
                mapping["pairs"][pair_id] = {
                    "caseId": case_id, "feature": intervention["feature"],
                    "condition": intervention["condition"], "baselineRecordId": baseline_id,
                    "interventionRecordId": intervention_id, "ARecordId": a_id, "BRecordId": b_id,
                }
    if len(outputs) != 780 or len(pairs) != 768:
        raise ValueError(f"expected 780 outputs and 768 pairs, found {len(outputs)} and {len(pairs)}")
    mapping["counts"] = {"uniqueOutputs": 780, "pairs": 768, "cases": 12}
    mapping["publicOutputSha256"] = sha256_json(outputs)
    mapping["publicPairSha256"] = sha256_json(pairs)
    return outputs, pairs, mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    try:
        outputs, pairs, mapping = prepare(read_jsonl(args.normalized), args.seed)
        names = ("blind-outputs.jsonl", "blind-pairs.jsonl", "private-mapping.json")
        if any(forbidden_leakage(name) for name in names):
            raise ValueError("blind filename leakage check failed")
        write_jsonl(args.out_dir / names[0], outputs)
        write_jsonl(args.out_dir / names[1], pairs)
        write_json(args.out_dir / names[2], mapping)
        write_json(args.out_dir / "blind-manifest.json", {
            "schemaVersion": 1, "uniqueOutputCount": 780, "pairedComparisonCount": 768,
            "blindSeed": args.seed, "privateMapping": names[2], "leakageCheck": "passed",
        })
        print("prepared unique_outputs=780 pairs=768")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
