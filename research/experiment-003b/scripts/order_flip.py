"""Compute order-swapped repeat reliability after private role unblinding."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from contract import ALL_DIMENSIONS


def _sign(value: Any) -> int | None:
    if not isinstance(value, int):
        return None
    return 1 if value > 0 else -1 if value < 0 else 0


def order_flip_report(scores: list[dict[str, Any]], mapping: dict[str, Any]) -> dict[str, Any]:
    """Compare primary and order-swapped repeat deltas in canonical roles.

    ``orderSwap`` is written by the worker on repeat rows. For compatibility,
    ``repeatIndex=1`` is treated as swapped when the flag is omitted. The
    function never exposes feature identities; callers may keep its output
    private with the rest of the analysis report.
    """
    by_record: defaultdict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in scores:
        by_record[str(row["record_id"])][int(row.get("repeatIndex", 0))] = row
    pair_mapping = mapping.get("pairs", {})
    per_dimension: dict[str, dict[str, int]] = {dimension: {"compared": 0, "flips": 0} for dimension in ALL_DIMENSIONS}
    for pair_id, meta in pair_mapping.items():
        primary = by_record.get(str(pair_id), {}).get(0)
        repeat = by_record.get(str(pair_id), {}).get(1)
        if not primary or not repeat:
            continue
        primary_intervention_role = "A" if meta.get("ARecordId") == meta.get("interventionRecordId") else "B"
        swapped = bool(repeat.get("orderSwap", True))
        repeat_intervention_role = ("B" if primary_intervention_role == "A" else "A") if swapped else primary_intervention_role
        primary_scores = primary.get("scores", {}); repeat_scores = repeat.get("scores", {})
        baseline_role = "B" if primary_intervention_role == "A" else "A"
        repeat_baseline_role = "B" if repeat_intervention_role == "A" else "A"
        for dimension in ALL_DIMENSIONS:
            a = primary_scores.get(primary_intervention_role, {}).get(dimension); b = primary_scores.get(baseline_role, {}).get(dimension)
            c = repeat_scores.get(repeat_intervention_role, {}).get(dimension); d = repeat_scores.get(repeat_baseline_role, {}).get(dimension)
            first = _sign(a - b) if isinstance(a, int) and isinstance(b, int) else None
            second = _sign(c - d) if isinstance(c, int) and isinstance(d, int) else None
            if first is None or second is None:
                continue
            per_dimension[dimension]["compared"] += 1
            if first != second:
                per_dimension[dimension]["flips"] += 1
    compared = sum(item["compared"] for item in per_dimension.values())
    flips = sum(item["flips"] for item in per_dimension.values())
    return {"repeatPairs": sum(1 for rows in by_record.values() if 0 in rows and 1 in rows), "comparedCells": compared, "flippedCells": flips, "orderFlipRate": flips / compared if compared else None, "byDimension": {dimension: {**item, "rate": item["flips"] / item["compared"] if item["compared"] else None} for dimension, item in per_dimension.items()}}
