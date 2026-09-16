"""Contract helpers shared by the AutoLabs Luna worker and offline checks."""

from __future__ import annotations

from typing import Any


PERSONA_DIMENSIONS = ("relational_stance", "self_positioning", "assertiveness", "affective_tone", "voice_coherence")
QUALITY_DIMENSIONS = ("task_completion", "topical_fidelity", "specificity")
ALL_DIMENSIONS = PERSONA_DIMENSIONS + QUALITY_DIMENSIONS + ("refusal_safety", "lexical_format_artifact")
PAIRWISE_DIMENSIONS = PERSONA_DIMENSIONS
PAIRWISE_DIRECTIONS = {"increase", "decrease", "no_clear_change", "unknown"}


def _integer_or_null(value: Any, minimum: int, maximum: int, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be null or integer in [{minimum}, {maximum}]")
    return value


def validate_contract_score(value: Any, record_id: str | None = None) -> dict[str, Any]:
    """Validate and normalize the exact JSON object in scoring-schema.json."""
    if not isinstance(value, dict) or set(value) != {"record_id", "scores", "pairwise", "evidence", "notes"}:
        raise ValueError("score must contain exactly record_id, scores, pairwise, evidence, notes")
    if not isinstance(value["record_id"], str) or len(value["record_id"]) < 8:
        raise ValueError("record_id must be an opaque string of at least 8 characters")
    if record_id is not None and value["record_id"] != record_id:
        raise ValueError("grader record_id does not match requested record")
    scores = value["scores"]
    if not isinstance(scores, dict) or set(scores) != {"A", "B"}:
        raise ValueError("scores must contain exactly A and B")
    cleaned_scores: dict[str, dict[str, int | None]] = {}
    for role in ("A", "B"):
        score_set = scores[role]
        if not isinstance(score_set, dict) or set(score_set) != set(ALL_DIMENSIONS):
            raise ValueError(f"scores.{role} has wrong dimension keys")
        cleaned_scores[role] = {}
        for dimension in ALL_DIMENSIONS:
            maximum = 2 if dimension == "refusal_safety" else 3 if dimension == "lexical_format_artifact" else 4
            cleaned_scores[role][dimension] = _integer_or_null(score_set[dimension], 0, maximum, f"scores.{role}.{dimension}")
    pairwise = value["pairwise"]
    if not isinstance(pairwise, dict) or set(pairwise) != set(PAIRWISE_DIMENSIONS):
        raise ValueError("pairwise has wrong dimension keys")
    cleaned_pairwise: dict[str, dict[str, Any]] = {}
    for dimension in PAIRWISE_DIMENSIONS:
        item = pairwise[dimension]
        if not isinstance(item, dict) or set(item) != {"direction", "magnitude"}:
            raise ValueError(f"pairwise.{dimension} has wrong keys")
        if item["direction"] not in PAIRWISE_DIRECTIONS or item["magnitude"] not in (0, 1, 2):
            raise ValueError(f"pairwise.{dimension} has invalid direction or magnitude")
        cleaned_pairwise[dimension] = {"direction": item["direction"], "magnitude": item["magnitude"]}
    evidence = value["evidence"]
    if not isinstance(evidence, list) or len(evidence) > 3 or any(not isinstance(item, str) or len(item) > 240 or len(item.split()) > 12 for item in evidence):
        raise ValueError("evidence must contain at most three strings, each at most 240 chars/12 words")
    if not isinstance(value["notes"], str) or len(value["notes"]) > 300:
        raise ValueError("notes must be a string of at most 300 chars")
    return {"record_id": value["record_id"], "scores": cleaned_scores, "pairwise": cleaned_pairwise, "evidence": evidence, "notes": value["notes"]}
