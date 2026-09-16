import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import read_json, read_jsonl
from contract import ALL_DIMENSIONS, validate_contract_score
from prepare_dataset import prepare
from validate_input import validate

INTAKE = ROOT.parent / "experiment-003a" / "private-intake"


def test_existing_intake_counts_and_pairing():
    rows = read_jsonl(INTAKE / "completed-v5/persona-discovery/screen-responses.jsonl")
    cases = read_json(INTAKE / "recovery-dataset/persona-discovery/cases.json")
    features = read_json(INTAKE / "recovery-metadata/selection.json")["features"]
    normalized, report = validate(rows, cases, features)
    assert report["valid"] is True
    assert report["counts"]["rows"] == 780
    assert report["counts"]["truncated"] == 660
    outputs, pairs, mapping = prepare(normalized, 20260915)
    assert len(outputs) == 780 and len(pairs) == 768
    assert len(mapping["records"]) == 780 and len(mapping["pairs"]) == 768
    assert len(pairs[0]["record_id"]) == 8


def test_contract_score_is_strict_and_nullable():
    score_set = {name: None for name in ALL_DIMENSIONS}
    value = {"record_id": "p0000001", "scores": {"A": score_set, "B": score_set}, "pairwise": {name: {"direction": "unknown", "magnitude": 0} for name in ("relational_stance", "self_positioning", "assertiveness", "affective_tone", "voice_coherence")}, "evidence": [], "notes": ""}
    assert validate_contract_score(value, "p0000001")["record_id"] == "p0000001"
    for invalid in ({**value, "extra": 1}, {**value, "record_id": "wrong"}, {**value, "evidence": ["x"] * 4}):
        try:
            validate_contract_score(invalid, "p0000001")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid contract score was accepted")
