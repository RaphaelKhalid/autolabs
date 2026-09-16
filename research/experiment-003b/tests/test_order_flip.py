import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from contract import ALL_DIMENSIONS
from order_flip import order_flip_report


def _scores(a, b):
    return {"A": {key: a for key in ALL_DIMENSIONS}, "B": {key: b for key in ALL_DIMENSIONS}}


def test_order_swap_is_canonicalized_before_flip_rate():
    mapping = {"pairs": {"p0000001": {"ARecordId": "r0000001", "BRecordId": "r0000002", "interventionRecordId": "r0000001"}}}
    scores = [{"record_id": "p0000001", "repeatIndex": 0, "scores": _scores(3, 1)}, {"record_id": "p0000001", "repeatIndex": 1, "orderSwap": True, "scores": _scores(1, 3)}]
    result = order_flip_report(scores, mapping)
    assert result["repeatPairs"] == 1
    assert result["flippedCells"] == 0
    assert result["orderFlipRate"] == 0


def test_real_change_is_counted_as_flip():
    mapping = {"pairs": {"p0000001": {"ARecordId": "r0000001", "BRecordId": "r0000002", "interventionRecordId": "r0000001"}}}
    scores = [{"record_id": "p0000001", "repeatIndex": 0, "scores": _scores(3, 1)}, {"record_id": "p0000001", "repeatIndex": 1, "orderSwap": True, "scores": _scores(3, 2)}]
    result = order_flip_report(scores, mapping)
    assert result["flippedCells"] == len(ALL_DIMENSIONS)
    assert result["orderFlipRate"] == 1
