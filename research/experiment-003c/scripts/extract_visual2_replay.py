"""Build lib/persona-3c-visual-2.ts (the site replay dataset) from the visual-2
artifacts and the harness status JSON for that run.

Usage (from repo root):
  curl -s "$WORKER/api/persona-3c/status?runId=persona-3c-c58185a5-f7a0-49d0-a8dd-47556bbce5b3" > /tmp/v2status.json
  python research/experiment-003c/scripts/extract_visual2_replay.py /tmp/v2status.json

Only recorded values are copied; nothing is estimated here.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "research/experiment-003c/smoke-runs/visual-2"
OUT = ROOT / "lib/persona-3c-visual-2.ts"
TRAIN = [  # train-checkpoint-* payloads from persona_3c_records for the run
    {"tokens": 1004535, "steps": 772, "fve": 0.5612, "dead": 0.7294, "loss": 1.0738},
    {"tokens": 2006225, "steps": 1556, "fve": 0.6547, "dead": 0.4449, "loss": 0.8943},
    {"tokens": 3008519, "steps": 2348, "fve": 0.6881, "dead": 0.2511, "loss": 0.8104},
    {"tokens": 4011931, "steps": 3144, "fve": 0.7075, "dead": 0.1382, "loss": 0.7476},
    {"tokens": 5017555, "steps": 3940, "fve": 0.7194, "dead": 0.0693, "loss": 0.7143},
    {"tokens": 6018473, "steps": 4748, "fve": 0.7262, "dead": 0.0348, "loss": 0.6899},
    {"tokens": 7021604, "steps": 5536, "fve": 0.7323, "dead": 0.0146, "loss": 0.6702},
    {"tokens": 8002995, "steps": 6320, "fve": 0.7332, "dead": 0.0042, "loss": 0.6667},
]
EXAMPLES = [
    ("feature", 699, -1, "self-description", "Feature 699 (−), dose 2.0"),
    ("feature", 702, -1, "self-description", "Feature 702 (−), dose 2.0"),
    ("control", "sycophantic_honest", 1, "disagreement", "Control: sycophantic (+), dose 2.0"),
    ("random", 2, 0, "disagreement", "Random direction 2, dose 1.0"),
]


def cut(text: str, n: int = 620) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def main(status_path: str) -> None:
    s = json.loads((BASE / "summary.json").read_text(encoding="utf-8"))
    st = json.loads(Path(status_path).read_text(encoding="utf-8"))
    scen = {x["id"]: x["prompt"] for x in json.loads((ROOT / "research/experiment-003c/pipeline/scenarios.json").read_text(encoding="utf-8"))}

    feats = []
    for f in s["features"]:
        agg = defaultdict(list)
        for r in f["rows"]:
            agg[(r["sign"], r["dose"])].append(r)
        rows = [{"sign": k[0], "dose": k[1], "editDistance": round(sum(r["edit_distance"] for r in v) / len(v), 3), "coherent": sum(1 for r in v if r["coherent"]), "of": len(v)} for k, v in sorted(agg.items())]
        feats.append({"id": f["feature"], "density": round(f["density"], 5), "maxCoherentDose": f["max_coherent_dose"], "rows": rows})
    rnd: dict = defaultdict(lambda: defaultdict(list))
    for r in s["random_control"]:
        rnd[r["random_index"]][r["dose"]].append(r)
    random = [{"index": i, "rows": [{"sign": 1, "dose": d, "editDistance": round(sum(r["edit_distance"] for r in v) / len(v), 3), "coherent": sum(1 for r in v if r["coherent"]), "of": len(v)} for d, v in sorted(dd.items())]} for i, dd in sorted(rnd.items())]
    screen = [{"kind": r["kind"], "id": r["id"], "sign": r["sign"], "dose": r["dose"], "residAuc": round(r["separability"]["resid"]["auc"], 4), "lexAuc": round(r["separability"]["lexical"]["auc"], 4), "consistency": round(r["consistency"]["mean_cos"], 4), "nullConsistency": round(r["consistency"]["null_mean_cos"], 4), "coherentFraction": r["coherent_fraction"]} for r in s["screen"]]
    v = s["screen_verdict"]
    verdict = {"maxRandomResidAuc": v["max_random_resid_auc"], "maxRandomResidAucP95": v["max_random_resid_auc_p95"], "randomDirections": v["random_directions"], "featureDirectionsPassing": v["feature_directions_passing"], "featureDirectionsTotal": v["feature_directions_total"], "featuresPassing": v["features_passing"], "featuresTotal": v["features_total"], "controlsPassing": v["controls_passing"], "controlsTotal": v["controls_total"], "features": v["features_detail"], "controls": v["controls_detail"]}

    def find(kind, id_, sign, scn):
        return next(g for g in s["screen_generations"] if g["kind"] == kind and g["id"] == id_ and g["sign"] == sign and g["scenario"] == scn)

    ex = []
    for kind, id_, sign, scn, label in EXAMPLES:
        g = find(kind, id_, sign, scn)
        ex.append({"label": label, "kind": kind, "scenario": scn, "prompt": cut(scen[scn], 300), "baseline": cut(g["baseline_text"]), "steered": cut(g["text"]), "residAuc": next(r["residAuc"] for r in screen if r["kind"] == kind and r["id"] == id_ and r["sign"] == sign)})
    pt = s["post_train_check"]
    data = {
        "runId": st["run"]["id"], "commit": "e22c474", "pod": "8n2nscddzdfb8f", "gpu": "RTX A6000 48 GB", "pricePerHour": 0.53, "costUsd": 1.10,
        "startedAt": st["run"]["createdAt"], "endedAt": st["run"]["completedAt"],
        "config": {"model": "Qwen2.5-7B-Instruct", "layer": 19, "width": 8192, "k": 40, "tokens": 8000000, "stepsPerBatch": 4, "features": 8, "doses": [0.25, 0.5, 1.0, 2.0], "maxNewTokens": 256, "randomDirections": 20, "calibrateScenarios": 4, "screenScenarios": 8, "controls": 3},
        "timeline": [{"at": e["at"], "stage": e["stage"], "title": e["title"], "summary": e["summary"]} for e in st["events"]],
        "train": TRAIN,
        "postTrain": {"heldOutFve": round(pt["held_out_fve"], 4), "heldOutL0": pt["held_out_l0"], "matchFraction": round(pt["match_fraction"], 4), "minMatchFraction": pt["min_match_fraction"], "replacedFve": round(pt["replaced_fve"], 4), "ceDelta": round(pt["ce_delta"], 4), "assistantPositions": pt["assistant_positions"]},
        "calibrate": {"doses": [0.25, 0.5, 1.0, 2.0], "features": feats, "random": random},
        "screen": {"rows": screen, "verdict": verdict},
        "examples": ex,
        "describe": {"planned": 96, "results": 0, "note": "The pod client's 30-second read timeout stopped the drive loop while the Worker kept judging; no describe results were produced for this run. Fixed in commit 89d8b18 (300 s, 10 jobs per call)."},
    }
    out = "// Generated from research/experiment-003c/smoke-runs/visual-2 and the harness ledger. Do not edit by hand.\n// Regenerate with research/experiment-003c/scripts/extract_visual2_replay.py.\n\nexport const visual2 = " + json.dumps(data, indent=1, ensure_ascii=False) + " as const;\n\nexport type Visual2 = typeof visual2;\n"
    OUT.write_text(out, encoding="utf-8")
    print(f"wrote {OUT} ({len(out)} bytes)")


if __name__ == "__main__":
    main(sys.argv[1])
