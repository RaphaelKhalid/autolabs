"""Run 3b unblinding analysis and attach order-swapped repeat reliability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analyze_contract import analyze, filter_strict_scores
from common import read_json, read_jsonl, write_json, write_jsonl
from order_flip import order_flip_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        pairs = read_jsonl(args.pairs); scores = read_jsonl(args.scores); mapping = read_json(args.mapping); manifest = read_json(args.manifest)
        report, summaries = analyze(pairs, scores, mapping, manifest)
        strict_scores, _ = filter_strict_scores(scores)
        report["reliability"]["orderFlip"] = order_flip_report(strict_scores, mapping)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir / "analysis.json", report)
        write_jsonl(args.out_dir / "feature-sign-summary.jsonl", summaries)
        print(f"analyzed pairs={report['counts']['pairs']} order_flip_rate={report['reliability']['orderFlip']['orderFlipRate']}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
