"""Safe scoring entrypoint.

Live scoring must pass through ``autolabs_relay.py`` and the authenticated
AutoLabs worker. This compatibility command intentionally has no provider
client and refuses ``--execute`` so a stale local script cannot bypass the
worker's authentication, shard leases, telemetry, and budget guard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import read_json, read_jsonl, sha256_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="disabled; use autolabs_relay.py")
    args = parser.parse_args()
    if args.execute:
        print("ERROR: direct Luna execution is disabled; use autolabs_relay.py --execute through AutoLabs.", file=sys.stderr)
        return 2
    try:
        manifest = read_json(args.manifest)
        inputs = read_jsonl(args.inputs)
        if len(inputs) != 768:
            raise ValueError(f"expected 768 blinded pair inputs, found {len(inputs)}")
        repeat = int(manifest.get("reliability", {}).get("repeatRows", 117))
        ceiling = int(manifest.get("execution", {}).get("apiCallCeiling", 1014))
        planned = len(inputs) + repeat
        if planned > ceiling:
            raise ValueError(f"planned calls {planned} exceed ceiling {ceiling}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir / "scoring-state.json", {"schemaVersion": 1, "status": "worker-required", "inputs": len(inputs), "repeatRows": repeat, "plannedCalls": planned, "callCeiling": ceiling, "sourceInputsSha256": sha256_json(inputs), "message": "Use autolabs_relay.py for authenticated worker execution."})
        print(f"dry-run worker_required=true inputs={len(inputs)} repeats={repeat} planned_calls={planned}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
