"""Owner-local Experiment 3A relay.

Reads PERSONA_3A_RELAY_TOKEN and KAGGLE_API_TOKEN only from the process environment.
No credentials are written to the repository.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

WORKER_URL = os.environ.get("PERSONA_3A_WORKER_URL", "https://autolabs-orchestrator.raphaelbahadurkhan.workers.dev").rstrip("/")
RELAY_TOKEN = os.environ.get("PERSONA_3A_RELAY_TOKEN", "")
KAGGLE_TOKEN = os.environ.get("KAGGLE_API_TOKEN", "")
STUDY_ID = "experiment-003a-v1"
PHASE = "development"
MAX_RUNTIME = 6600
MANIFEST_HASH = "sha256:9fd05fffe50a6f402fea4dab6f7cf51dcd66ef78ea9fa450374bd0f2411ca766"
KERNEL = "raphaelkhalid0/unsupervisedsaes"
DATASET_STALE = "8049631c405ae6576f93f445c6b81a66f76f5505a"
DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"


def call(path: str, method: str = "GET", payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(
        WORKER_URL + path,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {RELAY_TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def update(launch_id: str, status: str, error: str | None = None) -> None:
    payload = {"id": launch_id, "status": status}
    if error:
        payload["error"] = error[:500]
    result = call("/api/persona-3a/update", "POST", payload)
    print(json.dumps({"launch": launch_id, "status": result.get("status")}), flush=True)


def run_command(args: list[str], cwd: Path, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, env={**os.environ, "KAGGLE_API_TOKEN": KAGGLE_TOKEN}, text=True, capture_output=True, timeout=timeout)


def prepare_kernel() -> Path:
    directory = Path(tempfile.mkdtemp(prefix="autolabs-persona-3a-"))
    result = run_command([sys.executable, "-m", "kaggle", "kernels", "pull", KERNEL, "-p", str(directory), "-m"], directory, 180)
    if result.returncode != 0:
        raise RuntimeError("Kaggle notebook pull failed: " + (result.stderr or result.stdout)[-800:])
    notebook = directory / "unsupervisedsaes.ipynb"
    metadata = directory / "kernel-metadata.json"
    if not notebook.exists() or not metadata.exists():
        raise RuntimeError("Kaggle pull did not produce the expected notebook and metadata.")
    text = notebook.read_text(encoding="utf-8")
    if DATASET_STALE not in text and DATASET_REVISION not in text:
        raise RuntimeError("The expected dataset revision was not found; refusing to alter an unknown notebook.")
    notebook.write_text(text.replace(DATASET_STALE, DATASET_REVISION), encoding="utf-8")
    config = json.loads(metadata.read_text(encoding="utf-8"))
    if config.get("id") != KERNEL or not config.get("is_private") or config.get("machine_shape") != "NvidiaTeslaT4":
        raise RuntimeError("Kaggle metadata is not the expected private T4 notebook.")
    config["enable_gpu"] = True
    config["enable_internet"] = True
    metadata.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return directory


def main() -> int:
    if not RELAY_TOKEN or not KAGGLE_TOKEN:
        raise RuntimeError("PERSONA_3A_RELAY_TOKEN and KAGGLE_API_TOKEN are required in process memory.")
    queued = call("/api/persona-3a/next")
    launch = queued.get("launch")
    if not launch:
        print("No queued Experiment 3A request.", flush=True)
        return 0
    launch_id = launch.get("id", "")
    checks = {
        "studyId": launch.get("studyId") == STUDY_ID,
        "phase": launch.get("phase") == PHASE,
        "maxRuntimeSeconds": int(launch.get("maxRuntimeSeconds", 0)) == MAX_RUNTIME,
        "manifestHash": launch.get("manifestHash") == MANIFEST_HASH,
    }
    if not all(checks.values()):
        update(launch_id, "failed", "Safe hand-off refused: launch contract mismatch: " + json.dumps(checks, sort_keys=True))
        return 1
    print(json.dumps({"launch": launch_id, "validated": True}), flush=True)
    try:
        directory = prepare_kernel()
        result = run_command([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(directory), "-t", str(MAX_RUNTIME), "--accelerator", "NvidiaTeslaT4"], directory, 300)
        if result.returncode != 0:
            raise RuntimeError("Kaggle notebook push failed: " + (result.stderr or result.stdout)[-1000:])
        update(launch_id, "started")
        print((result.stdout or "Kaggle run submitted.")[-1200:], flush=True)
        return 0
    except Exception as exc:
        try:
            update(launch_id, "failed", str(exc))
        finally:
            raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HTTPError, URLError) as exc:
        print(f"Relay request failed: {exc}", file=sys.stderr)
        raise SystemExit(1)