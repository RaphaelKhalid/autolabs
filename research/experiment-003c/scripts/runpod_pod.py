"""Minimal RunPod REST v1 helper for Experiment 3C pods (no secrets in code).

Reads RUNPOD_API_KEY from research/experiment-003c/.env.runpod (gitignored).
Usage (from repo root):
  python research/experiment-003c/scripts/runpod_pod.py create --name 3c-smoke --gpu "NVIDIA RTX A6000" [--gpu "NVIDIA A100 80GB PCIe"] [--volume 80]
  python research/experiment-003c/scripts/runpod_pod.py get <podId>
  python research/experiment-003c/scripts/runpod_pod.py ssh <podId>        # prints "ssh root@IP -p PORT" once ready
  python research/experiment-003c/scripts/runpod_pod.py list
  python research/experiment-003c/scripts/runpod_pod.py terminate <podId>

The RunPod API answers 403 (Cloudflare 1010) without a browser-like User-Agent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
ENV = ROOT / "research/experiment-003c/.env.runpod"
BASE = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) autolabs-3c"


def load_env() -> dict:
    out = {}
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def headers() -> dict:
    return {"Authorization": f"Bearer {load_env()['RUNPOD_API_KEY']}", "User-Agent": UA, "Content-Type": "application/json"}


def create(name: str, gpus: list[str], volume_gb: int, disk_gb: int) -> dict:
    pub = (Path.home() / ".ssh" / "autolabs_runpod.pub").read_text(encoding="utf-8").strip()
    body = {
        "name": name,
        "imageName": IMAGE,
        "cloudType": "SECURE",
        "gpuTypeIds": gpus,
        "gpuCount": 1,
        "containerDiskInGb": disk_gb,
        "volumeInGb": volume_gb,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "env": {"PUBLIC_KEY": pub, "PIP_BREAK_SYSTEM_PACKAGES": "1", "HF_HOME": "/workspace/3c/hf"},
    }
    r = requests.post(f"{BASE}/pods", headers=headers(), data=json.dumps(body), timeout=60)
    if r.status_code >= 300:
        raise SystemExit(f"create failed {r.status_code}: {r.text[:500]}")
    return r.json()


def get(pod_id: str) -> dict:
    r = requests.get(f"{BASE}/pods/{pod_id}", headers=headers(), timeout=60)
    r.raise_for_status()
    return r.json()


def list_pods() -> list:
    r = requests.get(f"{BASE}/pods", headers=headers(), timeout=60)
    r.raise_for_status()
    return r.json()


def terminate(pod_id: str) -> None:
    r = requests.delete(f"{BASE}/pods/{pod_id}", headers=headers(), timeout=60)
    if r.status_code >= 300:
        raise SystemExit(f"terminate failed {r.status_code}: {r.text[:300]}")


def ssh_target(pod: dict) -> str | None:
    ip = pod.get("publicIp")
    ports = pod.get("portMappings") or {}
    port = ports.get("22") if isinstance(ports, dict) else None
    if ip and port:
        return f"ssh -i ~/.ssh/autolabs_runpod -o StrictHostKeyChecking=no root@{ip} -p {port}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--gpu", action="append", required=True)
    c.add_argument("--volume", type=int, default=80)
    c.add_argument("--disk", type=int, default=40)
    g = sub.add_parser("get"); g.add_argument("pod_id")
    s = sub.add_parser("ssh"); s.add_argument("pod_id"); s.add_argument("--wait", type=int, default=600)
    sub.add_parser("list")
    t = sub.add_parser("terminate"); t.add_argument("pod_id")
    a = ap.parse_args()
    if a.cmd == "create":
        pod = create(a.name, a.gpu, a.volume, a.disk)
        print(json.dumps({k: pod.get(k) for k in ("id", "name", "desiredStatus", "machineId", "costPerHr", "gpu")}))
    elif a.cmd == "get":
        pod = get(a.pod_id)
        print(json.dumps({k: pod.get(k) for k in ("id", "desiredStatus", "publicIp", "portMappings", "costPerHr", "lastStatusChange", "machine")}, default=str))
    elif a.cmd == "ssh":
        deadline = time.time() + a.wait
        while time.time() < deadline:
            target = ssh_target(get(a.pod_id))
            if target:
                print(target)
                return
            time.sleep(10)
        raise SystemExit("pod has no public ip/port yet")
    elif a.cmd == "list":
        for pod in list_pods():
            print(json.dumps({k: pod.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "publicIp")}))
    elif a.cmd == "terminate":
        terminate(a.pod_id)
        print("terminated", a.pod_id)


if __name__ == "__main__":
    main()
