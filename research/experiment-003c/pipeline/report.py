"""Reporting client for the AutoLabs orchestrator Worker.

Every stage of the pipeline calls into this module to POST progress and
records to the Worker. Network failures must never block or crash the GPU
job: every method here logs and swallows errors after retrying, except
`start_run`, whose failure is also swallowed (the caller decides whether a
missing run id is fatal).

Env vars:
    AUTOLABS_3C_WORKER_URL  base URL of the orchestrator Worker
    AUTOLABS_3C_TOKEN       bearer token for the Worker
    AUTOLABS_3C_RUN_ID      run id, if the run was already started elsewhere
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:  # pragma: no cover - requests is a hard requirement on the pod
    requests = None  # type: ignore

logger = logging.getLogger("autolabs_3c.report")

MAX_RECORDS_PER_CALL = 200


def canonical_json(payload: Any) -> str:
    """Canonical JSON per research/experiment-003b/HASHING.md: sorted keys,
    compact separators, UTF-8, no ASCII escaping."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_of_payload(payload: Any) -> str:
    data = canonical_json(payload).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _chunk(items: List[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


class WorkerClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        run_id: Optional[str] = None,
        max_retries: int = 5,
        timeout: float = 30.0,
        session: Any = None,
    ) -> None:
        self.base_url = (base_url if base_url is not None else os.environ.get("AUTOLABS_3C_WORKER_URL", "")).rstrip("/")
        self.token = token if token is not None else os.environ.get("AUTOLABS_3C_TOKEN", "")
        self.run_id = run_id if run_id is not None else os.environ.get("AUTOLABS_3C_RUN_ID", "") or None
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = session

    # -- internal --------------------------------------------------------
    def _get_session(self):
        if self._session is not None:
            return self._session
        if requests is None:
            return None
        if not hasattr(self, "_lazy_session"):
            self._lazy_session = requests.Session()
        return self._lazy_session

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 autolabs-3c-relay"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _post(self, path: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.base_url:
            logger.warning("AUTOLABS_3C_WORKER_URL is not set; skipping report to %s", path)
            return None
        session = self._get_session()
        if session is None:
            logger.warning("requests is unavailable; skipping report to %s", path)
            return None

        url = f"{self.base_url}{path}"
        backoff = 1.0
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = session.post(url, json=body, headers=self._headers(), timeout=self.timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RuntimeError(f"retryable status {resp.status_code}: {resp.text[:300]}")
                if resp.status_code >= 400:
                    logger.error(
                        "worker rejected %s (status %s): %s", path, resp.status_code, resp.text[:500]
                    )
                    return None
                try:
                    return resp.json()
                except ValueError:
                    return {}
            except Exception as exc:  # noqa: BLE001 - network code must never raise
                last_error = exc
                logger.warning(
                    "report POST %s failed (attempt %d/%d): %s", path, attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2
        logger.error("giving up on report POST %s after %d attempts: %s", path, self.max_retries, last_error)
        return None

    def _base_body(
        self,
        stage: str,
        progress: Optional[Dict[str, Any]],
        status: Optional[str],
        gpu_hours: Optional[float],
        message: Optional[str],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"runId": self.run_id, "stage": stage}
        if progress is not None:
            body["progress"] = progress
        if status is not None:
            body["status"] = status
        if gpu_hours is not None:
            body["gpuHours"] = gpu_hours
        if message is not None:
            body["message"] = message
        return body

    # -- public API --------------------------------------------------------
    def start_run(
        self, manifest_hash: Optional[str], budget_usd: Optional[float], idempotency_key: Optional[str]
    ) -> Optional[str]:
        """Start a run on the Worker, unless AUTOLABS_3C_RUN_ID was already set."""
        if self.run_id:
            return self.run_id
        body = {
            "manifestHash": manifest_hash,
            "budgetUsd": budget_usd,
            "idempotencyKey": idempotency_key,
        }
        result = self._post("/api/persona-3c/start", body)
        if result and result.get("runId"):
            self.run_id = result["runId"]
            return self.run_id
        logger.error("start_run did not return a runId; continuing without one")
        return None

    def report(
        self,
        stage: str,
        progress: Optional[Dict[str, Any]] = None,
        records: Optional[List[Dict[str, Any]]] = None,
        status: Optional[str] = None,
        gpu_hours: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        """POST /api/persona-3c/report. Never raises.

        `records` is a list of {recordId, payload} (or {recordId, payload,
        sha256} if the sha was already computed); sha256 is filled in here
        if missing. Records are sent in batches of at most
        MAX_RECORDS_PER_CALL, each batch carrying the same stage/progress/
        status/gpu_hours/message.
        """
        prepared: List[Dict[str, Any]] = []
        for record in records or []:
            payload = record.get("payload")
            sha = record.get("sha256") or sha256_of_payload(payload)
            prepared.append({"recordId": record.get("recordId"), "payload": payload, "sha256": sha})

        if not prepared:
            body = self._base_body(stage, progress, status, gpu_hours, message)
            body["records"] = []
            self._post("/api/persona-3c/report", body)
            return

        batches = list(_chunk(prepared, MAX_RECORDS_PER_CALL))
        for i, batch in enumerate(batches):
            body = self._base_body(stage, progress, status, gpu_hours, message)
            body["records"] = batch
            body["batch"] = {"index": i, "count": len(batches)}
            self._post("/api/persona-3c/report", body)


_default_client: Optional[WorkerClient] = None


def _client() -> WorkerClient:
    global _default_client
    if _default_client is None:
        _default_client = WorkerClient()
    return _default_client


def start_run(manifest_hash: Optional[str], budget_usd: Optional[float], idempotency_key: Optional[str]) -> Optional[str]:
    return _client().start_run(manifest_hash, budget_usd, idempotency_key)


def report(
    stage: str,
    progress: Optional[Dict[str, Any]] = None,
    records: Optional[List[Dict[str, Any]]] = None,
    status: Optional[str] = None,
    gpu_hours: Optional[float] = None,
    message: Optional[str] = None,
) -> None:
    _client().report(
        stage, progress=progress, records=records, status=status, gpu_hours=gpu_hours, message=message
    )
