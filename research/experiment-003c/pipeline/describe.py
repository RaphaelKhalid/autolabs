"""Describe stage (DESCRIBE / judge pass one, see ../VISUAL-1.md "Next").

The screen stage (screen.py) tells us *whether* a direction's steered-vs-
baseline effect is separable and consistent, but not *what* it is: a
persona-like shift, a topic change, or a formatting artifact could all
produce the same separability numbers. This stage answers that by showing
a blinded judge a pair of texts for the same scenario -- one baseline, one
steered, order randomized -- and asking it to describe the difference (or
say there is none) without ever seeing which is which, a persona
vocabulary, or a feature id.

The actual judging happens inside the Worker (`/api/persona-3c/judge/*`,
see `orchestrator-worker/src/persona-3c.ts`), which owns the OpenAI call,
budget reservation, and retry bookkeeping -- this module only builds the
blinded pairs, drives the Worker's queue to completion, and clusters the
resulting descriptions per direction. Everything here except the two
`WorkerClient` calls (`submit_judge_plan`, `drive_judge`, `fetch_results`)
is pure Python/numpy and CPU-testable.
"""
from __future__ import annotations

import html
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import screen
from report import WorkerClient, _progress

logger = logging.getLogger("autolabs_3c.describe")

JUDGE_PLAN_BATCH = 500
CLUSTER_SIMILARITY_THRESHOLD = 0.5
LEXICAL_DIMS = 1024
ABOUT_KEYS = ("speaker", "content", "format")


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------
def _sign_label(sign: Any) -> str:
    if sign == 1:
        return "pos"
    if sign == -1:
        return "neg"
    return "na"


def direction_key(kind: Any, direction_id: Any, sign: Any) -> str:
    return f"{kind}-{direction_id}-{_sign_label(sign)}"


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _resid_auc(row: Dict[str, Any]) -> float:
    auc = (row.get("separability") or {}).get("resid", {}).get("auc")
    return auc if _is_finite_number(auc) else float("-inf")


def _consistency_mean_cos(row: Dict[str, Any]) -> float:
    mean_cos = (row.get("consistency") or {}).get("mean_cos")
    return mean_cos if _is_finite_number(mean_cos) else float("-inf")


def build_judge_pairs(
    screen_generations: Sequence[Dict[str, Any]], directions: Sequence[Dict[str, Any]], top_n: int
) -> List[Dict[str, Any]]:
    """Selects the `top_n` directions by residual-view separability AUC
    (ties broken by consistency `mean_cos`), then for every scenario each
    selected direction was screened on emits two blinded pairs -- one per
    order:

    - `orderSwap=False`: `textA` = baseline, `textB` = steered.
    - `orderSwap=True`:  `textA` = steered,  `textB` = baseline.

    `directions` is `screen_records.json`'s (or `summary["screen"]`'s)
    per-direction payloads (`{kind, id, sign, separability, consistency,
    ...}`); `screen_generations` is `screen_generations.json`'s (or
    `summary["screen_generations"]`'s) flat per-(direction, scenario)
    payloads (`{kind, id, sign, scenario, text, baseline_text, ...}`).
    Returns a flat list of `{directionKey, scenario, textA, textB,
    orderSwap}` dicts ready for `submit_judge_plan`.
    """
    ranked = sorted(directions, key=lambda row: (_resid_auc(row), _consistency_mean_cos(row)), reverse=True)
    selected = ranked[: max(0, top_n)]
    selected_keys = {direction_key(row["kind"], row["id"], row["sign"]) for row in selected}

    pairs: List[Dict[str, Any]] = []
    for gen in screen_generations:
        key = direction_key(gen["kind"], gen["id"], gen["sign"])
        if key not in selected_keys:
            continue
        baseline = gen.get("baseline_text") or ""
        steered = gen.get("text") or ""
        scenario = gen["scenario"]
        pairs.append({"directionKey": key, "scenario": scenario, "textA": baseline, "textB": steered, "orderSwap": False})
        pairs.append({"directionKey": key, "scenario": scenario, "textA": steered, "textB": baseline, "orderSwap": True})
    return pairs


def _chunk(items: List[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Worker driving (network; never raises -- WorkerClient swallows failures)
# ---------------------------------------------------------------------------
def submit_judge_plan(
    client: WorkerClient, run_id: str, pairs: List[Dict[str, Any]], budget: float, ceiling: int
) -> Dict[str, int]:
    """POSTs `pairs` to `/api/persona-3c/judge/plan` in batches of at most
    `JUDGE_PLAN_BATCH`. Returns the summed `{queued, duplicate}` counts
    across every batch (a batch the Worker rejects contributes 0 to both,
    since `WorkerClient._post` already logged the rejection)."""
    queued_total = 0
    duplicate_total = 0
    for batch in _chunk(list(pairs), JUDGE_PLAN_BATCH):
        result = client.judge_plan(run_id, batch, budget, ceiling)
        if result:
            queued_total += int(result.get("queued", 0) or 0)
            duplicate_total += int(result.get("duplicate", 0) or 0)
    return {"queued": queued_total, "duplicate": duplicate_total}


def drive_judge(client: WorkerClient, run_id: str, max_jobs: int = 25, poll_seconds: float = 5.0) -> Dict[str, Any]:
    """Repeatedly calls `/api/persona-3c/judge/run` until the queue is
    drained (`remainingQueued <= 0`) or a request fails outright (network
    error, not a budget/ceiling refusal -- those still return 200 with
    `remainingQueued` unchanged, so a stuck budget will loop here; callers
    should size `judge_budget_usd`/`judge_call_ceiling` to actually cover
    the planned pairs). Returns the last successful response."""
    last: Dict[str, Any] = {"ran": 0, "complete": 0, "failed": 0, "remainingQueued": 0, "judgeSpentUsd": 0.0}
    while True:
        result = client.judge_run(run_id, max_jobs)
        if result is None:
            logger.error("[describe] judge_run request failed; stopping the drive loop")
            return last
        last = result
        if int(result.get("remainingQueued", 0) or 0) <= 0:
            return last
        time.sleep(poll_seconds)


def fetch_results(client: WorkerClient, run_id: str) -> List[Dict[str, Any]]:
    """GET /api/persona-3c/judge/results?runId=...: every complete judge
    job's `{jobId, directionKey, scenario, orderSwap, response, usage,
    costUsd}` (never the blinded texts)."""
    result = client.judge_results(run_id)
    if not result:
        return []
    return list(result.get("results", []) or [])


# ---------------------------------------------------------------------------
# Clustering (pure numpy; CPU-testable)
# ---------------------------------------------------------------------------
def _tfidf_vectors(texts: List[str]) -> np.ndarray:
    """Simple TF-IDF over a small corpus of one-sentence descriptions:
    token counts normalized by document length (TF), scaled by
    log((1+N)/(1+df)) + 1 (smoothed IDF), rows L2-normalized so cosine
    similarity is a plain dot product. No sklearn on the pod image."""
    docs = [screen.tokenize_words(t) for t in texts]
    n_docs = len(docs)
    doc_freq: Dict[str, int] = {}
    for doc in docs:
        for token in set(doc):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    terms = sorted(doc_freq)
    index = {term: i for i, term in enumerate(terms)}
    matrix = np.zeros((n_docs, len(terms)), dtype=np.float64)
    for row, doc in enumerate(docs):
        if not doc:
            continue
        counts: Dict[str, int] = {}
        for token in doc:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            tf = count / len(doc)
            idf = math.log((1 + n_docs) / (1 + doc_freq[token])) + 1.0
            matrix[row, index[token]] = tf * idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return matrix / norms


def _combined_similarity(texts: List[str]) -> np.ndarray:
    """Average of two independent cosine-similarity views -- the hashed
    lexical vector `screen.hashed_lexical_vector` reuses from the screen
    stage, and a fresh TF-IDF over just this corpus of descriptions -- so a
    cluster isn't an artifact of one representation."""
    if not texts:
        return np.zeros((0, 0))
    lexical = np.stack([screen.hashed_lexical_vector(t, dims=LEXICAL_DIMS) for t in texts])
    tfidf = _tfidf_vectors(texts)
    lexical_similarity = lexical @ lexical.T
    tfidf_similarity = tfidf @ tfidf.T
    return 0.5 * lexical_similarity + 0.5 * tfidf_similarity


def _agglomerative_clusters(similarity: np.ndarray, threshold: float = CLUSTER_SIMILARITY_THRESHOLD) -> List[List[int]]:
    """Average-linkage agglomerative clustering on a precomputed cosine
    similarity matrix: repeatedly merges the two clusters with the lowest
    average pairwise distance (1 - similarity) until the best available
    merge would exceed `threshold`, or only one cluster remains. Small-n
    (a handful to a few dozen descriptions per direction) so an O(n^3)-ish
    pure-Python loop is fine; no scipy on the pod image."""
    n = similarity.shape[0]
    clusters: List[List[int]] = [[i] for i in range(n)]
    if n <= 1:
        return clusters
    distance = 1.0 - similarity
    while len(clusters) > 1:
        best_avg: Optional[float] = None
        best_pair: Optional[tuple] = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                idx_a, idx_b = clusters[a], clusters[b]
                avg = float(np.mean([distance[i, j] for i in idx_a for j in idx_b]))
                if best_avg is None or avg < best_avg:
                    best_avg = avg
                    best_pair = (a, b)
        if best_avg is None or best_avg > threshold:
            break
        a, b = best_pair  # type: ignore[misc]
        merged = clusters[a] + clusters[b]
        clusters = [c for i, c in enumerate(clusters) if i not in (a, b)] + [merged]
    return clusters


def cluster_descriptions(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per direction: unblinds each judge response using its local
    `orderSwap` (a response is `steered_is_B` iff `orderSwap` is False --
    the pipeline never rewrites the description text itself, since a
    one-sentence natural-language difference can't be mechanically
    negated), then reports `n_none`, `n_described`, the largest cluster's
    size and centroid sentence (the description with the highest mean
    similarity to the rest of its cluster), `about_counts`, and `named`
    (largest cluster >= 50% of described AND a strict majority of that
    cluster's `about` is "speaker")."""
    by_direction: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        by_direction.setdefault(row["directionKey"], []).append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for key, rows in by_direction.items():
        n_none = 0
        described: List[Dict[str, Any]] = []
        about_counts = {k: 0 for k in ABOUT_KEYS}

        for row in rows:
            response = row.get("response") or {}
            is_none = bool(response.get("none"))
            difference = str(response.get("difference") or "").strip()
            if is_none or not difference:
                n_none += 1
                continue
            about = response.get("about")
            described.append({
                "text": difference,
                "about": about,
                "steered_is_B": not bool(row.get("orderSwap")),
            })
            if about in about_counts:
                about_counts[about] += 1

        n_described = len(described)
        if n_described == 0:
            out[key] = {
                "n_none": n_none, "n_described": 0, "largest_cluster_size": 0,
                "centroid_sentence": None, "about_counts": about_counts, "named": False,
            }
            continue

        texts = [d["text"] for d in described]
        similarity = _combined_similarity(texts)
        clusters = _agglomerative_clusters(similarity)
        largest = max(clusters, key=len)
        largest_size = len(largest)

        cluster_similarity = similarity[np.ix_(largest, largest)]
        mean_similarity_to_cluster = cluster_similarity.mean(axis=1)
        centroid_local_idx = int(np.argmax(mean_similarity_to_cluster))
        centroid_sentence = described[largest[centroid_local_idx]]["text"]

        cluster_about = [described[i]["about"] for i in largest]
        speaker_count = sum(1 for a in cluster_about if a == "speaker")
        named = (largest_size >= 0.5 * n_described) and (speaker_count * 2 > largest_size)

        out[key] = {
            "n_none": n_none,
            "n_described": n_described,
            "largest_cluster_size": largest_size,
            "centroid_sentence": centroid_sentence,
            "about_counts": about_counts,
            "named": named,
        }
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_describe_report(output: Dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value))

    rows = []
    for key, summary in sorted(output.get("clusters", {}).items()):
        about = summary.get("about_counts") or {}
        rows.append(
            "<tr>"
            f"<td>{esc(key)}</td>"
            f"<td>{esc(summary.get('n_none'))}</td>"
            f"<td>{esc(summary.get('n_described'))}</td>"
            f"<td>{esc(summary.get('largest_cluster_size'))}</td>"
            f"<td>{esc(summary.get('centroid_sentence'))}</td>"
            f"<td>speaker {esc(about.get('speaker'))} / content {esc(about.get('content'))} / format {esc(about.get('format'))}</td>"
            f"<td>{'yes' if summary.get('named') else 'no'}</td>"
            "</tr>"
        )
    body = "\n".join(rows) or "<tr><td colspan=\"7\">No describe results.</td></tr>"
    plan = output.get("plan") or {}
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Experiment 3C describe stage</title></head><body>"
        "<h1>Describe stage (judge pass one)</h1>"
        f"<p>Judge plan: {esc(plan.get('queued'))} queued, {esc(plan.get('duplicate'))} duplicate.</p>"
        "<table border=\"1\" cellpadding=\"4\"><thead><tr>"
        "<th>direction</th><th>n_none</th><th>n_described</th><th>largest cluster</th>"
        "<th>centroid sentence</th><th>about</th><th>named</th>"
        "</tr></thead><tbody>" + body + "</tbody></table></body></html>"
    )


# ---------------------------------------------------------------------------
# Orchestration (called from run_smoke.stage_describe)
# ---------------------------------------------------------------------------
def run_describe(
    config: Any,
    workdir: Path,
    client: WorkerClient,
    run_id: str,
    directions: List[Dict[str, Any]],
    screen_generations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Runs the full describe stage: builds blinded pairs for the top
    `config.describe_top_n` directions, plans and drives them through the
    Worker's judge queue, fetches and clusters the results, writes
    `describe_results.json` + `describe-report.html` under `workdir`, and
    reports a compact per-direction summary record to the harness under
    stage `"judge"`. Resumable: skips entirely if `describe_results.json`
    already exists."""
    path = workdir / "describe_results.json"
    if path.exists():
        logger.info("[describe] results already exist, skipping describe stage")
        return json.loads(path.read_text(encoding="utf-8"))

    pairs = build_judge_pairs(screen_generations, directions, config.describe_top_n)
    plan = submit_judge_plan(client, run_id, pairs, config.judge_budget_usd, config.judge_call_ceiling)
    logger.info("[describe] judge plan: %s queued, %s duplicate (%d pairs)", plan.get("queued"), plan.get("duplicate"), len(pairs))
    drive_judge(client, run_id, max_jobs=25, poll_seconds=5)
    results = fetch_results(client, run_id)
    clusters = cluster_descriptions(results)

    output = {"plan": plan, "results": results, "clusters": clusters}
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    (workdir / "describe-report.html").write_text(render_describe_report(output), encoding="utf-8")

    records = [
        {"recordId": f"describe-{key}", "payload": {"directionKey": key, **summary}}
        for key, summary in clusters.items()
    ]
    client.report("judge", progress=_progress(len(records), max(len(records), 1)), records=records)
    return output
