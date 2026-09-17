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

The judge answers `{"property": <=200 chars or "", "more_in": "A"|"B"|
"neither", "about": "speaker"|"content"|"format", "confidence": "low"|
"medium"|"high"}` for each blinded pair. The first live judge pass (48
pairs) found two problems this module now corrects for: (1) the judge
never returned `neither`, so a `none_rate`-only signal cannot separate a
real direction from a random one under greedy decoding -- see
`describe_null_directions`/`describe_null_margin` in config.py and
`compute_null_stats`/`apply_named_above_null` below; and (2) about half
of every direction's pairs are order-swapped, so clustering on raw
`property` text without unblinding the direction first collapses
"steered sounds warmer" and "steered sounds colder" (same property,
opposite direction) into one cluster -- see `_steered_has_more` and
`direction_agreement` below.
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
# Average-linkage merge stops once the best available merge's average
# pairwise distance (1 - cosine similarity) exceeds this. The first live
# judge pass showed paraphrase clustering across swapped orders failing
# badly (largest cluster 1-2 of 8) at the old threshold of 0.5, so this is
# loosened to 0.35 and backed by a fallback keyword-overlap merge
# (`_keyword_overlap_merge`) for short descriptions the cosine view under-
# or over-weights.
CLUSTER_LINKAGE_THRESHOLD = 0.35
# Fallback pass after agglomerative clustering: merge any two remaining
# clusters whose content-word (post stopword-strip) vocabularies overlap
# at least this much (Jaccard), independent of the TF-IDF/hashed-lexical
# cosine view.
FALLBACK_KEYWORD_OVERLAP_THRESHOLD = 0.6
LEXICAL_DIMS = 1024
ABOUT_KEYS = ("speaker", "content", "format")
CONFIDENCE_KEYS = ("low", "medium", "high")
# direction_key() puts `kind` first (e.g. "random-3-na"), and kind values
# never contain a hyphen themselves, so this prefix reliably identifies
# the random-direction nulls among a describe run's directions.
NULL_KEY_PREFIX = "random-"

# Generic function words stripped before TF-IDF/keyword-overlap clustering
# (the judge's own about/more_in fields already carry direction and
# category, so quantifiers like "more"/"less" are stripped as function
# words here too rather than kept as content signal).
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "as", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "into", "about", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "they", "them", "their",
    "he", "she", "his", "her", "you", "your", "we", "our", "us", "i", "my", "not", "no", "do", "does",
    "did", "has", "have", "had", "more", "most", "less", "least", "much", "many", "very", "also",
    "there", "here", "which", "who", "whom", "what", "when", "where", "how", "can", "could", "would",
    "should", "will", "shall", "may", "might", "just", "only", "such", "some", "any", "all", "both",
    "each", "other", "same",
})


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
    screen_generations: Sequence[Dict[str, Any]],
    directions: Sequence[Dict[str, Any]],
    top_n: int,
    null_directions: int = 0,
) -> List[Dict[str, Any]]:
    """Selects the `top_n` directions by residual-view separability AUC
    (ties broken by consistency `mean_cos`), *always* also selects the
    `null_directions` lowest-resid-AUC `kind="random"` directions from the
    screen (config.describe_null_directions; a random direction that
    scores well enough to land in top_n on its own is included either
    way -- selection is a set union, so it's never judged twice), then for
    every scenario each selected direction was screened on emits two
    blinded pairs -- one per order:

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

    random_candidates = sorted(
        (row for row in directions if row.get("kind") == "random"),
        key=lambda row: (_resid_auc(row), _consistency_mean_cos(row)),
    )
    null_selected = random_candidates[: max(0, null_directions)]
    selected_keys |= {direction_key(row["kind"], row["id"], row["sign"]) for row in null_selected}

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


def drive_judge(client: WorkerClient, run_id: str, max_jobs: int = 10, poll_seconds: float = 5.0) -> Dict[str, Any]:
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
def _content_tokens(text: str) -> List[str]:
    """Lowercased word tokens (`screen.tokenize_words`) with `STOPWORDS`
    removed."""
    return [t for t in screen.tokenize_words(text) if t not in STOPWORDS]


def _ngram_doc(text: str) -> List[str]:
    """Unigrams + bigrams over the stopword-stripped content tokens (the
    bigrams span whatever content words end up adjacent after stopwords
    are dropped, not necessarily adjacent in the original sentence)."""
    tokens = _content_tokens(text)
    grams = list(tokens)
    grams += [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return grams


def _tfidf_vectors(texts: List[str]) -> np.ndarray:
    """Simple TF-IDF over a small corpus of one-sentence descriptions:
    token counts normalized by document length (TF), scaled by
    log((1+N)/(1+df)) + 1 (smoothed IDF), rows L2-normalized so cosine
    similarity is a plain dot product. No sklearn on the pod image.
    Documents are unigrams+bigrams over lowercased, stopword-stripped
    tokens (`_ngram_doc`), not raw words, so common connective words don't
    dominate the vocabulary of these short descriptions."""
    docs = [_ngram_doc(t) for t in texts]
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


def _agglomerative_clusters(similarity: np.ndarray, threshold: float = CLUSTER_LINKAGE_THRESHOLD) -> List[List[int]]:
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


def _keyword_overlap_merge(
    clusters: List[List[int]], texts: List[str], threshold: float = FALLBACK_KEYWORD_OVERLAP_THRESHOLD
) -> List[List[int]]:
    """Fallback merge pass after `_agglomerative_clusters`: merges any two
    remaining clusters whose union of content-word (post stopword-strip)
    vocabularies overlaps at least `threshold` (Jaccard), independent of
    the TF-IDF/hashed-lexical cosine view. Short one-sentence descriptions
    can swing the cosine view a lot on a single extra or missing word;
    this catches paraphrases that view still separates."""
    if len(clusters) <= 1:
        return clusters
    keyword_sets = [set(_content_tokens(t)) for t in texts]

    def vocab_of(cluster: List[int]) -> set:
        vocab: set = set()
        for i in cluster:
            vocab |= keyword_sets[i]
        return vocab

    merged_any = True
    while merged_any and len(clusters) > 1:
        merged_any = False
        vocabs = [vocab_of(c) for c in clusters]
        best_overlap: Optional[float] = None
        best_pair: Optional[tuple] = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                va, vb = vocabs[a], vocabs[b]
                if not va or not vb:
                    continue
                overlap = len(va & vb) / len(va | vb)
                if overlap >= threshold and (best_overlap is None or overlap > best_overlap):
                    best_overlap = overlap
                    best_pair = (a, b)
        if best_pair is not None:
            a, b = best_pair
            clusters = [c for i, c in enumerate(clusters) if i not in (a, b)] + [clusters[a] + clusters[b]]
            merged_any = True
    return clusters


def is_null_key(direction_key_value: str) -> bool:
    return direction_key_value.startswith(NULL_KEY_PREFIX)


def _steered_has_more(more_in: Any, order_swap: bool) -> Optional[bool]:
    """Unblinds a judge verdict using the pair's local `orderSwap`. With
    `orderSwap=False`, `textB` is the steered text, so `more_in=="B"`
    already means "the steered response shows more of this property";
    with `orderSwap=True`, `textA` is steered and the sense flips.
    `more_in=="neither"` (or anything not "A"/"B") has no direction."""
    if more_in not in ("A", "B"):
        return None
    return (more_in == "B") != bool(order_swap)


def cluster_descriptions(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per direction: unblinds each judge response's `more_in` with its
    local `orderSwap` (`_steered_has_more`; a response of `more_in ==
    "neither"` -- no meaningful difference -- counts toward `n_none`
    regardless of whether `property` is also empty), clusters the
    remaining `property` text (direction-free: clustering never sees
    `more_in`/`steered_has_more`, only the property description itself),
    and reports per direction:

    - `n_none`, `n_described`, `none_rate` (`n_none / (n_none +
      n_described)`, `0.0` if both are zero).
    - `largest_cluster_size` and `cluster_property` (the described
      property with the highest mean similarity to the rest of the
      largest cluster -- its centroid).
    - `direction_agreement`: within the largest cluster, the fraction of
      members whose unblinded `steered_has_more` matches that cluster's
      majority direction (`None` if there's nothing described).
    - `about_counts`, `confidence_counts`.
    - `named`: largest cluster >= 50% of described, `direction_agreement`
      >= 0.8, and a strict majority of the cluster's `about` is
      "speaker".
    - `consistency_score` = `largest_cluster_size / n_described` *
      `direction_agreement` (`0.0` if nothing described) -- the primary
      per-direction statistic; see `apply_named_above_null`. The first
      live judge pass found `none_rate` couldn't separate a real
      direction from a random null (greedy decoding makes every steered
      text differ from its baseline in *some* way), so a direction only
      counts as `named_above_null` by clearing what the random-direction
      nulls themselves achieve on `consistency_score`, not by an absolute
      `none_rate`/`named` threshold alone.
    """
    by_direction: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        by_direction.setdefault(row["directionKey"], []).append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for key, rows in by_direction.items():
        n_none = 0
        described: List[Dict[str, Any]] = []
        about_counts = {k: 0 for k in ABOUT_KEYS}
        confidence_counts = {k: 0 for k in CONFIDENCE_KEYS}

        for row in rows:
            response = row.get("response") or {}
            more_in = response.get("more_in")
            property_text = str(response.get("property") or "").strip()
            if more_in == "neither" or not property_text:
                n_none += 1
                continue
            about = response.get("about")
            confidence = response.get("confidence")
            described.append({
                "text": property_text,
                "about": about,
                "steered_has_more": _steered_has_more(more_in, bool(row.get("orderSwap"))),
            })
            if about in about_counts:
                about_counts[about] += 1
            if confidence in confidence_counts:
                confidence_counts[confidence] += 1

        n_described = len(described)
        total = n_none + n_described
        none_rate = (n_none / total) if total else 0.0

        if n_described == 0:
            out[key] = {
                "n_none": n_none, "n_described": 0, "none_rate": none_rate,
                "largest_cluster_size": 0, "cluster_property": None,
                "direction_agreement": None, "about_counts": about_counts,
                "confidence_counts": confidence_counts, "named": False,
                "consistency_score": 0.0,
            }
            continue

        texts = [d["text"] for d in described]
        similarity = _combined_similarity(texts)
        clusters = _agglomerative_clusters(similarity)
        clusters = _keyword_overlap_merge(clusters, texts)
        largest = max(clusters, key=len)
        largest_size = len(largest)

        cluster_similarity = similarity[np.ix_(largest, largest)]
        mean_similarity_to_cluster = cluster_similarity.mean(axis=1)
        centroid_local_idx = int(np.argmax(mean_similarity_to_cluster))
        cluster_property = described[largest[centroid_local_idx]]["text"]

        cluster_about = [described[i]["about"] for i in largest]
        speaker_count = sum(1 for a in cluster_about if a == "speaker")

        cluster_directions = [described[i]["steered_has_more"] for i in largest]
        true_count = sum(1 for d in cluster_directions if d is True)
        false_count = sum(1 for d in cluster_directions if d is False)
        direction_agreement = max(true_count, false_count) / largest_size

        largest_cluster_fraction = largest_size / n_described
        named = (
            largest_cluster_fraction >= 0.5
            and direction_agreement >= 0.8
            and speaker_count * 2 > largest_size
        )
        consistency_score = largest_cluster_fraction * direction_agreement

        out[key] = {
            "n_none": n_none,
            "n_described": n_described,
            "none_rate": none_rate,
            "largest_cluster_size": largest_size,
            "cluster_property": cluster_property,
            "direction_agreement": direction_agreement,
            "about_counts": about_counts,
            "confidence_counts": confidence_counts,
            "named": named,
            "consistency_score": consistency_score,
        }
    return out


def attach_arm(clusters: Dict[str, Dict[str, Any]], directions: Sequence[Dict[str, Any]]) -> None:
    """In place: adds `arm` to every direction's cluster summary, looked up
    by `direction_key` from `directions` (screen_records payloads, which
    carry `arm` for every kind: `"unsupervised"`/`"quantile"`/`"shift"`
    for a feature, `"control"` for a positive control, `"random"` for a
    null -- see rank.py/screen.py/steer.py). `None` for a key with no
    matching direction (shouldn't happen in practice, since every judged
    direction came from `directions` in the first place)."""
    arm_by_key = {direction_key(row["kind"], row["id"], row["sign"]): row.get("arm") for row in directions}
    for key, summary in clusters.items():
        summary["arm"] = arm_by_key.get(key)


def compute_arm_named_counts(clusters: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Per-arm (excluding `kind="random"` nulls, see `is_null_key`) named /
    named_above_null / total counts, for the describe report's per-arm
    breakdown -- requires `attach_arm` to have run first (reads each
    summary's `arm` field, defaulting to `"unknown"` if missing)."""
    counts: Dict[str, Dict[str, int]] = {}
    for key, summary in clusters.items():
        if is_null_key(key):
            continue
        arm = summary.get("arm") or "unknown"
        entry = counts.setdefault(arm, {"total": 0, "named": 0, "named_above_null": 0})
        entry["total"] += 1
        if summary.get("named"):
            entry["named"] += 1
        if summary.get("named_above_null"):
            entry["named_above_null"] += 1
    return counts


def compute_null_stats(clusters: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates the `kind="random"` null directions' (see `is_null_key`)
    `none_rate`/`consistency_score` across a describe run: `null_none_rate`
    and `null_consistency_mean` are means, `null_consistency_max` is the
    single highest `consistency_score` any null achieved (the ceiling
    `apply_named_above_null` gates real directions against), and
    `null_named` counts how many nulls came out `named` on their own."""
    null_entries = [v for k, v in clusters.items() if is_null_key(k)]
    none_rates = [e["none_rate"] for e in null_entries]
    consistency_scores = [e["consistency_score"] for e in null_entries]
    return {
        "n_null_directions": len(null_entries),
        "null_none_rate": float(np.mean(none_rates)) if none_rates else 0.0,
        "null_named": sum(1 for e in null_entries if e["named"]),
        "null_consistency_max": float(max(consistency_scores)) if consistency_scores else 0.0,
        "null_consistency_mean": float(np.mean(consistency_scores)) if consistency_scores else 0.0,
    }


def apply_named_above_null(clusters: Dict[str, Dict[str, Any]], null_margin: float) -> Dict[str, Any]:
    """Computes `compute_null_stats(clusters)` and, in place, adds
    `named_above_null` to every direction's cluster summary: `named` AND
    `consistency_score` clears `null_consistency_max + null_margin`
    (config.describe_null_margin). Gating on `consistency_score` against
    the null ceiling -- rather than `none_rate` against an absolute bar --
    is deliberate: the first live judge pass found the judge never
    returned `neither` at all, for real directions or random nulls alike,
    so `none_rate` could not discriminate between them."""
    null_stats = compute_null_stats(clusters)
    threshold = null_stats["null_consistency_max"] + null_margin
    null_stats["named_above_null_threshold"] = threshold
    for entry in clusters.values():
        entry["named_above_null"] = bool(entry["named"] and entry["consistency_score"] > threshold)
    return null_stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_describe_report(output: Dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value))

    def fmt(value: Any) -> Any:
        return None if value is None else round(value, 3)

    header_cols = (
        "<th>direction</th><th>arm</th><th>n_none</th><th>n_described</th><th>none_rate</th>"
        "<th>largest cluster</th><th>cluster property</th><th>direction agreement</th>"
        "<th>about</th><th>confidence</th><th>named</th><th>consistency score</th>"
    )

    def row(key: str, summary: Dict[str, Any], show_named_above_null: bool) -> str:
        about = summary.get("about_counts") or {}
        confidence = summary.get("confidence_counts") or {}
        cells = (
            "<tr>"
            f"<td>{esc(key)}</td>"
            f"<td>{esc(summary.get('arm'))}</td>"
            f"<td>{esc(summary.get('n_none'))}</td>"
            f"<td>{esc(summary.get('n_described'))}</td>"
            f"<td>{esc(fmt(summary.get('none_rate')))}</td>"
            f"<td>{esc(summary.get('largest_cluster_size'))}</td>"
            f"<td>{esc(summary.get('cluster_property'))}</td>"
            f"<td>{esc(fmt(summary.get('direction_agreement')))}</td>"
            f"<td>speaker {esc(about.get('speaker'))} / content {esc(about.get('content'))} / format {esc(about.get('format'))}</td>"
            f"<td>low {esc(confidence.get('low'))} / medium {esc(confidence.get('medium'))} / high {esc(confidence.get('high'))}</td>"
            f"<td>{'yes' if summary.get('named') else 'no'}</td>"
            f"<td>{esc(fmt(summary.get('consistency_score')))}</td>"
        )
        if show_named_above_null:
            cells += f"<td>{'yes' if summary.get('named_above_null') else 'no'}</td>"
        return cells + "</tr>"

    clusters = output.get("clusters", {})
    direction_rows = [row(k, v, True) for k, v in sorted(clusters.items()) if not is_null_key(k)]
    null_rows = [row(k, v, False) for k, v in sorted(clusters.items()) if is_null_key(k)]
    direction_body = "\n".join(direction_rows) or "<tr><td colspan=\"13\">No describe results.</td></tr>"
    null_body = "\n".join(null_rows) or "<tr><td colspan=\"12\">No null directions judged.</td></tr>"

    plan = output.get("plan") or {}
    null_stats = output.get("null_stats") or {}
    arm_named_counts = output.get("arm_named_counts") or {}
    arm_named_line = ", ".join(
        f"{esc(arm)} named {esc(counts.get('named'))} ({esc(counts.get('named_above_null'))} above null) of "
        f"{esc(counts.get('total'))}"
        for arm, counts in sorted(arm_named_counts.items())
    ) or "no directions judged"
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Experiment 3C describe stage</title></head><body>"
        "<h1>Describe stage (judge pass one)</h1>"
        f"<p>Judge plan: {esc(plan.get('queued'))} queued, {esc(plan.get('duplicate'))} duplicate.</p>"
        f"<p>Named counts by arm: {arm_named_line}.</p>"
        "<h2>Directions</h2>"
        "<table border=\"1\" cellpadding=\"4\"><thead><tr>" + header_cols + "<th>named_above_null</th>"
        "</tr></thead><tbody>" + direction_body + "</tbody></table>"
        "<h2>Null (random-direction) block</h2>"
        f"<p>null_none_rate {esc(fmt(null_stats.get('null_none_rate')))}, "
        f"null_named {esc(null_stats.get('null_named'))} of {esc(null_stats.get('n_null_directions'))}, "
        f"null_consistency_max {esc(fmt(null_stats.get('null_consistency_max')))}, "
        f"null_consistency_mean {esc(fmt(null_stats.get('null_consistency_mean')))}, "
        f"named_above_null_threshold {esc(fmt(null_stats.get('named_above_null_threshold')))}.</p>"
        "<table border=\"1\" cellpadding=\"4\"><thead><tr>" + header_cols +
        "</tr></thead><tbody>" + null_body + "</tbody></table></body></html>"
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
    `config.describe_top_n` directions plus `config.describe_null_directions`
    random-direction nulls, plans and drives them through the Worker's
    judge queue, fetches and clusters the results, gates each direction
    against the null ceiling (`apply_named_above_null`), writes
    `describe_results.json` (`{plan, results, clusters, null_stats,
    arm_named_counts}`) +
    `describe-report.html` under `workdir`, and reports a compact
    per-direction summary record to the harness under stage `"judge"`.
    Resumable: skips entirely if `describe_results.json` already exists."""
    path = workdir / "describe_results.json"
    if path.exists():
        logger.info("[describe] results already exist, skipping describe stage")
        return json.loads(path.read_text(encoding="utf-8"))

    pairs = build_judge_pairs(screen_generations, directions, config.describe_top_n, config.describe_null_directions)
    plan = submit_judge_plan(client, run_id, pairs, config.judge_budget_usd, config.judge_call_ceiling)
    logger.info("[describe] judge plan: %s queued, %s duplicate (%d pairs)", plan.get("queued"), plan.get("duplicate"), len(pairs))
    drive_judge(client, run_id, max_jobs=10, poll_seconds=5)
    results = fetch_results(client, run_id)
    clusters = cluster_descriptions(results)
    null_stats = apply_named_above_null(clusters, config.describe_null_margin)
    attach_arm(clusters, directions)
    arm_named_counts = compute_arm_named_counts(clusters)

    output = {
        "plan": plan,
        "results": results,
        "clusters": clusters,
        "null_stats": null_stats,
        "arm_named_counts": arm_named_counts,
    }
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    (workdir / "describe-report.html").write_text(render_describe_report(output), encoding="utf-8")

    records = [
        {"recordId": f"describe-{key}", "payload": {"directionKey": key, **summary}}
        for key, summary in clusters.items()
    ]
    client.report("judge", progress=_progress(len(records), max(len(records), 1)), records=records)
    return output
