"""Screen stage: separate real feature effects from random directions.

Smoke-2 showed that edit distance and the coherence proxy cannot tell a
real SAE feature from a matched-norm random direction (both are equally
coherent and produce similar edit distances at the same dose -- see
SMOKE-2.md "Steering"). Only *consistency* of the steering effect across
scenarios, and whether steered-vs-baseline is *separable* on held-out
scenarios, can. This module computes both, for three kinds of "direction":

- ``kind="feature"``: a trained SAE feature's decoder direction (the same
  ones swept in ``steer.run_calibration``), at its own max-coherent dose
  from that calibration sweep.
- ``kind="control"``: a positive-control persona vector (Chen et al. /
  Arditi et al. / Sleight et al. construction -- the mean layer residual
  over generated tokens under a trait-inducing system prompt minus the
  same under the opposite-trait system prompt), which *should* separate
  and be consistent if the screen's methodology is sound at all.
- ``kind="random"``: a random unit direction, the null this experiment is
  actually testing features and controls against.

Two independent embedding views are used so a real effect isn't an
artifact of one representation: the model's own layer-19 residual, mean-
pooled over generated tokens (``capture_mean_residual``), and a hashed
bag-of-words lexical vector (``hashed_lexical_vector``) that shares no
machinery with the model at all.

Everything that touches the model (``capture_mean_residual``,
``build_persona_vector``, ``sweep_control_doses``, ``generate_steered_set``,
``generate_baseline_set``, ``run_screen``) needs a real transformers model
on a GPU and is not exercised by the CPU test suite, mirroring steer.py.
Everything else here (the hashed lexical vector, the leave-one-scenario-out
logistic regression + AUC, direction consistency vs. a null, and the dose-
bookkeeping helpers) is pure numpy/Python and is unit-tested.

**Batching.** Every generation loop here goes through
``generation.generate_batch`` instead of one ``model.generate`` per
scenario: baselines are one batch across ``screen_scenarios``, a control's
mini dose sweep batches every (sign, dose, scenario) triple, and a
direction's steered set batches across its scenarios. Residual-stream
capture (``capture_mean_residual``) is likewise batched
(``capture_mean_residual_batch``/``capture_generated_hidden_batch``),
mean-pooling each row over only its own real generated-token positions.
"""
from __future__ import annotations

import hashlib
import logging
import re
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import generation
import steer

logger = logging.getLogger("autolabs_3c.screen")

LEXICAL_DIMS = 4096
# Number of fresh random-direction nulls the screen stage draws is
# `config.random_directions` (smoke and full both default to 20 -- see
# config.py). This is independent of steer.run_calibration's own 2
# random-direction controls used for its dose sweep.

LOGREG_L2 = 1e-2
LOGREG_LR = 0.5
LOGREG_EPOCHS = 300


# ---------------------------------------------------------------------------
# Hashed lexical embedding (CPU-testable: deterministic, no model needed)
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z0-9']+")


def tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _hash_token(token: str, dims: int) -> Tuple[int, float]:
    # md5, not Python's built-in hash(), which is randomized per-process
    # (PYTHONHASHSEED) unless explicitly disabled -- this must be
    # deterministic across runs and resumes.
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % dims
    sign = 1.0 if int(digest[8], 16) % 2 == 0 else -1.0
    return idx, sign


def hashed_lexical_vector(text: str, dims: int = LEXICAL_DIMS) -> np.ndarray:
    """Hashed bag of word unigrams + bigrams (the "hashing trick", with a
    hashed sign per bucket to de-bias collisions), L2-normalized. An
    independent view of a generation from the model's own residual stream,
    so separability/consistency agreeing across both views is stronger
    evidence than either alone."""
    words = tokenize_words(text)
    vec = np.zeros(dims, dtype=np.float64)
    for w in words:
        idx, sign = _hash_token(w, dims)
        vec[idx] += sign
    for i in range(len(words) - 1):
        idx, sign = _hash_token(words[i] + "_" + words[i + 1], dims)
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec


# ---------------------------------------------------------------------------
# Logistic regression (numpy only, no sklearn) + leave-one-scenario-out AUC
# ---------------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def train_logreg(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = LOGREG_L2,
    lr: float = LOGREG_LR,
    epochs: int = LOGREG_EPOCHS,
) -> Tuple[np.ndarray, float]:
    """Full-batch gradient descent, L2-regularized on the weights only (not
    the bias). No sklearn dependency (not part of the pod image)."""
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    y = y.astype(np.float64)
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        err = p - y
        grad_w = (X.T @ err) / n + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney-U AUC (rank-based, average-rank tie handling). NaN if
    either class is absent (undefined)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[i : j + 1] = ranks[i : j + 1].mean()
        i = j + 1
    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks

    rank_sum_pos = full_ranks[y_true == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def leave_one_scenario_out_separability(
    steered_by_scenario: Dict[str, np.ndarray],
    baseline_by_scenario: Dict[str, np.ndarray],
    l2: float = LOGREG_L2,
    lr: float = LOGREG_LR,
    epochs: int = LOGREG_EPOCHS,
) -> Dict[str, float]:
    """Leave-one-scenario-out logistic regression classifying steered vs.
    baseline embeddings: for each held-out scenario, train on every other
    scenario's steered+baseline pair (features standardized on the training
    fold), then score the held-out scenario's steered and baseline
    embedding. Aggregates accuracy and AUC across folds. Needs >=2
    scenarios (one to hold out, >=1 left to train on)."""
    scenarios = sorted(set(steered_by_scenario) & set(baseline_by_scenario))
    if len(scenarios) < 2:
        return {"acc": float("nan"), "auc": float("nan")}

    correct = 0
    total = 0
    all_labels: List[int] = []
    all_scores: List[float] = []

    for held_out in scenarios:
        train_scenarios = [s for s in scenarios if s != held_out]
        X_train = np.stack(
            [steered_by_scenario[s] for s in train_scenarios]
            + [baseline_by_scenario[s] for s in train_scenarios]
        )
        y_train = np.array([1] * len(train_scenarios) + [0] * len(train_scenarios))
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0) + 1e-8
        w, b = train_logreg((X_train - mu) / sigma, y_train, l2=l2, lr=lr, epochs=epochs)

        for label, emb in ((1, steered_by_scenario[held_out]), (0, baseline_by_scenario[held_out])):
            x_n = ((emb - mu) / sigma).reshape(1, -1)
            score = float(predict_proba(x_n, w, b)[0])
            pred = 1 if score >= 0.5 else 0
            correct += int(pred == label)
            total += 1
            all_labels.append(label)
            all_scores.append(score)

    acc = correct / total if total else float("nan")
    auc = auc_score(np.array(all_labels), np.array(all_scores))
    return {"acc": acc, "auc": auc}


# ---------------------------------------------------------------------------
# Direction consistency (cosine similarity of per-scenario diff vectors)
# ---------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mean_pairwise_cosine(vectors: Sequence[np.ndarray]) -> float:
    n = len(vectors)
    if n < 2:
        return float("nan")
    sims = [cosine_similarity(vectors[i], vectors[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(sims))


def null_pairwise_cosine(
    own_vectors: Sequence[np.ndarray],
    other_vectors: Sequence[np.ndarray],
    rng: np.random.Generator,
    n_samples: Optional[int] = None,
) -> float:
    """Null consistency: cosine similarity between this direction's diff
    vectors and diff vectors drawn from OTHER directions (a different
    feature/control/random each draw), rather than from itself. This is
    the counterfactual smoke-2 called for: a real feature's per-scenario
    effect should point the same way across scenarios more than it agrees
    with an unrelated direction's effect."""
    if not own_vectors or not other_vectors:
        return float("nan")
    if n_samples is None:
        n_samples = max(len(own_vectors) * 4, 20)
    own_idx = rng.integers(0, len(own_vectors), size=n_samples)
    other_idx = rng.integers(0, len(other_vectors), size=n_samples)
    sims = [cosine_similarity(own_vectors[i], other_vectors[j]) for i, j in zip(own_idx, other_idx)]
    return float(np.mean(sims))


def direction_consistency(
    own_diffs: Sequence[np.ndarray],
    other_diffs_pool: Sequence[np.ndarray],
    rng: np.random.Generator,
    n_null_samples: Optional[int] = None,
) -> Dict[str, float]:
    return {
        "mean_cos": mean_pairwise_cosine(own_diffs),
        "null_mean_cos": null_pairwise_cosine(own_diffs, other_diffs_pool, rng, n_null_samples),
    }


# ---------------------------------------------------------------------------
# Dose bookkeeping (pure Python/dict math, CPU-testable)
# ---------------------------------------------------------------------------
def extract_feature_info(calibration_records: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """{feature: {density, quantile, max_activation, arm}}, deduped, from
    the calibrate stage's "steered" records (tolerates either a raw
    payload dict or a {"payload": ...} wrapper, so tests can pass either).
    `arm` is the rank stage's `"unsupervised"`/`"quantile"`/`"shift"` tag
    (or `None` if calibrate ran without a rank-stage candidate list),
    propagated from `steer.run_calibration`'s steered payload."""
    info: Dict[int, Dict[str, Any]] = {}
    for rec in calibration_records:
        payload = rec.get("payload", rec) if isinstance(rec, dict) else rec
        if payload.get("kind") != "steered":
            continue
        f_idx = payload["feature"]
        info.setdefault(
            f_idx,
            {
                "density": payload.get("density"),
                "quantile": payload.get("quantile"),
                "max_activation": payload.get("max_activation"),
                "arm": payload.get("arm"),
            },
        )
    return info


def feature_max_coherent_doses(
    calibration_records: Sequence[Dict[str, Any]]
) -> Dict[int, Dict[str, Optional[float]]]:
    """{feature: {"pos": dose|None, "neg": dose|None}}, computed from the
    calibrate stage's own dose sweep via ``steer.max_coherent_dose``."""
    by_key: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for rec in calibration_records:
        payload = rec.get("payload", rec) if isinstance(rec, dict) else rec
        if payload.get("kind") != "steered":
            continue
        key = (payload["feature"], payload["sign"])
        by_key.setdefault(key, []).append(payload)

    out: Dict[int, Dict[str, Optional[float]]] = {}
    for (f_idx, sign), rows in by_key.items():
        out.setdefault(f_idx, {"pos": None, "neg": None})
        out[f_idx]["pos" if sign > 0 else "neg"] = steer.max_coherent_dose(rows)
    return out


def resolve_dose(value: Optional[float], fallback: float) -> float:
    return value if value is not None else fallback


def flatten_doses(doses_by_feature: Dict[int, Dict[str, Optional[float]]]) -> List[Optional[float]]:
    flat: List[Optional[float]] = []
    for entry in doses_by_feature.values():
        flat.append(entry.get("pos"))
        flat.append(entry.get("neg"))
    return flat


def median_dose_with_fallback(values: Sequence[Optional[float]], fallback: float) -> float:
    """Median of a list of (possibly-missing) doses, substituting
    `fallback` for any `None` before taking the median, and returning
    `fallback` itself if the list is empty (e.g. zero features selected).
    Used to pick the single dose random-direction nulls steer at (config:
    "random controls use the median of feature max-coherent doses")."""
    resolved = [v if v is not None else fallback for v in values]
    if not resolved:
        return fallback
    return float(statistics.median(resolved))


# ---------------------------------------------------------------------------
# Model-touching helpers (need a real transformers model + GPU)
# ---------------------------------------------------------------------------
def capture_generated_hidden_batch(
    model: Any,
    layer: int,
    prompt_ids_list: Sequence[Sequence[int]],
    generated_ids_list: Sequence[Sequence[int]],
    device: Any,
) -> List[torch.Tensor]:
    """Batched capture of the layer's residual stream at each row's own
    generated-token positions only, via one extra forward pass (no
    generation, no KV-cache growth -- a plain forward pass, so right
    padding is safe: causal masking already blocks every real token from
    attending to anything after it, so trailing pad columns cannot affect
    an earlier real token's hidden state) over the whole (prompt +
    generated) batch with a forward hook -- the same "extra pass with a
    capture hook" pattern ``steer.coherence_logprob_batch`` uses. This
    module runs after the full (untruncated) model is restored for
    steering, so it registers its own light hook on the live decoder layer
    rather than reusing ``harvest.ActivationHarvester`` (which truncates
    every layer past ``layer``, and is only meant for the harvest stage's
    forward-only, no-generation pass).

    Returns one `[gen_len_i, d_model]` tensor per row (zero rows for a row
    with no generated tokens), padding entirely ignored by indexing each
    row's own real (prompt_len, prompt_len + gen_len) slice directly rather
    than via an explicit mask."""
    base = getattr(model, "model", model)
    n = len(prompt_ids_list)
    d_model = model.config.hidden_size
    full_seqs = [list(p) + list(g) for p, g in zip(prompt_ids_list, generated_ids_list)]
    max_len = max((len(s) for s in full_seqs), default=0)
    if max_len == 0:
        return [torch.zeros(0, d_model) for _ in range(n)]

    input_ids = torch.zeros(n, max_len, dtype=torch.long, device=device)
    attention_mask = torch.zeros(n, max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(full_seqs):
        m = len(seq)
        if m == 0:
            continue
        input_ids[i, :m] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, :m] = 1

    captured: Dict[str, torch.Tensor] = {}

    def hook(module: Any, inputs: Any, output: Any) -> None:
        captured["hidden"] = output[0] if isinstance(output, tuple) else output

    handle = base.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    hidden = captured["hidden"]
    out: List[torch.Tensor] = []
    for i in range(n):
        prompt_len = len(prompt_ids_list[i])
        gen_len = len(generated_ids_list[i])
        if gen_len == 0:
            out.append(torch.zeros(0, hidden.shape[-1]))
        else:
            out.append(hidden[i, prompt_len : prompt_len + gen_len])
    return out


def capture_mean_residual_batch(
    model: Any,
    layer: int,
    prompt_ids_list: Sequence[Sequence[int]],
    generated_ids_list: Sequence[Sequence[int]],
    device: Any,
) -> List[np.ndarray]:
    """Mean-pools each row's `capture_generated_hidden_batch` slice over its
    own generated-token positions only. `np.zeros(d_model)` for a row with
    no generated tokens (matching the previous single-generation
    behavior)."""
    d_model = model.config.hidden_size
    hiddens = capture_generated_hidden_batch(model, layer, prompt_ids_list, generated_ids_list, device)
    out = []
    for h in hiddens:
        if h.shape[0] == 0:
            out.append(np.zeros(d_model, dtype=np.float64))
        else:
            out.append(h.float().mean(dim=0).cpu().numpy().astype(np.float64))
    return out


def capture_mean_residual(
    model: Any, layer: int, prompt_ids: Sequence[int], generated_ids: Sequence[int], device: Any
) -> np.ndarray:
    """Single-generation wrapper (batch of one) around
    `capture_mean_residual_batch`."""
    return capture_mean_residual_batch(model, layer, [prompt_ids], [generated_ids], device)[0]


def generate_with_system_prompt(
    model: Any, tokenizer: Any, system_prompt: str, user_prompt: str, max_new_tokens: int, device: Any
) -> Dict[str, Any]:
    """Single-generation wrapper (batch of one) around
    `generation.generate_batch`."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return generation.generate_batch(model, tokenizer, [messages], max_new_tokens, device, hook=None, batch_size=1)[0]


def build_persona_vector(
    model: Any,
    tokenizer: Any,
    layer: int,
    control: Dict[str, str],
    scenarios: Sequence[Dict[str, str]],
    max_new_tokens: int,
    device: Any,
    batch_size: int = 8,
) -> np.ndarray:
    """The Chen/Arditi/Sleight persona-vector construction: mean layer
    residual over assistant (generated) tokens of generations under
    ``positive_system_prompt``, minus the same under
    ``negative_system_prompt``, on the same user scenarios -- averaged over
    scenarios. No steering hook involved here; this only measures the
    model's own natural response under each system prompt.

    Batched: both the positive and negative generation, for every scenario,
    go through one `generation.generate_batch` call (`2 * len(scenarios)`
    rows, chunked at `batch_size`) instead of `2 * len(scenarios)`
    sequential `model.generate` calls."""
    combos = [(label, scenario) for scenario in scenarios for label in ("pos", "neg")]
    msgs = [
        [
            {
                "role": "system",
                "content": control["positive_system_prompt"] if label == "pos" else control["negative_system_prompt"],
            },
            {"role": "user", "content": scenario["prompt"]},
        ]
        for label, scenario in combos
    ]
    gens = generation.generate_batch(model, tokenizer, msgs, max_new_tokens, device, hook=None, batch_size=batch_size)
    embeddings = capture_mean_residual_batch(
        model, layer, [g["prompt_ids"] for g in gens], [g["generated_ids"] for g in gens], device
    )

    by_scenario: Dict[str, Dict[str, np.ndarray]] = {}
    for (label, scenario), emb in zip(combos, embeddings):
        by_scenario.setdefault(scenario["id"], {})[label] = emb

    diffs = [by_scenario[s["id"]]["pos"] - by_scenario[s["id"]]["neg"] for s in scenarios]
    return np.mean(diffs, axis=0)


def sweep_control_doses(
    config: Any,
    model: Any,
    tokenizer: Any,
    persona_vector: np.ndarray,
    scenarios: Sequence[Dict[str, str]],
    device: Any,
) -> List[Dict[str, Any]]:
    """Mini dose-coherence sweep for one control's persona vector, across
    ``config.doses`` x sign in (1, -1) x ``scenarios`` -- the same
    coherence-based dose selection ``steer.run_calibration`` does for real
    features, just scoped to this module since persona vectors are new to
    the screen stage. The persona vector already has a natural residual-
    stream-scale magnitude (it's a difference of real activations, unlike
    a feature's unit-norm decoder direction), so `dose` multiplies it
    directly rather than `dose * max_activation * unit_direction`. Every
    (sign, dose, scenario) triple is generated in a single batched call
    (the hook's per-row scale carries `sign * dose`). Returns rows shaped
    for ``steer.max_coherent_dose``."""
    vector_t = torch.tensor(persona_vector, dtype=torch.float32)
    boundary_by_scenario = {s["id"]: steer.compute_generation_boundary(tokenizer, s["prompt"]) for s in scenarios}

    combos = [(sign, dose, scenario) for sign in (1, -1) for dose in config.doses for scenario in scenarios]
    msgs = [[{"role": "user", "content": scenario["prompt"]}] for _sign, _dose, scenario in combos]
    boundaries = torch.tensor(
        [boundary_by_scenario[scenario["id"]] for _sign, _dose, scenario in combos], dtype=torch.long
    )
    scales = torch.tensor([sign * dose for sign, dose, _scenario in combos], dtype=torch.float32)
    hook_factory = steer.build_steering_hook_factory(model, config.layer, vector_t, boundaries, scales=scales)
    gens = generation.generate_batch(
        model, tokenizer, msgs, config.max_new_tokens, device, hook=hook_factory, batch_size=config.generation_batch_size
    )
    coherences = steer.compute_coherence_batch(
        model, tokenizer, [g["prompt_ids"] for g in gens], [g["generated_ids"] for g in gens], device
    )

    rows: List[Dict[str, Any]] = []
    for (sign, dose, scenario), coherence in zip(combos, coherences):
        rows.append({"sign": sign, "dose": dose, "scenario": scenario["id"], "coherent": coherence["coherent"]})
    return rows


def generate_steered_set(
    config: Any,
    model: Any,
    tokenizer: Any,
    vector: torch.Tensor,
    scenarios: Sequence[Dict[str, str]],
    device: Any,
) -> Dict[str, Dict[str, Any]]:
    """One batched steered generation across every scenario with a fixed
    additive vector (assistant-position-only, same masking as
    ``steer.run_calibration``). Returns {scenario_id: {prompt_ids,
    generated_ids, text, ..., coherence}}."""
    boundary_by_scenario = {s["id"]: steer.compute_generation_boundary(tokenizer, s["prompt"]) for s in scenarios}
    msgs = [[{"role": "user", "content": s["prompt"]}] for s in scenarios]
    boundaries = torch.tensor([boundary_by_scenario[s["id"]] for s in scenarios], dtype=torch.long)
    hook_factory = steer.build_steering_hook_factory(model, config.layer, vector, boundaries)
    gens = generation.generate_batch(
        model, tokenizer, msgs, config.max_new_tokens, device, hook=hook_factory, batch_size=config.generation_batch_size
    )
    coherences = steer.compute_coherence_batch(
        model, tokenizer, [g["prompt_ids"] for g in gens], [g["generated_ids"] for g in gens], device
    )
    out: Dict[str, Dict[str, Any]] = {}
    for scenario, gen, coherence in zip(scenarios, gens, coherences):
        out[scenario["id"]] = {**gen, "coherence": coherence}
    return out


def generate_baseline_set(
    config: Any, model: Any, tokenizer: Any, scenarios: Sequence[Dict[str, str]], device: Any
) -> Dict[str, Dict[str, Any]]:
    """One batched, unsteered generation across every scenario."""
    msgs = [[{"role": "user", "content": s["prompt"]}] for s in scenarios]
    gens = generation.generate_batch(
        model, tokenizer, msgs, config.max_new_tokens, device, hook=None, batch_size=config.generation_batch_size
    )
    return {s["id"]: gen for s, gen in zip(scenarios, gens)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _sign_label(sign: Any) -> str:
    if sign == 1:
        return "pos"
    if sign == -1:
        return "neg"
    return "na"


def build_generation_records(
    directions: Sequence[Dict[str, Any]], baseline_texts: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Flatten every direction's per-scenario screen generations into report
    records: one per (direction, scenario) pair, so the report can show the
    actual steered-vs-baseline text pairs behind each screen-table row
    rather than only the aggregate separability/consistency numbers.

    ``directions`` is ``pending`` from ``run_screen`` (or any sequence of
    dicts shaped like it): each needs ``kind``, ``id``, ``sign``, ``dose``,
    ``steered`` ({scenario_id: {text, finish_reason, coherence, ...}}), and
    optionally ``extra`` (carrying ``arm``, e.g. ``{"arm": "control"}`` --
    absent/``None`` if the entry predates arm tagging).
    ``baseline_texts`` is {scenario_id: baseline text}, shared across every
    direction since baselines are generated once per scenario.

    Returns ``{recordId, payload}`` records ready for
    ``report.report(stage="screen", records=...)``, with
    ``recordId = f"screen-gen-{kind}-{id}-{sign}-{scenario}"`` (``sign``
    rendered as ``pos``/``neg``/``na``, matching every other screen-stage
    recordId) and ``payload = {kind, id, sign, dose, arm, scenario, text,
    baseline_text, finish_reason, coherence}``.
    """
    records: List[Dict[str, Any]] = []
    for entry in directions:
        kind = entry["kind"]
        direction_id = entry["id"]
        sign = entry["sign"]
        dose = entry["dose"]
        arm = (entry.get("extra") or {}).get("arm")
        sign_label = _sign_label(sign)
        for scenario_id, gen in entry["steered"].items():
            payload = {
                "kind": kind,
                "id": direction_id,
                "sign": sign,
                "dose": dose,
                "arm": arm,
                "scenario": scenario_id,
                "text": gen.get("text"),
                "baseline_text": baseline_texts.get(scenario_id, ""),
                "finish_reason": gen.get("finish_reason"),
                "coherence": gen.get("coherence"),
            }
            records.append(
                {"recordId": f"screen-gen-{kind}-{direction_id}-{sign_label}-{scenario_id}", "payload": payload}
            )
    return records


def run_screen(
    config: Any,
    model: Any,
    tokenizer: Any,
    sae: Any,
    feature_stats: Dict[str, Any],
    calibration_records: Sequence[Dict[str, Any]],
    calibrate_scenarios: Sequence[Dict[str, str]],
    screen_scenarios: Sequence[Dict[str, str]],
    device: Any,
    seed: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Runs the full screen stage and returns
    ``{"directions": [...], "generations": [...]}``, both lists of
    ``{recordId, payload}`` records ready for ``report.report(stage="screen",
    records=...)``. ``"directions"`` is one record per direction (feature/
    sign, control/sign, or random null) with its separability/consistency
    verdict inputs; ``"generations"`` is one record per (direction,
    scenario) pair carrying the actual steered/baseline text (see
    ``build_generation_records``).

    ``calibrate_scenarios`` is the (smaller) scenario set the calibrate
    stage already swept (``config.steer_scenarios``) -- reused here for
    dose selection (features: read straight from ``calibration_records``;
    controls: their own mini dose-coherence sweep, since persona vectors
    are new to this stage). ``screen_scenarios`` is the larger set
    (``config.screen_scenarios``) every direction is finally scored on.
    """
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator(device="cpu").manual_seed(seed)
    d_model = sae.d_in
    W_dec = sae.W_dec.detach()

    # -- baselines across the full screen scenario set --------------------
    baselines = generate_baseline_set(config, model, tokenizer, screen_scenarios, device)
    baseline_sids = list(baselines.keys())
    baseline_resids = capture_mean_residual_batch(
        model,
        config.layer,
        [baselines[sid]["prompt_ids"] for sid in baseline_sids],
        [baselines[sid]["generated_ids"] for sid in baseline_sids],
        device,
    )
    baseline_resid_by_scenario = dict(zip(baseline_sids, baseline_resids))
    baseline_lexical_by_scenario = {sid: hashed_lexical_vector(baselines[sid]["text"]) for sid in baseline_sids}

    pending: List[Dict[str, Any]] = []

    # -- features: reuse calibrate-stage max_coherent_dose -----------------
    feature_info = extract_feature_info(calibration_records)
    feature_doses = feature_max_coherent_doses(calibration_records)
    for f_idx, info in sorted(feature_info.items()):
        direction_vec = W_dec[f_idx]
        max_act = info.get("max_activation") or 0.0
        for sign in (1, -1):
            dose_key = "pos" if sign > 0 else "neg"
            dose = resolve_dose(feature_doses.get(f_idx, {}).get(dose_key), config.screen_dose_fallback)
            vector = (sign * dose * max_act) * direction_vec
            steered = generate_steered_set(config, model, tokenizer, vector, screen_scenarios, device)
            pending.append(
                {
                    "kind": "feature",
                    "id": f_idx,
                    "sign": sign,
                    "dose": dose,
                    "steered": steered,
                    "extra": {
                        "density": info.get("density"),
                        "quantile": info.get("quantile"),
                        "arm": info.get("arm"),
                    },
                }
            )

    # -- controls: persona vectors, own mini dose-coherence sweep ----------
    for control in config.control_prompts:
        persona_vector = build_persona_vector(
            model,
            tokenizer,
            config.layer,
            control,
            calibrate_scenarios,
            config.max_new_tokens,
            device,
            batch_size=config.generation_batch_size,
        )
        sweep_rows = sweep_control_doses(config, model, tokenizer, persona_vector, calibrate_scenarios, device)
        for sign in (1, -1):
            sign_rows = [r for r in sweep_rows if r["sign"] == sign]
            chosen_dose = resolve_dose(steer.max_coherent_dose(sign_rows), config.screen_dose_fallback)
            vector = torch.tensor((sign * chosen_dose) * persona_vector, dtype=torch.float32)
            steered = generate_steered_set(config, model, tokenizer, vector, screen_scenarios, device)
            pending.append(
                {
                    "kind": "control",
                    "id": control["name"],
                    "sign": sign,
                    "dose": chosen_dose,
                    "steered": steered,
                    "extra": {"arm": "control"},
                }
            )

    # -- random-direction nulls: dose = median of feature doses ------------
    random_dose = median_dose_with_fallback(flatten_doses(feature_doses), config.screen_dose_fallback)
    for r in range(config.random_directions):
        random_unit = torch.randn(d_model, generator=torch_gen)
        random_unit = random_unit / random_unit.norm().clamp_min(1e-8)
        vector = random_dose * random_unit
        steered = generate_steered_set(config, model, tokenizer, vector, screen_scenarios, device)
        pending.append(
            {
                "kind": "random",
                "id": r,
                "sign": 0,
                "dose": random_dose,
                "steered": steered,
                "extra": {"arm": "random"},
            }
        )

    # -- embeddings + per-scenario diff vectors for every direction --------
    # One capture_mean_residual_batch call per direction (across its own
    # scenarios) rather than one per (direction, scenario) pair.
    for entry in pending:
        sids = list(entry["steered"].keys())
        gens = [entry["steered"][sid] for sid in sids]
        resids = capture_mean_residual_batch(
            model, config.layer, [g["prompt_ids"] for g in gens], [g["generated_ids"] for g in gens], device
        )
        resid_by_scenario = dict(zip(sids, resids))
        lexical_by_scenario = {sid: hashed_lexical_vector(g["text"]) for sid, g in zip(sids, gens)}
        diffs_resid = [
            resid_by_scenario[sid] - baseline_resid_by_scenario[sid]
            for sid in sids
            if sid in baseline_resid_by_scenario
        ]
        coherent_flags = [bool(g["coherence"]["coherent"]) for g in gens]
        entry["resid_by_scenario"] = resid_by_scenario
        entry["lexical_by_scenario"] = lexical_by_scenario
        entry["diffs_resid"] = diffs_resid
        entry["coherent_fraction"] = (sum(coherent_flags) / len(coherent_flags)) if coherent_flags else float("nan")

    # -- separability + consistency (needs the full pool for the null) -----
    records: List[Dict[str, Any]] = []
    for entry in pending:
        sep_resid = leave_one_scenario_out_separability(entry["resid_by_scenario"], baseline_resid_by_scenario)
        sep_lexical = leave_one_scenario_out_separability(entry["lexical_by_scenario"], baseline_lexical_by_scenario)

        other_pool = [d for other in pending if other is not entry for d in other["diffs_resid"]]
        consistency = direction_consistency(entry["diffs_resid"], other_pool, rng)

        payload: Dict[str, Any] = {
            "kind": entry["kind"],
            "id": entry["id"],
            "sign": entry["sign"],
            "dose": entry["dose"],
            "n_scenarios": len(entry["steered"]),
            "separability": {"resid": sep_resid, "lexical": sep_lexical},
            "consistency": consistency,
            "coherent_fraction": entry["coherent_fraction"],
        }
        payload.update(entry["extra"])

        record_id = f"screen-{entry['kind']}-{entry['id']}-{_sign_label(entry['sign'])}-{entry['dose']}"
        records.append({"recordId": record_id, "payload": payload})

    baseline_texts = {sid: gen["text"] for sid, gen in baselines.items()}
    generation_records = build_generation_records(pending, baseline_texts)

    return {"directions": records, "generations": generation_records}
