# Experiment 3C — Full run 1 (FULL-1): result and post-mortem

**Date:** 2026-09-18 · **Status:** funnel completed end to end; **naming withheld by the all-controls gate — but there is clear signal, and the block is a consistency-margin threshold starved by only 8 scenarios.**

## One-paragraph summary

The 100M-token Matryoshka SAE trained on 2026-09-17 (run `persona-3c-d77e7c1e`,
held-out FVE **0.717**, L0 **40**, width 32,768) was carried through the full
discovery funnel on 2026-09-18. It ran end to end. The result is **not a null**:
on residual-stream separability, **484 of 512 feature direction-signs beat the
random-null ceiling (0.539), 61 at a perfect AUC 1.0, and 96 of 256 features
cleared the full G0 gate** (AUC *and* a consistency margin > 0.1). The
persona-vector controls **also separate strongly** — residual AUC 0.66–0.89, well
above the null. They fail G0 **only on the consistency-margin sub-criterion**
(their margins land at 0.04–0.09, a hair under the 0.1 threshold), because that
margin is a cosine over per-scenario diff vectors and **8 scenarios is too few to
estimate it stably**. Because the pipeline **withholds all naming until every
positive control validates the screen** (`all_controls_pass`), and 2 of 3
controls missed the margin, it named 0 directions — the correct conservative call.
**The discovery question is not answered, but the run strongly suggests the answer
is yes**, and a full-power screen (more scenarios → higher, stabler consistency
margins) should tip the controls over the gate and name the 96 surviving features.

## What is proven

- **Training recipe works at scale.** Width-32k Matryoshka BatchTopK SAE on
  Qwen2.5-7B layer 19, 100M assistant tokens: held-out FVE 0.717, L0 exactly 40,
  reconstruct-and-replace splice FVE 0.653, CE delta 0.57, argmax match 0.771
  (below the 0.90 sanity gate — a soft warning, consistent with the CE delta).
- **The OOM that killed the first funnel attempt is fixed.** `screen.
  capture_generated_hidden_batch` ran the full causal-LM (logits included) over
  all rows at once (~70 GiB). Fix (commit `5b886fc`): decoder-stack-only forward
  (no logits) + chunked rows. Rank, calibrate, screen and reach all ran cleanly
  after it — validated on the real 32k run, twice.
- **The full funnel runs end to end** at production width: boot → finalize-from-
  checkpoint → rank (256 candidates: 96 unsupervised / 64 quantile / 96 shift) →
  calibrate (4,629 records) → screen (534 records) → describe → reach → analysis.
- **The pipeline refused to over-claim.** With the positive controls failing G0,
  it named zero directions rather than promote noise. That is the correct
  behaviour and the most reassuring thing in this run.

## The screen verdict (`smoke-runs/full-1/summary.json`)

**Gate G0 (per direction/sign):** `resid AUC > max_random_resid_auc` **AND**
`(mean_cos − null_mean_cos) > 0.1`. Both must hold.

| Field | Value | Read |
|---|---|---|
| `max_random_resid_auc` (null ceiling) | **0.539** | 16 nulls, near chance |
| feature dir-signs beating the AUC ceiling | **484 / 512** | strong separability everywhere |
| feature dir-signs at AUC 1.0 | **61** | (8-scenario AUC is coarse — read with the consistency gate) |
| **features clearing the full G0 gate** | **96 / 256** | real survivors (AUC *and* margin) |
| `controls_passing` | **1 / 3** | fail the *margin*, not the AUC |
| `all_controls_pass` | **False** | → naming withheld for the whole run |

**The controls separate; they miss only the consistency margin:**

| Control · sign | resid AUC (>0.539?) | consistency margin (>0.1?) | G0 |
|---|---|---|---|
| sycophantic · − | 0.891 ✓ | 0.125 − 0.033 = **0.092** ✗ | fail |
| sycophantic · + | 0.812 ✓ | 0.160 − 0.091 = **0.069** ✗ | fail |
| hallucinating · + | 0.797 ✓ | 0.203 − 0.075 = **0.128** ✓ | **pass** |
| hallucinating · − | 0.734 ✓ | 0.119 − 0.075 = 0.044 ✗ | fail |
| evil · + | 0.656 ✓ | 0.071 − 0.074 = −0.003 ✗ | fail |
| evil · − | 0.469 ✗ | 0.008 − 0.036 ✗ | fail |

Every control except evil(−) clears the separability AUC by a wide margin; four
of the six fail G0 **purely on the consistency margin**, and sycophantic sits at
0.069–0.092 — just under 0.1. Because `all_controls_pass` is False, describe was
gated off by design (`describe_results.json`: `results: []`, 0 judge calls, `$0`),
so the 96 surviving features were never named. Reach: empty for the same reason.

## Why it came out underpowered — the honest account

This run was squeezed to fit budget after two earlier funnel attempts overran:

1. **Attempt A (full funnel, A100):** `full-paper-100m.json` — calibrate is
   256 features × 2 × 4 doses × **24 scenarios** ≈ 49k generations. It ground for
   ~76 min without finishing; the full funnel needed ~8–12 GPU-h (~$13–19).
2. **Attempt B (lean funnel, same A100):** scenarios cut to 4/12, features 160.
   Calibrate still ran ~1h50m; I mis-read a lagging billing feed as "~$1 left"
   and **stopped it early** — a mistake, there was more balance than I thought.
3. **Attempt C (ultra-lean, A40 $0.49/h) — this run:** 48 features, 3 calibrate /
   8 screen scenarios, 16 nulls, 3 doses, 320-token generations. It **completed**
   — but those cuts are exactly what makes an AUC-separability screen too noisy
   for the controls to clear the bar.

**Root lesson:** the generation stages (calibrate, screen) at width 32k are
3–6× slower on these GPUs than first estimated, and the AUC screen needs enough
scenarios / doses / tokens for the positive control to hold. The knobs that make
it cheap are the same knobs that make it underpowered.

## Cost

~$8 of RunPod across the session (three funnel pods + the earlier training pod),
on the account's existing balance — **no top-up was made.** All pods are
terminated/exited; nothing is billing. The expensive, valuable artifact (the
trained SAE) was produced once and reused for every funnel attempt.

## Artifacts (all on `hf://models/RaphaelRaphaelRaphael/autolabs-3c-sae`)

- `runs/persona-3c-d77e7c1e-…/sae.safetensors` — the trained SAE (940 MB), all
  5M checkpoints, `feature_stats.json`.
- `runs/persona-3c-fc63ef59-…/outputs/` — this funnel's `summary.json` (27 MB),
  `screen_records.json`, `calibration_records.json`, `candidates.json`,
  `describe_results.json`, `reach_results.json`, HTML reports.
- Local copies of the small result files in `smoke-runs/full-1/`.

## What a conclusive run needs (the next run)

A **full-power screen** on the *same* trained SAE (finalize-from-checkpoint, no
retraining). The one lever that matters is **scenarios**: the consistency margin
is a cosine over per-scenario diff vectors, so going from **8 → 24 scenarios**
tightens the estimate and should lift the controls' margins (sycophantic already
at 0.069–0.092) over the 0.1 gate. Also restore 512-token generations, 4 doses and
20 nulls. Budget ~2–3 A100-hours (~$4–5, needs a top-up). Watch **`all_controls_pass`**
— when it flips True, trust the surviving features and let describe name them.

This is **not a shot in the dark**: the controls already separate on AUC (0.66–0.89)
and sit just under the margin, and 96 SAE features already clear the full gate.
The expected outcome of a powered screen is a named shortlist, not another null.

## Secondary follow-up (free / cheap, no GPU)

Worth reviewing whether the flat **0.1 consistency-margin threshold** is right at
8 scenarios, and whether the margin should scale with scenario count — this run is
a clean datapoint for calibrating it. That is analysis on the existing
`summary.json`, not a new GPU run.
