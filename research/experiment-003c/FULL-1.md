# Experiment 3C — Full run 1 (FULL-1): result and post-mortem

**Date:** 2026-09-18 · **Status:** funnel completed end to end; **discovery result inconclusive (underpowered screen).**

## One-paragraph summary

The 100M-token Matryoshka SAE trained on 2026-09-17 (run `persona-3c-d77e7c1e`,
held-out FVE **0.717**, L0 **40**, width 32,768) was carried through the full
discovery funnel on 2026-09-18. Training and ranking are solid and preserved on
the Hub. The **screen stage did not produce a trustworthy verdict**: the
persona-vector positive controls (evil, sycophantic) failed to separate above a
near-chance null ceiling, so the pipeline **correctly named nothing**. The cause
is not the SAE — it is that the run was forced into an **ultra-lean, underpowered
screen** to fit a nearly-exhausted RunPod balance. The discovery question is
therefore **still open**; a full-power screen is needed to answer it.

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

## The screen verdict (`smoke-runs/full-1/summary.json` → `screen_verdict`)

| Field | Value | Read |
|---|---|---|
| `max_random_resid_auc` (null ceiling) | **0.539** | barely above chance (0.5) — a *low, noisy* ceiling |
| `random_directions` | 16 | fewer nulls than the design's 20 |
| `features_passing` | **96 / 256** | nominally clear the low ceiling |
| `feature_directions_passing` | 124 / 512 | per-sign |
| `controls_passing` | **1 / 3** | **G0 FAILED** — positive control did not hold |
| `all_controls_pass` | **False** | screen is inconclusive by its own gate |

Controls detail: `hallucinating_factual` (+) passed; **`evil_benevolent` and
`sycophantic_honest` failed both signs.** Those are strong, well-known persona
directions — if the screen cannot separate *them* from random noise, its "96
passing features" are not evidence of anything. Describe was correctly gated off
(`describe_results.json`: `results: []`, 0 judge calls, `$0`). Reach: empty.

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
retraining): 24 scenarios, 512-token generations, 4 doses, 20 nulls, and calibrate
scenarios back up (≥6). Budget it properly: ~2–3 GPU-hours on an A100
(~$4–5), which needs a top-up. The pass/fail gate to watch is **`all_controls_pass`**
— only trust the feature passes in a run where the persona-vector controls clearly
beat the null ceiling. Everything is staged to do this in one command once funded.
