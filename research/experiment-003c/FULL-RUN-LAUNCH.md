# Experiment 3C full run: launch record (2026-09-17, 06:20 UTC)

Status: FAILED at 09:1x UTC, GPU hardware fault on the pod host (driver reported no devices), before the first 10M-token checkpoint. Nothing was recoverable from that pod. Both pods terminated; no GPU is billing. Decision needed in the morning, see the bottom of this file.

| Item | Value |
|---|---|
| Harness run | `persona-3c-1e87ec7f-c175-4f3a-804a-afd077bf3eb7` |
| Pipeline commit | `905213b` |
| Config | `pipeline/configs/full.json`, sha256 `b9889fd4a91de1b2…` (recorded as the harness manifest hash) |
| Pod | `l9vcz65vhr5gqb`, RTX A6000 48 GB, secure cloud, EU, $0.53/h, 80 GB volume |
| Started | 2026-09-17 06:18 UTC |
| RunPod balance at launch | $16.82 |
| Judge cap | $10 (`judge_budget_usd`), 2,000 calls |

## What runs, in order

1. Boot checks (identity hook, assistant mask).
2. Harvest + train: matryoshka BatchTopK SAE, width 32,768, shells 1k/4k/16k/32k, k 40, 150M assistant tokens from UltraChat, 8 optimizer steps per harvested batch, checkpoints every 10M tokens with progress records to the harness. Expected about 16 hours on this GPU.
3. Assistant-only reconstruct check; held-out FVE.
4. Rank: three selection arms, 96 unsupervised (assistant-specificity, breadth, topic invariance, shell), 64 density-quantile, 96 prompt-shift. No trait is specified anywhere in the unsupervised arm.
5. Calibrate: 256 features x 2 signs x 4 doses on 4 scenarios, batched, coherence guard, per-feature max coherent dose.
6. Screen: each direction at its dose on 24 open-ended scenarios, 3 persona-vector controls, 20 random nulls; leave-one-scenario-out separability in residual and lexical views; consistency vs null. Gate G0: controls must beat every random direction.
7. Describe: top 40 directions plus 3 nulls to the Luna judge, blinded pairs both orders, "name one property of how the speaker comes across and which response has more of it"; clustered into data-derived names; consistency-score gate against the null ceiling.
8. Reach: up to 12 named directions plus named controls; four prompt variants including few-shot with the direction's own steered replies; behavioral and mechanistic reach reported by arm. A measured outcome, not a gate.
9. Analysis, summary.json, HTML reports; results copied off the pod; pod stopped.

## Claim under test

Chen, Arditi, Sleight et al. list three limitations of prompt-derived persona vectors: the trait must be specified in advance, it needs a precise description, and it must be prompt-inducible. The headline claim here is the first two: unsupervised discovery of persona-relevant directions with no trait specified. Prompt reachability is reported per arm, not gated, because Qwen is prompt-inducible for nearly everything.

## Caveats known at launch

- The generation stages (batched generation, rank, three arms, reach) have not yet run on a GPU; they are being validated on the visual-2 pod in parallel. If a fix is needed the full run resumes from its training checkpoints on the same pod.
- No HF_TOKEN on the pod, so the SAE upload is a logged no-op; the SAE is copied off the pod over SSH after training instead.
- Budget: about $10 of GPU for this run leaves no room for a second training attempt tonight.

## Timeline

- 06:18 UTC launch.
- 06:44 UTC Worker deployed with the order-independent judge schema (89d8b18); visual-2 finished under the old schema, see VISUAL-2.md.
- 06:52 UTC validation run of commit 905213b (smoke config, all new stages) launched on a separate A40 pod `u8k3qftr747czw`, harness run `persona-3c-2e8cafc4-5df4-4959-b54f-bcb71724aba8`.
- 07:27 UTC validation 1 (commit 905213b) crashed in calibrate: batched steering hook received whole-request boundaries for a chunk. Fixed in b224c9c.
- 08:41 to 09:08 UTC validation 2 (commit b224c9c, pod udallb31ra7485) ran the entire funnel end to end: rank (3 unsupervised, 2 quantile, 3 shift), calibrate, screen (20 nulls, max random AUC 0.56; unsupervised 3/3 pass, shift 3/3, quantile 0/2, controls 2/3), describe (144 judge pairs, about $0.65), reach. Generation stages took 27 minutes batched. Artifacts in `smoke-runs/validate-2/`.
- Describe named nothing: judge properties are paraphrases and TF-IDF clustering split them into singletons (largest cluster fraction about 0.12 everywhere, null ceiling 0.08). Fix in progress: sentence-embedding clustering calibrated on these outputs. The full run will resume onto the fixed commit before its describe stage.
- 09:11 UTC full run still training, no 10M checkpoint yet after 2h53m; rate check pending.
- 09:19 UTC full-run pod lost its GPU (`Unable to determine the device handle for GPU0 ... No devices were found`). Training crashed after about 3 hours with no checkpoint written (first checkpoint was set at 10M tokens). Harness run marked failed by the pipeline's own failure report. Pod terminated. RunPod balance after: $13.60. GPU spent tonight: about $4.60 across validation 1, validation 2, and the failed full run.

## Morning decision

What is proven tonight: the complete funnel runs end to end on the final code (validation 2), the harness records every stage, the judge path works with the new schema, and the SAE recipe reaches held-out FVE 0.72 at width 8k. What is not yet fixed: describe-stage clustering (semantic embeddings, in progress on the laptop, no GPU needed) and the training rate of the 32k dictionary, which did not reach 10M tokens in 3 hours at 8 steps per batch, implying well over 30 hours for 150M.

Options for relaunch, all on secure cloud:

| Option | Config | Est. time | Est. GPU cost | Fits $13.60? |
|---|---|---|---|---|
| A | 32k width, 100M tokens, 4 steps/batch, A6000 $0.53/h | about 19 h train + 1 h stages | about $11 | yes, no reserve |
| B | 32k width, 100M tokens, 4 steps/batch, A100 80GB $1.59/h | about 7 h train + 0.5 h stages | about $12 | yes, no reserve |
| C | 32k width, 60M tokens, 4 steps/batch, A6000 | about 12 h | about $7 | yes, $6 reserve |
| D | top up RunPod by $20, then option A or B with 150M | 24 to 30 h | $15 to $20 | needs top-up |

Recommendation: D if the 150M target still matters, otherwise B for speed with a checkpoint every 5M tokens and the new per-1M-token throughput log so the rate is known within the first hour. Either way, launch only after the clustering fix is committed, so the run does not need to be resumed onto a new commit before describe.

