# Experiment 3C full run: launch record (2026-09-17, 06:20 UTC)

Status at writing: LAUNCHED, training in progress. This file is updated as stages complete.

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
