# Experiment 3C smoke run 2 (2026-09-16)

Run `persona-3c-46c6412d-e22d-44eb-8b18-6ef6f2dc7d8f`, commit 8c04f90, pod `qoxrbyjuapkcp6` (RTX A6000 48 GB, secure, $0.53/h, EU-SE-1), 22:35 to 23:46 UTC, about 71 minutes, roughly $0.65. Artifacts: `smoke-runs/smoke-2/`.

## SAE (width 8192, shells 1k/4k/8k, k 40, 8M assistant tokens, one step per harvested batch)

| tokens | loss | FVE | dead fraction |
|---|---|---|---|
| 1M | 1.252 | 0.435 | 0.814 |
| 3M | 0.994 | 0.591 | 0.602 |
| 6M | 0.869 | 0.661 | 0.359 |
| 8M | 0.827 | 0.679 | 0.250 |

Held-out FVE 0.675, L0 40. The curve is healthy but step-starved (about 2,000 optimizer steps); commit 45fa563 adds `train_steps_per_batch` and warmup for the next run.

## Reconstruct-and-replace check: still failing, and now explained

Argmax match 0.148 over 89 prompt tokens, and FVE of the replaced residual is negative (-3.65) even though held-out FVE on harvested activations is 0.68. The SAE was trained only on assistant-turn positions; the check replaces every position of a plain (non-chat-template) prompt, including tokens the SAE never saw. The check must render the chat template and score assistant positions only. Until then this check says nothing about SAE quality.

## Steering (fractions 0.25 to 2.0 of each feature's max activation, assistant positions only)

- Every feature is coherent at doses 0.25 to 1.0 on 8/8 generations; dose 2.0 breaks the densest features (0/8 coherent for features 26 and 1014) and spares the sparse ones (8/8 for 787 and 1019). Per-feature max coherent dose is now measurable, which was the goal.
- Random directions at matched norm are equally coherent and produce similar edit distances (0.60 to 0.87). Edit distance and coherence therefore cannot separate a feature from a random direction; only consistency of the change across scenarios can. That is the job of the screen stage (held-out separability, then clustered difference descriptions), not of calibration.
- Quantile selection now spans densities 0.0012 to 0.086.
- Harness: only the 2 boot records landed; calibrate records were rejected because floats serialized differently in Python and the Worker. Fixed in bfa09ad (hash over exact client bytes); this run used the older client.

## Next (smoke-3 or the visual pipeline)

1. Post-train check on chat-templated assistant positions only, reporting FVE and argmax match there.
2. Retrain with `train_steps_per_batch` 4 and warmup; target held-out FVE above 0.8 and dead below 10% at 8M tokens before scaling width.
3. Add the screen stage: embeddings of steered versus baseline, held-out classifier across scenarios, positive controls from published persona vectors, random-direction nulls in the same batch.
4. Run on the new client so calibrate and train records reach the harness.
