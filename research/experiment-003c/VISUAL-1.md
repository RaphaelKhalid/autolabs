# Experiment 3C visual run 1 (2026-09-17)

Run `persona-3c-d929d842-6d2a-4b22-870d-424195d55f08`, commit b864e82, pod `yuxhrngdutugnu` (RTX A6000, $0.53/h, EU-SE-1), 02:08 to 03:52 UTC, about 105 minutes, roughly $0.95. Artifacts in `smoke-runs/visual-1/`. All 327 records (boot 2, train 9, calibrate 292, screen 24) reached the harness.

## SAE (width 8192, 8M assistant tokens, 4 steps per batch, warmup 500)

| tokens | steps | FVE | dead |
|---|---|---|---|
| 1M | 772 | 0.561 | 72.9% |
| 4M | 3,144 | 0.707 | 13.7% |
| 8M | 6,320 | 0.733 | 0.5% |

Held-out FVE 0.724. Assistant-only reconstruct-and-replace check over 118 positions: argmax match 0.66, replaced FVE 0.615, cross-entropy increase 0.63 nats. Below the 0.9 match target but now a valid measurement; the curve is still rising at 8M, which supports 150M for the full run.

## Screen (8 features x 2 signs, 3 persona-vector controls x 2 signs, 2 random nulls; 8 scenarios; leave-one-scenario-out)

- Both random directions sit at chance: residual AUC 0.46 and 0.48. The null is calibrated.
- Sycophantic/honest control (+): residual AUC 1.00, lexical AUC 1.00, consistency 0.55 vs null 0.10. Evil/benevolent (+) AUC 0.83, hallucinating/factual (+) AUC 0.81; their negative signs are weak.
- Features 75(-) and 1007(+) reach residual AUC 1.00 at dose 2.0 with consistency 0.30 and 0.19 against nulls of 0.07 and 0.05; feature 75(-) also separates lexically (AUC 0.95). Feature 1019(+) has the highest feature consistency (0.45) but only 3 of 8 generations coherent.
- Verdict as implemented: 3 of 16 feature directions beat the maximum random AUC with a consistency margin above 0.1; 2 of 6 control directions do, so `all_controls_pass` is false. The verdict counts control signs separately; G0 should count each control as passing if either sign passes, which gives 2 of 3 here, with evil failing only on the consistency margin (0.10 vs 0.06).

## Reading

The funnel separates for the first time: random directions stay dark, the strongest positive control lights up on both views, and two SAE features separate as cleanly as the control. Whether those features are persona-like or topic or format effects is exactly the question the describe stage (judge pass one) answers; nothing here names them. Eight scenarios and two nulls are too few for a claim; the full run uses 24 scenarios and 20 nulls.

## Next

1. Verdict per control, not per sign; persist screen-stage generation texts so the report can show them; 20 random nulls in the smoke config too.
2. Describe stage: blinded pairs to the judge with "describe the difference or say none", clustered per direction, run through the Worker.
3. Full run config: width 32k, 150M tokens, 256 features ranked by persona-context shift, 24 scenarios, 512 tokens.
