# Experiment 3C visual run 2 (2026-09-17)

Run `persona-3c-c58185a5-f7a0-49d0-a8dd-47556bbce5b3`, commit e22c474, pod `8n2nscddzdfb8f` (RTX A6000), 04:25 to 06:32 UTC, about $1.10. Artifacts in `smoke-runs/visual-2/`. First run of the whole funnel in one process: train, calibrate, screen with 20 random nulls, and an automated describe plan.

- SAE reproduced visual-1: held-out FVE 0.723, dead 0.4%.
- Screen with 20 nulls: max random residual AUC 0.594 (95th percentile 0.564), a more honest null ceiling than the two-draw 0.48 of visual-1. 5 of 8 features and 2 of 3 controls pass gate G0 by the either-sign rule.
- Describe: 96 pairs planned to the Worker; the old client's 30 s read timeout stopped the drive loop while the Worker kept judging. Fixed in 89d8b18 (300 s, 10 jobs per call). No describe results were produced for this run.
- Unbatched generation timing confirmed the need for batching before the full run (calibrate + screen took about 70 minutes for 8 features).
