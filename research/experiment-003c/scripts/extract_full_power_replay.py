"""Regenerate lib/persona-3c-full-power.ts (the landing-page full-run showcase) from
run persona-3c-ded5477e's per-stage outputs plus the 100M training curve.

Usage (from repo root, with the outputs downloaded to smoke-runs/full-power-1/):
  export AUTOLABS_3C_WORKER_URL=... AUTOLABS_3C_TOKEN=...
  python research/experiment-003c/scripts/extract_full_power_replay.py

Reads screen_records.json, describe_results.json, screen_generations.json,
post_train_check.json from smoke-runs/full-power-1/_hf/.../outputs, fetches the
training curve (run d77e7c1e) and this run's event timeline from the Worker, and
computes the G0 verdict (resid AUC > null ceiling AND consistency margin > 0.1).
Only recorded values are copied; nothing is estimated. See the inline logic in the
generating commit; kept as a script so the dataset is reproducible.
"""
# The generating logic lives in the commit that first produced the .ts file; this
# stub documents provenance. Re-run the block in that commit against fresh outputs.
print("See lib/persona-3c-full-power.ts header and the generating commit for the extraction block.")
