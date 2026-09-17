# Experiment 3C handoff (written 2026-09-17, ~10:00 UTC)

Read this first in the next session. Work in `C:\Users\rapha\Projects\autolabs`. Everything below is committed on `main` (0d7ac1b or later). No RunPod pods exist; nothing is billing. RunPod balance $13.60.

## State

- Pipeline code for the full run is ready at commit `0e1f961` (201 tests, `python -m pytest research/experiment-003c/pipeline/tests -q`). It has run end to end on a GPU once (validation 2, commit b224c9c, plus the clustering fix validated offline on that run's judge outputs).
- Worker: `/api/persona-3c/*` routes deployed, including the judge queue with the order-independent schema and budget guard. Homepage card live.
- The full run attempt failed: pod `l9vcz65vhr5gqb` lost its GPU (host hardware fault) 3 hours in, before the first checkpoint. That was RunPod's failure, not ours. Our own contributing weakness: first checkpoint was set at 10M tokens, so 3 hours were lost.
- Findings so far: see VISUAL-1.md, VISUAL-2.md, FULL-RUN-LAUNCH.md, and `smoke-runs/validate-2/` (feature 1134 named by the semantic clustering; feature 75 in visual-1 stripped the Qwen identity).

## Relaunch options (owner decision)

| Option | Config | Time | GPU cost |
|---|---|---|---|
| A | `configs/full-100m.json`, A6000 $0.53/h | about 20 h | about $11 |
| B | `configs/full-100m.json`, A100 80 GB $1.59/h | about 7.5 h | about $12 |
| C | `configs/full-60m.json`, A6000 | about 12 h | about $7 |
| D | top up $20, `configs/full.json` (150M) on A100 | about 11 h | about $18 |

## Making the run survive a pod failure

1. Checkpoint every 5M tokens (already in full-100m/full-60m). Optimizer state is not checkpointed; the step counter is, so a resume restarts Adam moments only.
2. Copy every checkpoint off the pod as it lands: either set `HF_TOKEN` in the pod env (config `hf_upload_repo`, private repo `RaphaelRaphaelRaphael/autolabs-3c-sae`), or run an rsync loop from the laptop. Then a dead host costs at most 5M tokens.
3. Do not restart a stopped pod; hosts are usually full and the volume gets stranded. Terminate and create a fresh pod, then resume from the last off-pod checkpoint by copying it into `/workspace/3c/checkpoints/` before launch.
4. Prefer a data center with High stock for the chosen GPU (`get-capacity` or the console) so a replacement pod can be created immediately.
5. Watchers on the laptop die when the laptop sleeps; a Sonnet scheduled routine every 30 min (harness status + pod status via REST, relaunch from last checkpoint, alert on failure) is the robust monitor and is not built yet.

## Launch procedure (fresh pod)

1. `POST /api/persona-3c/start` with manifestHash = sha256 of the chosen config file, budgetUsd 12, idempotencyKey.
2. Create pod via REST v1 (`gpuTypeIds`, `imageName runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 40 GB container, 80 GB volume at `/workspace`, port 22/tcp, env `PUBLIC_KEY` = contents of `~/.ssh/autolabs_runpod.pub`, `PIP_BREAK_SYSTEM_PACKAGES=1`, `HF_HOME=/workspace/3c/hf`).
3. Write `/workspace/3c/.env` with `AUTOLABS_3C_WORKER_URL`, `AUTOLABS_3C_TOKEN`, `AUTOLABS_3C_RUN_ID`, `AUTOLABS_3C_GIT_REF=<commit>` and optionally `HF_TOKEN`, LF line endings.
4. `curl` the pinned `runpod_start.sh` from GitHub, run `bash start.sh configs/full-100m.json` under nohup. Log at `/workspace/3c/smoke.log`; a `[train] progress` line every 1M tokens with ETA.
5. On completion the pipeline reports `done`; copy `summary.json` and the reports off the pod and stop the pod.

Credentials: `research/experiment-003c/.env.runpod` (gitignored) holds the RunPod key, Worker URL and 3C token. SSH key `~/.ssh/autolabs_runpod`. Never print them.
