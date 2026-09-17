# Experiment 3C handoff

## Update 2026-09-17, 20:40 UTC: full run in progress, unattended

Run `persona-3c-d77e7c1e-8faa-46fd-b262-912d22b38c90` (config `pipeline/configs/full-paper-100m.json`, commit `1c2b7ae`) is training on pod `m111ksx2quob1g` (A100 SXM, $1.59/h, `ssh -i ~/.ssh/autolabs_runpod root@154.54.102.23 -p 10056`). See FULL-RUN-LAUNCH.md attempt 3. Checkpoints and stage outputs upload to `hf://RaphaelRaphaelRaphael/autolabs-3c-sae/runs/<run id>/`. A guardian script on the pod (`/workspace/3c/guardian.sh`, log `guardian.log`) relaunches the pipeline up to 3 times if it exits early and then runs `runpodctl stop pod` so billing ends. The homepage card follows the run.

If the host dies: create a fresh pod (`scripts/runpod_pod.py create ...`), write `/workspace/3c/.env` with the same run id, `AUTOLABS_3C_GIT_REF=1c2b7ae4648daabbcf037c523b13e678abfdb9f5`, `HF_TOKEN`, `PIP_BREAK_SYSTEM_PACKAGES=1`, and run `bash start.sh configs/full-paper-100m.json`; training resumes from the last Hub checkpoint. If the Worker has marked the run failed (409 on reports), register a new run and set `AUTOLABS_3C_CHECKPOINT_RUN_ID` to the old id. To finish a run whose training cannot continue, add `--finalize-from-checkpoint`.

When it completes: pull `summary.json` and the reports from the Hub `outputs/` folder into `smoke-runs/full-1/`, write `FULL-1.md`, and put the results on the site.

## Original handoff (written 2026-09-17, ~10:00 UTC)

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

## Making the run survive a pod failure (updated 2026-09-17 afternoon)

Implemented in the pipeline (see pipeline/README.md "Resumability" and "Throughput"):

1. Checkpoint every 5M tokens with weights, step counter, Adam state, dead-window clocks, feature statistics and per-source conversation counts.
2. With `HF_TOKEN` in the pod env, every checkpoint uploads to the private repo `RaphaelRaphaelRaphael/autolabs-3c-sae` under `runs/<run_id>/checkpoints/`, and a fresh pod with the same `AUTOLABS_3C_RUN_ID` downloads the latest one before training. No manual copying.
3. Resume skips the conversations already harvested from each source and offsets the shuffle seed, so the same prefix is not re-read.
4. Do not restart a stopped pod; terminate it, create a fresh one, set the same run id and `HF_TOKEN`, run `start.sh` again.
5. Prefer a data center with High stock for the chosen GPU (`get-capacity` or the console).
6. Still not built: the scheduled monitor (every 30 min: harness status + pod status via REST, relaunch from the last checkpoint on failure, alert). Until it exists, a laptop watcher or manual checks are the monitor.

Not yet validated on a GPU: the throughput changes (TF32, GPU-resident buffer, prefetch thread, incremental decode) and the HF checkpoint round trip. Run the smoke config once on a pod with `HF_TOKEN` set, kill the process after the first checkpoint, delete `checkpoints/` and rerun, and confirm the log says "restored checkpoint ... from hf://" before launching the full run.

## Launch procedure (fresh pod)

1. `POST /api/persona-3c/start` with manifestHash = sha256 of the chosen config file, budgetUsd 12, idempotencyKey.
2. Create pod via REST v1 (`gpuTypeIds`, `imageName runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 40 GB container, 80 GB volume at `/workspace`, port 22/tcp, env `PUBLIC_KEY` = contents of `~/.ssh/autolabs_runpod.pub`, `PIP_BREAK_SYSTEM_PACKAGES=1`, `HF_HOME=/workspace/3c/hf`).
3. Write `/workspace/3c/.env` with `AUTOLABS_3C_WORKER_URL`, `AUTOLABS_3C_TOKEN`, `AUTOLABS_3C_RUN_ID`, `AUTOLABS_3C_GIT_REF=<commit>` and `HF_TOKEN` (required for off-pod checkpoints and the gated WildChat source), LF line endings.
4. `curl` the pinned `runpod_start.sh` from GitHub, run `bash start.sh configs/full-paper.json` (owner's choice on 2026-09-17: the paper's Appendix M data mix at our 32k / k 40 / 150M parameters; LMSYS-Chat-1M terms must be accepted on the Hub by the token's account) under nohup. Log at `/workspace/3c/smoke.log`; a `[train] progress` line every 1M tokens with ETA.
5. On completion the pipeline reports `done`; copy `summary.json` and the reports off the pod and stop the pod.

Credentials: `research/experiment-003c/.env.runpod` (gitignored) holds the RunPod key, Worker URL and 3C token. SSH key `~/.ssh/autolabs_runpod`. Never print them.
