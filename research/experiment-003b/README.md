# Experiment 3b — Existing-screen behavioral scoring

This local pipeline scores only the completed Experiment 3A development
screen. It consumes 780 existing responses (12 scenarios × [1 baseline + 32
positive + 32 negative]); it does not generate responses or run a GPU job.

Run from `C:\Users\rapha\Projects\autolabs`. The raw intake is the ignored
local copy under `research/experiment-003a/private-intake`; no OneDrive path is
used.

## Stages

1. `validate_input.py` checks counts, case pairing, source hashes, feature
   coverage, generated-token counts, and truncation. The source has 660/780
   truncated rows.
2. `prepare_dataset.py` emits 780 opaque source records and 768 randomized A/B
   matched pairs. Each pair compares an intervention with its same-scenario
   shared baseline. `private-mapping.json` keeps feature IDs, signs, and role
   assignments private; duplicate text hashes remain separate records.
3. `autolabs_relay.py` is the only live controller. It authenticates to the
   AutoLabs worker, starts/claims shards, submits A/B payloads, persists worker
   score receipts and telemetry, and never calls OpenAI directly. It is
   dry-run by default. The worker owns the USD 10 budget guard and telemetry.
4. `analyze_with_reliability.py` joins validated scores with the private
   mapping, computes matched scenario-level differences, truncation endpoints,
   selection decisions, and order-flip reliability.

`score_luna.py` is deliberately a safe compatibility command: it performs a
plan-only check and rejects `--execute`. This prevents stale local code from
bypassing AutoLabs. Use `autolabs_relay.py --execute` only after the owner has
reviewed the worker run and authenticated relay environment.

## Frozen accounting

The call plan is 768 primary matched pair requests representing 780 source
records, 117 independent A/B order-swapped repeats, and a shared 129-attempt
adjudication/retry reserve. The hard ceiling is 1,014 attempts. Shared
baselines are never treated as independent baseline observations.

## Dry-run

```powershell
python research/experiment-003b/scripts/validate_input.py --responses research/experiment-003a/private-intake/completed-v5/persona-discovery/screen-responses.jsonl --cases research/experiment-003a/private-intake/recovery-dataset/persona-discovery/cases.json --selection research/experiment-003a/private-intake/recovery-metadata/selection.json --out-dir research/experiment-003b/run/validated
python research/experiment-003b/scripts/prepare_dataset.py --normalized research/experiment-003b/run/validated/normalized-responses.jsonl --out-dir research/experiment-003b/run/blind
python research/experiment-003b/scripts/autolabs_relay.py --pairs research/experiment-003b/run/blind/blind-pairs.jsonl --mapping research/experiment-003b/run/blind/private-mapping.json --manifest research/experiment-003b/scoring-manifest.json --out-dir research/experiment-003b/run/relay
```

The manifest, prompt, schema, and source hashes are frozen. No credentials are
stored in the repository, prompts, manifests, logs, or result artifacts.
