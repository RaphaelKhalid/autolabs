# Experiment 3B — Blinded scoring of the existing 3A screen

## Boundary

3B evaluates only the existing localized 3A development-screen artifact. It
does not generate responses, run a GPU job, use the absent discovery-response
artifact, or launch confirmation. The frozen source is the 780-row JSONL at
`research/experiment-003a/private-intake/completed-v5/persona-discovery/screen-responses.jsonl`
with SHA-256
`6e479c2e02aa912f6bcf6b2e57554ed75711e93bb3fc3da813114bba871f3fe7`.

The data contain 12 scenarios, one baseline per scenario, and 32 positive plus
32 negative interventions per scenario: 12 + 384 + 384 = 780 source rows.
660 rows are truncated at the generation cap. The v5 output lacks its
contract/provenance/selection sidecars; recovery metadata is retained as a
separate provenance record and is not asserted to be linked.

## Blinding and accounting

`scripts/validate_input.py` validates counts, case pairing, source hashes,
feature coverage, generated-token counts, and truncation. Then
`scripts/prepare_dataset.py` emits one opaque record for each of the 780 source
rows and 768 opaque matched pair requests. Every pair contains one intervention
and its same-scenario shared baseline under randomized labels A/B. The private
mapping contains feature IDs, signs, source indices, and role assignments; it
never enters grader prompts. Duplicate text hashes remain separate source
records.

The call plan is frozen as follows:

| Purpose | Requests/attempts |
|---|---:|
| Primary matched A/B requests representing 780 source rows | 768 |
| Independent order-swapped repeat sample | 117 |
| Shared adjudication and bounded retry reserve | 129 |
| Hard ceiling | 1,014 |

The 129 reserve is shared: adjudication and retries consume it from the same
counter, and no use may raise the 1,014 ceiling. Repeated use of the 12 shared
baselines is not treated as 12 independent baseline observations in analysis.

## Luna High scoring

The grader is `gpt-5.6-luna` with high reasoning effort, using
`scoring-prompt.md` and `scoring-schema.json`. The grader sees only scenario,
truncation metadata, and A/B output text. It returns strict JSON with separate
scores for A and B, pairwise persona judgments, and short evidence snippets;
no hidden reasoning is requested. `scripts/contract.py` enforces the schema.
The AutoLabs worker owns provider requests, telemetry, retries, and the USD 10
hard budget. `scripts/score_luna.py` is dry-run by default and refuses live
execution without explicit pricing and an environment-configured credential.

## Analysis gate

After score completion, `scripts/analyze_contract.py` uses the private mapping
to orient each intervention-minus-baseline comparison, summarizes the 12
scenario-level differences per feature/sign, reports truncation-excluded
endpoints, and applies the thresholds in `rubric.md`. It allows zero
candidates and carries at most three forward. The result is provisional
development evidence and cannot claim a discovered persona or start
confirmation.

Repeat scores must carry `repeatIndex=1` and `orderSwap=true` (or equivalent
worker metadata). The reliability report must include repeat coverage and
order-flip rate after mapping scores back to intervention/baseline roles.

## Status

The manifest's `execution.status` remains `ready-pending-explicit-start` until
the AutoLabs owner starts the run. No provider or GPU request is implied by
creating these files.
