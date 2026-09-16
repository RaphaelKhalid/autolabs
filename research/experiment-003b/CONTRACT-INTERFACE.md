# Frozen worker contract interface

Generated from the frozen local blind pair artifact by `scripts/generate_worker_contract.py --write`.

Server-only module: `orchestrator-worker/src/persona-3b-contract.generated.ts`

Exports for worker integration:

- `PERSONA_3B_JUDGE_PROMPT`
- `PERSONA_3B_SCORE_SCHEMA`
- `PERSONA_3B_PROMPT_SHA256`
- `PERSONA_3B_SCHEMA_SHA256`
- `PERSONA_3B_BLIND_MEMBERSHIP_SHA256`
- `PERSONA_3B_FINAL_MANIFEST_HASH`
- `PERSONA_3B_BLIND_MEMBERSHIP`
- `validatePersona3BInput(input, { disagreementRecordIds })`
- `validPersona3BScore(value)`
- Contract counters: `PERSONA_3B_TOTAL_RECORDS`, `PERSONA_3B_PRIMARY_PAIR_COUNT`, `PERSONA_3B_REPEAT_RECORD_COUNT`, `PERSONA_3B_SYNTHESIS_RESERVED_CALLS`, `PERSONA_3B_RETRY_ADJUDICATION_RESERVE`, `PERSONA_3B_CALL_CEILING`, `PERSONA_3B_OUTPUT_TOKENS`

The membership map contains opaque pair IDs, scenario/A/B SHA-256 values, truncation flags, and repeat eligibility only. It contains no raw outputs, feature IDs, signs, or candidate names. Primary and disagreement inputs require canonical A/B orientation; repeat inputs require swapped A/B orientation and membership in the deterministic 117-record repeat set. Disagreement inputs additionally require the frozen review ID set.

Current generated manifest hash is recorded in the generated module and `scoring-manifest.json`; deployment must set `PERSONA_3B_MANIFEST_HASH` to that value and reject missing/mismatched environment configuration.