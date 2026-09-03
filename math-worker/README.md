# Exact math worker

This job runs in a pinned Node container on GitHub Actions. It performs bounded,
deterministic bigint searches proposed by the five research agents and returns a
signed result to the append-only experiment ledger. Jobs may span several
research rounds; no LLM is trusted to evaluate a square or factorization.

Supported jobs:

- `divisor_completion`: exact factor-pair completion from two differences.
- `family_scan`: build candidate N values from every difference pair, then rank
  their exact support against the supplied family.
- `boundary_scan`: examine an uncovered, explicitly bounded difference range.

Every job clamps its total divisor checks and records the bound, completion flag,
timestamps, and exact candidates for reproduction.
