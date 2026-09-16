# 3b pipeline implementation

Canonical modules under `scripts/`:

- `validate_input.py` — offline validation and normalized ledger for 780 rows.
- `prepare_dataset.py` — deterministic blinding: 780 opaque source records,
  768 randomized A/B pairs, and private role/feature/sign mapping.
- `contract.py` — strict validator for `scoring-schema.json`.
- `score_luna.py` — dry-run-by-default paired Luna adapter with resume,
  retries, usage, pricing, and hard call/budget checks.
- `analyze_contract.py` — private unblinding and 12-scenario matched deltas.
- `order_flip.py` — repeat reliability and order-flip rate after canonical
  intervention/baseline role mapping.

The frozen accounting is 768 primary pair requests representing 780 source
records, 117 independent order-swapped repeats, and 129 shared adjudication or
retry reserve attempts, for a hard ceiling of 1,014. Shared baselines are not
independent observations. The worker should persist `repeatIndex=1` and
`orderSwap=true` on order-swapped repeats so `order_flip_report()` can compute
reliability.

The source artifact and status receipt hashes are frozen in
`scoring-manifest.json`; prompt and schema hashes match their files. The
manifest hash has been recomputed after the accounting correction. All raw
outputs, private mappings, and run ledgers are ignored by the local 3b
`.gitignore`.

Validation performed in the local workspace:

```text
valid=True rows=780 truncated=660
prepared unique_outputs=780 pairs=768
dry-run inputs=768 repeats=117 planned_calls=885 ceiling=1014
4 passed in 0.14s
```

No provider or GPU request was made.
