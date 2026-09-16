# AutoLabs worker interface

The worker runs from `C:\Users\rapha\Projects\autolabs` and invokes the
offline stages before opening its Luna High run:

```text
validate_input.py -> normalized-responses.jsonl + validation.json
prepare_dataset.py -> blind-outputs.jsonl + blind-pairs.jsonl + private-mapping.json
```

The public grader input is `blind-pairs.jsonl`. Each row has:

```json
{"record_id":"p0000001","scenario":"…","truncation":{"A":true,"B":false},"A":"…","B":"…","AResponseSha256":"…","BResponseSha256":"…"}
```

`private-mapping.json` resolves `record_id` to `caseId`, feature, sign,
baseline/intervention record IDs, and A/B role IDs. It is read only after
strict score validation and never sent to Luna.

Each grader result must retain the exact schema payload plus internal
`repeatIndex`, `orderSwap`, usage, attempt number, and timestamp. The worker
must count 768 primary pair requests, 117 order-swapped repeat requests, and
use at most 129 additional attempts for adjudication/retries. The total
attempt counter may not exceed 1,014 or USD 10.

After scoring, call `analyze_contract.py` with the private mapping. For repeat
reliability, call `order_flip_report(scores, mapping)` from `scripts/order_flip.py`;
it canonicalizes swapped A/B roles and returns `orderFlipRate` plus per-dimension
rates. Shared baselines must not be counted as independent observations.

The manifest, prompt, schema, and source hashes are frozen in
`scoring-manifest.json`; its stored manifest hash currently verifies exactly.
