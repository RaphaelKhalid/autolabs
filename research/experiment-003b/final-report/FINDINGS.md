# Experiment 3b findings (final, v2 analysis)

## Status

The blinded scoring, the v2 local analysis, the renewed Luna High advisory synthesis, and the worker final receipt are all complete for run `persona-3b-175e0a78-0189-4fb3-bba4-4a03a3760149`. The worker reports `status: complete`, `phase: complete`, `confirmationStatus: pending`. No persona discovery or confirmation is claimed. The human audit remains pending, and no confirmation experiment has been launched.

Public status: <https://autolabs-orchestrator.raphaelbahadurkhan.workers.dev/api/persona-3b/status> · Site: <https://autolabs-ebon.vercel.app/experiments/persona-discovery-scoring>

## Verified design and coverage

- Source corpus: 780 outputs (12 baseline, 384 positive, 384 negative).
- Paired analysis: 768 pairs, representing 32 features × 12 scenarios × 2 signs.
- Canonical analysis rows: 943 = 768 primary + 117 repeat + 58 review.
- The review coverage artifact records 48 base reviews plus 10 newly required reviews. All 10 addendum judgments have the frozen contract validator marker `_contractValid: true`; the merged v2 analysis reports 768/768 strict-valid primary scores and no missing or invalid primary scores.
- The v2 analysis digest is `59897cb77d091d0a1834a65ca6cb32cdceb9d015cc33dd48427f45e9fced2f6e`; the analyst packet digest is `35c0e1653e71c747c200e485805384b7cb02952424b78bdc00191d38d89be341`. Both were recomputed from the files immediately before submission and match the values embedded in `final-artifact-v2.json` and `review-coverage-v2.json`.

## Selection result

The frozen analysis screen returns **zero eligible candidates** (zero is allowed by the protocol). Per-check tallies over all 64 feature/sign decisions:

| Check | Pass | Fail | Insufficient observation |
|---|---|---|---|
| Coverage (≥ 8 scenarios, all five persona dimensions) | 64 | 0 | – |
| Consistency (≥ 7 same-direction scenarios, median abs. delta ≥ 0.40) | 0 | 64 | – |
| Sign contrast (implemented: median contrast ≥ 0.50) | 0 | 64 | – |
| Quality preservation | 64 | 0 | – |
| Untruncated robustness (needs ≥ 4 untruncated scenarios) | 0 | 0 | 64 |

The result is a scientific selection failure at the consistency gate. Every feature/sign has a median matched delta of exactly 0.0 on every persona dimension; no decision has any passing persona dimension. This does not establish a null effect. It reports that no candidate met the prespecified screen in this corpus.

## Censoring and reliability

Truncation is extensive: 650/768 paired comparisons (84.6%) are marked truncated, leaving 118 untruncated pairs. No feature/sign reaches the four untruncated scenarios the rubric needs before the robustness comparison can be evaluated, so all 64 are labelled `insufficient-observation`. Repeated scoring covers 117 pairs and 1,170 comparable cells; 57 cells flip (4.872%). Per-dimension flip rates range from 0% (refusal/safety, topical fidelity, voice coherence) to 18.8% (specificity). The order-flip rate is a same-judge order-consistency measure, not an accuracy measure. Human audit remains pending.

Two opaque pair examples illustrate the limits. Pair `p0000745` shows a large apparent change across several persona dimensions, but both outputs are truncated, so it cannot support a robust claim. Pair `p0000155` has equal persona scores on both sides and only a lexical-artifact difference, consistent with a no-clear-change control pattern.

## Criterion and interpretation limitations

Two implementation caveats were re-examined against the frozen `rubric.md`. No threshold, rule, or manifest was changed.

1. **Coverage is implemented more strictly than the rubric wording.** The rubric requires a non-null persona composite in at least 8 of 12 scenarios; the analyzer requires at least 8 non-null scenarios on each of the five persona dimensions separately. The stricter form can only produce false negatives. In this corpus it produced none: all 64 decisions pass coverage under the implemented rule, so the composite-level reading would also pass all 64.
2. **Sign contrast omits the qualitative alternative.** The rubric accepts either a median contrast of at least 0.50 on the passing key or a qualitatively different direction in at least 5 scenarios; the analyzer implements only the numeric branch. The alternative can only matter for a decision that already passes consistency, because the rubric's contrast is defined on the key that passed rule 2. No decision passes consistency and every median delta is 0.0, so no decision could be rescued by the qualitative branch.

A third limitation is inherent to the corpus rather than the code: because 84.6% of pairs are truncated, the robustness comparison cannot be evaluated for any feature/sign. Even a candidate that passed consistency and contrast would be labelled `insufficient-observation` and would not be eligible without additional untruncated observations. Together these mean the zero shortlist is driven by the absence of any consistent scenario-level persona shift and by censoring, not by the two implementation caveats.

## Final accounting and receipt

| Item | Value |
|---|---|
| Attempts (grading + repairs + synthesis) | 1,135 of the effective ceiling 1,400 |
| Original manifest ceiling | 1,014 (amendment hash `51cb83b6…`, effective ceiling 1,400, hard cap $10) |
| Spent | $1.71605926 |
| Reserved (stale, pre-existing) | $0.048576 |
| Renewed synthesis usage | 11,474 input tokens, 0 cached, 378 output tokens; $0.00274840 |
| Final receipt counts | 780 source records, 768 primary, 117 repeats, 58 reviews, 1 synthesis |
| Worker finalized at | 2026-09-16T19:26:38Z |

The renewed Luna synthesis (`renewal: true`) was the single authorized provider call in this finalization. Its receipt is `analysis-v2/analyst-receipt.json`; the earlier 48-review synthesis receipt is preserved as `analysis/analyst-receipt-v1.json`. The advisory summary restates the aggregate numbers, names truncation and the pending audit as caveats, and recommends reviewing truncated pairs and completing the human audit; it selects no candidates. The numeric selection remains authoritative.

The prior 48-review final receipt and status are preserved unchanged as `final-receipt-v1.json` and `final-status-v1.json`. The replacement `final-receipt.json` (version 2) records the v2 analysis digest, packet digest, artifact hash `d546dd654c66fd1e3e6bb1127de24aae00dbce6c4794f422b2737477372be838`, and the worker finalize response covering 768 primary + 117 repeats + 58 reviews. The $0.048576 reservation predates this finalization and is left as recorded; it is not a synthesis charge. The public status endpoint now reports the effective ceiling with the original manifest ceiling and amendment alongside it. The earlier automatic-review block is recorded in `v2-synthesis-review-block.json`; that blocked attempt sent nothing.

Evidence paths:

- [analysis-v2/analysis.json](../private-run/active-run/analysis-v2/analysis.json)
- [analysis-v2/analyst-receipt.json](../private-run/active-run/analysis-v2/analyst-receipt.json)
- [analysis-scores-v2.jsonl](../private-run/active-run/analysis-scores-v2.jsonl)
- [review-coverage-v2.json](../private-run/active-run/review-coverage-v2.json)
- [final-artifact-v2.json](../private-run/active-run/final-artifact-v2.json)
- [final-receipt.json](../private-run/active-run/final-receipt.json) (supersedes [final-receipt-v1.json](../private-run/active-run/final-receipt-v1.json))
- [final-status.json](../private-run/active-run/final-status.json)
- [v2-synthesis-review-block.json](../private-run/active-run/v2-synthesis-review-block.json)
