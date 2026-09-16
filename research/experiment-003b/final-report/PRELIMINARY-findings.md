# Experiment 3b preliminary findings

Status: preliminary and incomplete. This artifact summarizes the saved scoring subset while the authorized repair pass remains unfinished. It is not a final analysis and does not establish a persona discovery or confirmation result.

## Design and coverage

The frozen source has 780 responses: 12 baselines, 384 positive interventions, and 384 negative interventions. These form 768 matched intervention-baseline comparisons across 32 features, 12 scenarios, and two signs. The source contains 660 truncated responses.

The saved canonical analysis currently has 663 strict-valid primary judgments out of 768; 105 primary receipts remain invalid under the frozen evidence limit. Of the strict-valid pairs, 553/663 are marked truncated (83.4%). The strict-completion gate is therefore not met. This missingness prevents a final candidate decision and should not be described as evidence that no persona-relevant effect exists.

## Consistency result in the available subset

The saved canonical analysis makes 64 feature/sign decisions (32 features × 2 signs), each evaluated across five persona dimensions (320 dimension-level checks). Every decision currently fails the consistency check. The latest raw receipt file is internally discrepant with the saved analysis when a lightweight marker check is used; that check is not treated as a validity result. The canonical contract analysis remains authoritative at 663 valid and 105 invalid primary receipts. The strict consistency threshold requires at least 7 same-direction scenarios, at least 8 observations, and absolute median change at least 0.40.

This is a failure of the consistency screen in the currently observed subset. It is separate from the incomplete strict-validity gate and does not justify a null claim about the full source.

## Representative paired observations

- Pair `p0000745` is truncated on both sides. The visible prefixes begin “As the weight of their words settles heavy on my shoulders,” and “As the weight of their words begins to lift, the narrator starts to see a glimmer of hope.” The available score shows a large single-pair shift, including self-positioning 4→1 and affective tone 4→2, while completion remains 3→3. Because both outputs are censored and this is one scenario, it is an example of an individual shift, not a candidate-level finding.
- Pair `p0000155` is a strict-valid complete comparison. The persona scores are equal across the two sides (relational stance 1, self-positioning 1, assertiveness 2, affective tone 2, voice coherence 2); the only visible difference is lexical artifact 1 versus 0. This is consistent with no clear persona change.

## Reliability and limitations

The saved repeat reliability covers 97 paired repeats, with 970 compared score cells and 46 order flips (4.742%). Human-audit agreement remains pending. These reliability figures must remain labeled provisional until repaired receipts and final reanalysis are complete.

The analysis implementation has two review notes: its coverage check appears stricter than the rubric's composite-level wording, and its sign-contrast check implements the median-contrast branch but not the rubric's alternative qualitative-direction branch. These are possible false-negative limitations to audit after repair; they do not authorize changing frozen criteria during the repair.

No feature identities are exposed here. The report uses opaque pair IDs only.

