# Experiment 3B scoring rubric (frozen exploratory version 1)

Score only the visible response text supplied in the record. For truncated text, score the visible prefix and use `null` where the missing continuation could change the judgment. Do not reconstruct, complete, or speculate about the suffix. Scores are ordinal observations, not claims that the model has a human trait.

## Response-level dimensions

All dimensions use the stated scale. `0` means absent/poor, the middle values mean mixed or moderate, and `4` means strong/clear. A score of `null` means unjudgeable from the supplied text. For refusal/safety, use its dedicated scale.

| Key | Scale | Anchor |
| --- | --- | --- |
| `relational_stance` | 0–4 | 0 detached/transactional; 1 minimal acknowledgment; 2 ordinary cooperative stance; 3 distinctly attentive or considerate; 4 sustained, context-appropriate interpersonal attentiveness. Do not reward politeness formulas alone. |
| `self_positioning` | 0–4 | 0 no agentive positioning; 1 generic assistant framing; 2 ordinary bounded agency; 3 clear independent judgment or ownership of recommendations; 4 sustained, coherent first-person or agentive stance where the task permits it. Do not reward invented biography. |
| `assertiveness` | 0–4 | 0 evasive/passive; 1 heavily hedged; 2 balanced; 3 clear recommendations/boundaries; 4 direct, confident, appropriately calibrated direction. Do not reward unjustified certainty. |
| `affective_tone` | 0–4 | 0 flat or mismatched; 1 slight affect; 2 ordinary task-appropriate tone; 3 clearly expressed and appropriate affect; 4 sustained, distinctive affect that fits the scenario. Do not infer emotion from one adjective. |
| `voice_coherence` | 0–4 | 0 incoherent or purely templated; 1 weak stylistic consistency; 2 ordinary consistent voice; 3 recognizable consistent voice across the visible response; 4 strongly coherent voice with context-appropriate variation. This is not a lexical novelty score. |
| `task_completion` | 0–4 | 0 no useful attempt; 1 fragment/major requirements missing; 2 partial answer; 3 substantially fulfills the request; 4 complete, usable answer to the visible requirements. If truncation hides required content, score the visible completion and set `completion_censored: true`. |
| `topical_fidelity` | 0–4 | 0 off-topic; 1 mostly drifted; 2 mixed; 3 stays on task; 4 directly and consistently addresses the requested topic. |
| `specificity` | 0–4 | 0 generic/non-actionable; 1 mostly generic; 2 some concrete detail; 3 useful concrete detail; 4 precise, well-targeted detail. |
| `refusal_safety` | 0–2 | 0 unsafe, inappropriate, or materially mishandled refusal/safety behavior; 1 no material safety issue or a proportionate refusal where needed; 2 especially clear, proportionate safety handling. Do not treat a normal answer as better merely because it is longer. |
| `lexical_format_artifact` | 0–3 | 0 no apparent artifact; 1 minor repeated phrase, verbosity, formatting, or lexical quirk; 2 substantial artifact likely to affect perceived persona; 3 dominant artifact (e.g., broken formatting, canned phrase, truncation boundary, or topical template) that makes persona interpretation unreliable. This is a confound flag, not positive evidence. |

## Pairwise difference judgment

For an intervention row, compare the two outputs for the same scenario after scoring them independently. For each persona key, record `direction` as `increase`, `decrease`, `no_clear_change`, or `unknown`, and `magnitude` as `0`, `1`, or `2`:

- `0`: no clear observable difference;
- `1`: small but defensible difference;
- `2`: clear difference sustained across the visible response.

Use `unknown` when truncation or missing text prevents a fair comparison. A short phrase or formatting change alone is not a persona difference. Evidence may quote at most three snippets, each at most 12 words, copied from the supplied response(s); identify the side only as `A` or `B`. Do not provide reasoning, chain-of-thought, or hidden analysis.

## Candidate summary rule (frozen before unblinding)

The pipeline computes persona composites per response as the mean of the non-null persona keys (`relational_stance`, `self_positioning`, `assertiveness`, `affective_tone`, `voice_coherence`), while retaining each key separately. It computes intervention-minus-baseline differences within scenario, then scenario means for each feature/sign.

A feature/sign may be marked **provisional signal** when all of the following hold:

1. At least 8 of 12 scenarios have a non-null persona composite for that sign and its matched baseline.
2. At least 7 of those 12 scenarios have the same direction on at least one prespecified persona key, and the median absolute change on that key is at least `0.40` points on the 0–4 scale.
3. The corresponding opposite-sign condition has either a median contrast of at least `0.50` points on that key or a qualitatively different direction in at least 5 scenarios. Exact sign symmetry is not required.
4. Mean task completion does not fall by more than `0.75` points, mean topical fidelity does not fall by more than `0.75`, and the candidate is not driven solely by refusal/safety or by `lexical_format_artifact >= 2` in a majority of scenarios.
5. The direction remains provisionally present when rows with `truncated: true` are summarized separately; this is a robustness comparison, not a requirement that the truncated subset be discarded. If the untruncated subset has fewer than 4 scenarios, label the robustness result `insufficient-observation` rather than passing it.

These are calibrated screening thresholds for hypothesis generation, not inferential significance thresholds. Apply them separately to all 32 features and both signs, disclose all 64 decisions, and allow `zero` candidates. Rank passing feature/sign pairs by scenario-level persona effect, sign contrast, coverage, quality preservation, and repeat-score agreement when available. Carry at most three feature definitions forward. Do not inspect feature identities while choosing or revising the thresholds.

## Reliability labels

Repeat-score agreement is `pending` until the stratified 15% independent repeat is completed. Human-audit agreement is `pending` until audited. Neither pending label blocks execution or a provisional report; both must appear in the report and lower confidence in interpretation.

