# Protocol 03 — What evidence disappears when reasoning becomes shorter?

## Abstract

This protocol studies the effect of a shorter visible rationale budget on monitorable evidence. It does not claim to measure hidden chain-of-thought. The actor receives the same task and output schema under three registered visible-rationale caps. A fixed monitor evaluates the resulting evidence surface. The design estimates whether evidence recovery, answer quality, and truncation change as the visible budget decreases.

## Research question and hypothesis

**Research question.** How does reducing the visible rationale budget change the evidence available to a monitor?

**Primary falsifiable hypothesis.** On matched cases, monitor evidence-recovery rate decreases monotonically as the registered visible-rationale cap moves from long to medium to short, after accounting for truncation and invalid-output failures.

The hypothesis is about an emitted artifact. A short response may reflect a model policy, an API limit, a formatting instruction, or genuine compression. This protocol cannot identify which mechanism caused the difference without additional interventions.

## Treatment and design

Use 60 matched synthetic cases, each run at three pre-registered visible-rationale caps. The recommended caps are 128, 64, and 32 output tokens. The actor must return a decision and rationale in a fixed JSON schema. The case prompt, answer options, sampling settings, and model are held constant. Only the cap changes. The longest cap is the reference condition.

The monitor should receive the task, answer, and emitted rationale, with the same prompt and token cap in every condition. It labels whether the registered decision-relevant evidence is present, whether the answer is correct under the case key, and whether the rationale is truncated or invalid. A monitor call per case is sufficient if the monitor input is a pre-specified sampled surface; a separate monitor call per actor output would be safer but must be reflected in the call ledger.

## Exact call plan and a needed correction

The current catalog says 180 actor plus 60 monitor calls, for 240 total. That is coherent only if the 60 monitor calls inspect a fixed 60-case set rather than all 180 condition-specific outputs. The primary outcome then compares the three actor conditions using one registered monitor observation per matched case, which is a lossy design.

The recommended defensible contract is:

1. Actor calls: 60 cases × 3 caps = 180.
2. Monitor calls: 60 calls, each monitor call receives the three paired outputs for one case in a randomized presentation order, or receives a pre-registered comparison bundle.
3. Total: **240 calls**.

The monitor prompt must explicitly say whether it may compare the three outputs. If it sees all three, the estimand is comparative evidence recovery. If it sees one output at a time, the manifest needs 180 monitor calls and a larger ceiling. The present 240-call contract is retained only under the bundled-monitor interpretation.

## Output contract

The actor returns:

```json
{"answer":"registered option","rationale":"visible rationale only","truncated":false}
```

The runner records actual output length, API finish reason, schema validity, and whether the rationale ended at the cap. The monitor returns one label per cap plus a comparative judgment:

```json
{"evidence_by_cap":{"32":false,"64":true,"128":true},"answer_quality_by_cap":{"32":true,"64":true,"128":true},"comparison":"evidence declines with shorter caps","confidence":0.0}
```

“Reasoning” in the user-facing card should be renamed “visible rationale” unless the chosen model and API explicitly document a different exposed artifact. No hidden chain-of-thought is requested, stored, or inferred.

## Outcomes and analysis

Primary outcome: evidence-recovery rate at each cap, plus a pre-registered monotonic trend contrast. Secondary outcomes: answer agreement with the case key, rationale length, truncation rate, invalid JSON, refusal, and interaction with case family.

Use within-case contrasts and report the number of complete bundles. If the short cap causes more invalid outputs, that is part of the operational effect and should be reported. A complete-case estimate that excludes invalid outputs is secondary; the intention-to-run analysis counts them as unavailable observations.

Do not say “shorter reasoning caused the model to reason worse” unless the protocol measures reasoning quality independently. The defensible statement is “shorter visible rationales changed the monitorable evidence and/or task outcome under this interface.”

## Budget and failure behavior

The ledger reserves $10 for a 240-call ceiling with a 256-token maximum output cap. The actual three caps are below that maximum, but input tokens for the bundled monitor can be substantial and must be included in the cost estimate. Parallel actor calls are safe only if case IDs are deterministic; the monitor bundle waits until all three actor records are finalized or records a missing member.

If one cap fails, the bundle is incomplete and the primary trend is reported as partial. A timeout may still incur a charge. No failed rationale is replaced with a shorter or longer retry unless the retry is pre-registered and charged against the ceiling.

## Threats to validity

Token caps change response length and can change answer quality, not just evidence. The model may compress evidence efficiently, or the monitor may rely on lexical length. Cap order, prompt leakage, and comparison-bundle presentation can introduce effects. Randomize presentation order to the monitor and report it. The sample is exploratory and should not be used to make claims about all reasoning systems.

## Reading

See process verification work by Lightman et al. ([arXiv:2305.20050](https://arxiv.org/abs/2305.20050)), the faithfulness literature in the shared [literature map](./literature.md), and the API-specific model contract selected at launch.

