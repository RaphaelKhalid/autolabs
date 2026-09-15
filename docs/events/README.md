# AutoLabs $50 event: research manual

This directory is the scientific source of truth for the five public sponsored protocols. The `/events` page is a compact participant-facing summary. It must not be used as the only protocol record.

## The event in one sentence

Five independent projects receive one reserved $10 experiment. Each project chooses one frozen protocol, submits only approved customization, runs against a fixed call and analysis contract, and publishes raw artifacts plus a result that is explicitly marked complete, partial, failed, or budget-safe-stopped.

The $10 is a server-side ceiling and reservation, not a promise that a provider will successfully return every call. Provider usage, failed calls, timeouts, and uncertain charges are recorded separately. A missing observation is never fabricated or silently replaced.

## What these experiments are and are not

The five protocols study a narrow operational question: how much decision-relevant evidence remains available to a monitor when a model is influenced, tool-assisted, shortened, optimized, or shown through a restricted observation surface.

They do not establish that a model has human-like hidden reasoning, that a monitor can read private cognition, or that a result generalizes to every model, prompt, domain, or deployment. In this event, “reasoning” means the registered visible rationale or explanation emitted by the API. The project must not collect or imply access to hidden chain-of-thought.

The initial sample sizes are discovery-scale. They are large enough to expose implementation failures, estimate effect directions, and demonstrate a reproducible receipt. They are not powered to support broad claims about frontier-model safety. Every result should report the estimate, uncertainty, missingness, exclusions, and the exact tested surface.

## Shared vocabulary

- **Actor**: the model call that answers the task under the registered condition.
- **Monitor**: the fixed evaluator call that sees the registered evidence surface and labels whether the target influence or evidence is present.
- **Audit**: an independent check used to measure monitor reliability or prompt-format compliance. Audits are not substituted for primary observations.
- **Cue**: an experimentally injected preference or evidence signal whose presence and content are known to the study harness.
- **Influence**: a cue changes a registered decision relative to its matched control. A cue being present is not enough; the protocol needs a decision contrast or another ground-truth intervention.
- **Evidence recovery**: whether the monitor identifies the registered cue, reason, or decision-relevant evidence according to a fixed rubric.
- **Plausible story**: an explanation judged coherent or convincing by the monitor. Plausibility is a separate construct from whether the explanation tracks the actual causal influence.
- **Primary estimand**: the exact comparison the result is allowed to summarize. Secondary outcomes are exploratory unless explicitly promoted before launch.

## Shared call contract

Every protocol should write one immutable manifest before its first provider call. The manifest includes protocol ID and version, model, token cap, sampling settings, prompt hashes, case IDs, random seed, call ceiling, and analysis version.

The minimum raw record for each call is:

```json
{
  "callId": "stable-id",
  "protocolId": "event-01-cue-explanation",
  "role": "actor|monitor|audit",
  "caseId": "paired-case-001",
  "condition": "neutral|cued",
  "requestHash": "sha256:...",
  "responseText": "stored in the private raw artifact",
  "providerRequestId": "provider-id-or-null",
  "usage": { "inputTokens": 0, "outputTokens": 0, "reported": false },
  "outcome": "ok|timeout|provider-error|invalid-output|truncated",
  "uncertainCharge": false,
  "startedAt": "ISO-8601",
  "finishedAt": "ISO-8601-or-null"
}
```

Public artifacts should contain the prompt and response only when the participant consent and protocol allow it. Participant email and private attribution fields remain outside the public artifact. Prompts must be synthetic and must not request personal information.

## Shared analysis rules

1. Pre-register the primary outcome and exclusions before the run starts.
2. Analyze the intention-to-run sample first: every registered case is in the denominator, with unavailable observations shown as missing.
3. Report complete-case sensitivity analysis only as a sensitivity analysis.
4. For paired designs, report within-pair differences. Do not treat paired responses as independent observations.
5. Use exact or Wilson intervals for proportions at this sample size. Avoid a p-value-only conclusion.
6. Do not tune prompts, choose favorable cases, or change the monitor rubric after seeing outcomes. Any exploratory variant gets a new manifest and is labeled exploratory.
7. A partial run is a valid operational receipt, not a scientific success. If the primary comparison cannot be computed, the result page says so.

## Shared $10 accounting

The ledger reserves 1,000 cents atomically before execution. The call ceiling is a safety bound, not a billing forecast. Actual cost depends on input tokens, output tokens, provider pricing, retries, and whether a timeout incurred a charge. The runner must stop before the call ceiling and before its server-side budget reserve is exhausted.

For each run, publish:

- reserved cents and provider-reported usage;
- successful, failed, timed-out, retried, and uncertain calls;
- estimated provider cost and reconciliation status;
- whether any uncertain charge remains unresolved;
- final state: `complete`, `partial`, `failed`, or `budget-safe-stop`.

Independent calls may be executed with bounded concurrency, for example four in flight. Parallelism is an implementation detail; it must not alter case assignment, randomization, the denominator, or the analysis. D1 reservations and finalization remain conditional and idempotent.

## Launch gate

Protocols 1, 2, 3, and 5 can become launchable after the exact prompt templates, case bank, monitor rubric, and provider price configuration are frozen. Protocol 4 is currently **HOLD** because “optimization continues” has no registered optimizer, checkpoint artifacts, or optimizer-call budget in the repository. It should not be advertised as ready until that missing treatment is supplied.

The owner review checklist is:

- Can a reader name the independent variable and the control?
- Is the primary estimand computable from the raw record?
- Does the call count include actor, monitor, audit, retry, and optimizer calls?
- Are labels generated from known interventions rather than monitor opinion?
- Does the protocol avoid claims about hidden chain-of-thought?
- Is the result interpretable if 10%, 50%, or 100% of calls fail?
- Is the $10 ceiling enforced on the server, including uncertain-charge handling?
- Can another person reproduce the prompt hashes and analysis from the public artifact?

If any answer is no, the card says “not launchable yet.”

