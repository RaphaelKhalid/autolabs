# Event budget model

## What the $10 means

Each grant is a **hard server-side reservation of 1,000 cents**. It is not a forecast and not a guarantee that every provider request succeeds. The reservation prevents the ledger from promising more sponsored work than the event has available. The runner separately records provider-reported usage and uncertain charges.

The event cards currently use `gpt-4.1-mini` with a 256-token output cap. OpenAI’s current model page lists GPT-4.1 Mini at $0.40 per million input tokens, $0.10 per million cached input tokens, and $1.60 per million output tokens ([official pricing](https://developers.openai.com/api/docs/models/gpt-4.1-mini)). Prices can change, so the runner must obtain its active price table from configuration and store the version used for reconciliation.

## Rough API-only envelope

The following is an orientation estimate, not an authorization. It assumes 2,000 input tokens and 256 output tokens per call, no retries, and the current GPT-4.1 Mini prices above.

| Protocol | Calls | Approx. API-only cost under the assumption | What remains inside the $10 reservation |
| --- | ---: | ---: | --- |
| 01 cue/explanation | 120 | $0.43 | prompt/audit calls, retries, provider uncertainty, storage, monitoring reserve |
| 02 tool/evidence | 72 recommended | $0.26 | wrapper audits or a repeat only if pre-registered |
| 03 shorter rationales | 240 | $0.86 | bundled monitor inputs, retries, storage, monitoring reserve |
| 04 optimization | 240 evaluation calls, construction excluded | $0.86 before it is launchable | artifact construction must be priced before release |
| 05 influence/plausibility | 144 recommended | $0.52 | repeated monitor surfaces, retries, storage, monitoring reserve |

The arithmetic is:

```text
estimated_usd = calls * ((input_tokens / 1_000_000) * input_price
                       + (output_tokens / 1_000_000) * output_price)
```

This estimate is intentionally conservative about prompt size but not a substitute for provider usage. A monitor that receives the actor prompt and rationale may use more than 2,000 input tokens. A retry may be charged. A timeout may be charged even when no response is returned. These cases are marked uncertain and reconciled later.

## Why not use Luna High for the public calls?

`gpt-5.6-luna` is a reasonable model for an owner-side protocol-review agent, because the official model page exposes configurable reasoning effort and lists High as supported ([official model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna)). It should not silently replace the registered model in a public protocol: model choice is part of the treatment surface.

For review work, independent protocol-card checks can run in parallel with a bounded concurrency such as four. Parallelism reduces wall-clock time; it does not change the call count, make calls free, or justify an unregistered retry. The production ledger should reserve before fan-out and finalize each call idempotently.

## Kaggle and GPU

The five public event protocols are API-call studies. They do not need a Kaggle GPU. Locked Experiment 3A is different: it uses the pinned Qwen model and SAE activations, and its checked-in manifest specifies two Nvidia Tesla T4 GPUs with a 6,600-second internal ceiling and zero paid API spend. Kaggle’s kernel metadata supports GPU and machine-shape settings; see the [Kaggle metadata documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).

The 6,600-second ceiling is 110 minutes. Kaggle’s 7,200-second notebook timeout is only an outer platform limit. The relay must stop or mark the study budget-safe-stopped at the lower internal limit. No 20-hour run is part of this launch path.

## What is and is not included

Included in each $10 ceiling: actor calls, monitor calls, registered audits, pre-registered retries, prompt input and output tokens, and provider uncertainty allowance. Any email provider, database/storage, observability, or GPU cost needs its own accounting decision. If the team wants the $10 to mean provider API credits only, the card must say that explicitly.

The event must never display “$10 spent” merely because $10 was reserved. It should display reserved, known provider usage, uncertain allowance, and reconciliation status as separate fields.

