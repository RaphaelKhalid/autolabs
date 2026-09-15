# Experiment 3A — Unsupervised Persona Discovery

## Launch state

This is a locked research track, separate from the five public $10 protocols. The owner button queues one bounded Kaggle **development** run. It does not run the expensive confirmation phase, does not enable paid API calls, and does not claim that a discovered feature is a persona finding.

The default ceiling is 6,600 seconds (110 minutes) inside Kaggle’s 7,200-second notebook timeout. The current Afterlight contract uses two Tesla T4 GPUs, a pinned Qwen2.5-7B-Instruct revision, a pretrained layer-19 SAE, 1,024 discovery responses, and a 780-response development screen. These quantities are recorded in [launch-manifest.json](./launch-manifest.json) and must be hashed before a real run.

## What the experiment asks

Can a label-free activation scan find directions in a language model that behave like broad persona tendencies across held-out prompts, without first naming the trait and optimizing for it?

“Unsupervised” means the initial feature selection does not use a persona label, target trait, or hand-picked positive examples. It does not mean the whole project is free of researcher choices: model, layer, SAE, thresholds, dataset, prompt templates, steering direction, and screening scenarios are all choices and must be disclosed.

The output is a set of candidate directions and a development-screen report. It is not a claim that the model has a human personality, that the feature is a semantic unit, or that activation steering reveals a unique causal mechanism.

## Phase boundary

### Discovery

Run 1,024 pinned dataset responses through the base model and SAE. Select up to 32 candidate features using the registered occurrence, cosine, and norm criteria. Selection has no persona label.

### Development screen

Screen the candidate features on 12 neutral scenarios with positive and negative activation steering. The screen currently records 780 response units. It is a development stage for freezing behavioral definitions, prompts, and baseline comparisons. It is not confirmation.

### Confirmation — not enabled

The later plan would test at most three candidates over 600 scenarios, six conditions, and two repeats. This launch path deliberately cannot start that phase. It is not included in the one-click button, the 6,600-second budget, or the API budget.

## What the button does

The browser sends an owner-authenticated request to the Next.js control route. The server forwards a fixed `experiment-003a-v1`, `phase=development`, `maxRuntimeSeconds=6600` request to the orchestrator. The worker stores an idempotent launch request in D1. A separately configured owner-local relay claims that request and starts the private Kaggle notebook. Credentials remain on the owner machine; the Vercel page does not receive a Kaggle token or OpenAI key.

The queue is intentional. A Vercel or Cloudflare request should not be expected to hold open for a 110-minute GPU job. The relay is the durable hand-off between the one-click control and Kaggle.

## Required artifact record

The notebook must write:

- frozen manifest and SHA-256 hash;
- base-model and SAE revisions;
- dataset revision and seed;
- discovery and screen progress events;
- candidate feature IDs and selection metrics;
- positive/negative steering settings;
- raw response hashes and output truncation counts;
- per-stage elapsed time and GPU visibility;
- status `complete`, `partial`, `failed`, or `budget-safe-stop`;
- a clear statement that confirmation did not run.

The public page may show counts and hashes. Private raw outputs should be published only under the project’s consent and privacy policy. Synthetic research prompts must not request personal data.

## Interpretation contract

An apparent feature can arise from dataset artifacts, prompt format, token position, a correlated topic, SAE reconstruction error, steering magnitude, or the chosen screening prompts. A candidate that changes behavior is not automatically a stable persona direction. The development report must include neutral baselines, positive and negative steering, sign symmetry where possible, feature frequency, and a limitation section.

The correct conclusion after this bounded run is one of:

- development screen completed with candidates carried forward for a separately approved confirmation;
- partial screen with explicit missingness;
- failed or budget-safe-stopped run with no scientific conclusion.

## Safety and spend invariants

- `apiBudgetUsd` is zero in this manifest.
- The runner must not import an OpenAI key or call a paid provider.
- The owner relay must pass Kaggle’s timeout and enforce its own 6,600-second deadline.
- A notebook that exceeds the deadline is a failed or budget-safe-stopped operational run, not a successful result.
- The manifest cannot be edited after the launch request is queued.

## Reading

The project’s background reading is in [docs/events/literature.md](../../docs/events/literature.md). Particularly relevant are Representation Engineering ([arXiv:2310.01405](https://arxiv.org/abs/2310.01405)), Activation Addition ([arXiv:2308.10248](https://arxiv.org/abs/2308.10248)), RISE ([arXiv:2512.23988](https://arxiv.org/abs/2512.23988)), and Persona Vectors ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509)). These papers motivate design choices; they do not validate this exact model, SAE, or screen.

