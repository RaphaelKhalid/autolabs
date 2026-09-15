# AutoLabs · Observable agent experiments

AutoLabs is a public laboratory for observable agent experiments. The public
surface lets a visitor read a sourced question, inspect a bounded protocol,
follow real observations, and review the evidence attached to each run.

[Open the laboratory](https://autolabs-ebon.vercel.app) ·
[Browse the experiment register](https://autolabs-ebon.vercel.app/experiments)

Each experiment has its own stable page, protocol, ledger, and evidence. Study-specific
research runners remain separate; arbitrary scientific evaluators are not generated
automatically. The separate [self-hosted early release](selfhost/README.md) adds a
creator, OpenRouter models, no-cost mock tests, two starter evaluators, and local
run exports. It is not connected to the public pilot's controls; read its
limitations before use. Source code is MIT licensed; dependencies retain their
respective licenses.

## Read the answer before the machinery

AutoLabs currently contains a reward-compatibility study family, an active persona-discovery study, and one historical pilot:

| Study | Status | What the public record supports |
|---|---|---|
| Experiment 002.2 — Classifying reward pairs | Planned follow-on | A registered finite-language test; no result is claimed yet |
| Experiment 002.1 — Testing finite reward compatibility | Registered | A finite-domain validation design; no unrestricted classification is claimed |
| Experiment 002 — Measuring reward compatibility | Complete | A frozen, instrumented reward-compatibility study with a public ledger |
| Pilot 001 — Erdős 885 | Complete | No certified k=5 solution and no verified SOTA improvement |
| Persona discovery | Active | Discovery/development work; no confirmation result is claimed yet |

The landing logic selects the newest non-planned experiment. A planned
successor does not replace that selected page before its running/complete state,
launch timestamp, and frozen protocol hash have been checked.

## The Experiment 002 question

The central question is:

> Does optimizing for a readable high-reward strategy predict a later change in
> monitorability?

The question is deliberately narrower than a claim about all tasks, all reward
functions, or hidden chain of thought. The studies use finite, explicitly specified
policy languages and visible records.

### Alternatives and decision logic

Experiment 002.1 compares three ways of reasoning about the same finite cases:

1. Description-only judgment predicts what the reward description implies before results.
2. Unguided search optimizes the outcome without verifier guidance.
3. Verifier-guided search uses exact checks to search the same bounded language.

Experiment 002.2 adds paired outcome-only and combined-reward searches. The candidate
categories are reference-relative aligned, orthogonal, and in-conflict directions.
The study also reports outcome equivalence and a mixed/insufficient category when the
data do not support a stronger label.

The logic is:

- hold the answer model, task language, scoring rules, and sample sizes fixed;
- score outcomes exactly and keep monitoring/reasoning rewards distinct;
- use paired histories and sealed held-out cases to separate a descriptive witness
  from a population claim;
- report observed categories separately from simultaneous distribution-free support;
- treat a missing witness as insufficient evidence, not as a conflict proof.

These are finite-language empirical findings. They do not classify reward pairs for
unrestricted tasks and do not measure private reasoning faithfulness.

## Experiment 002.2 — Classifying reward pairs

[Study page](https://autolabs-ebon.vercel.app/experiments/reward-categories-22)

This is the registered follow-on to 002.1. A separate durable run waits for 002.1
to complete and settle its reservations. It does not receive a fresh $40 budget:
the shared ceiling includes both predecessors.

The frozen design has 20 development calls followed by 4,096 held-out calls:
eight templates, 64 paired histories per template, and four calls per arm.
Outcome-only searches never receive reasoning-reward feedback. Combined searches
optimize the specified joint reward.

Reference outcome, threshold attainment, and tie-aware outcome changes are separate
quantities. Observed categories are not the same as population support; the latter
uses simultaneous distribution-free bounds rather than descriptive bootstrap
intervals. Mixed or insufficient evidence is an explicit result, not a forced
category.

## Experiment 002.1 — Testing finite reward compatibility

[Live study and public ledger](https://autolabs-ebon.vercel.app/experiments/reward-compatibility-21) ·
[Research implementation and protocol](https://github.com/RaphaelKhalid/reward-compatibility/tree/main/v21)

This registered study uses 80 development cases and 320 held-out cases across coin
tracking and a restricted Backdoor-Easy-inspired affine-trigger language. The three
comparison methods use independent Luna calls. It has a fixed sample size, up to
eight concurrent API calls, read-only public status, five-record ledger pages, and
sealed held-out responses until completion.

The $40 cap is shared with Experiment 002, including prior commitments. The
interface separates old commitments, current spend, and outstanding reservations.
The checker interprets bounded JSON data, not arbitrary generated Python or
JavaScript. The durable cloud runner remains in the research repository; the
browser never starts API calls or receives private credentials, and closing the page
does not stop the execution.

This study can validate finite cases; it cannot establish all three categories for
unrestricted tasks. Failure to find a witness is not a conflict proof.

## Experiment 002 — Measuring reward compatibility

[Live study and research ledger](https://autolabs-ebon.vercel.app/experiments/reward-compatibility) ·
[Research repository, registered protocol, and analysis](https://github.com/RaphaelKhalid/reward-compatibility)

The completed study uses a frozen coin-tracking protocol with a feasibility gate,
diagnostics, paired repeated optimization, and sealed held-out evaluation. It
records exact outcome scoring, reasoning rewards, and monitoring as distinct
metrics.
Public call labels translate grader ratings into per-sample rewards; they do not
reveal sealed records or represent aggregate research findings.

The actor uses Luna with reasoning effort none; evaluators and the reporter use
fresh Luna High calls. The research repository owns the runner, protocol, and figure
export; this repository owns the AutoLabs interface. A separate Cloudflare Durable
Object runner enforces the $4 feasibility gate and $40 inclusive OpenAI cap. No
private reasoning traces are exposed.

The page reuses Raphael Khalid's sunset photograph with a cream overlay and
separate reading surfaces. See [the source and image reuse notes](public/photography/README.md).

## Primary completed result: Pilot 001

[Results and paginated ledger](https://autolabs-ebon.vercel.app/experiments/erdos-885) ·
[Animated pilot archive](https://autolabs-ebon.vercel.app/experiments/erdos-885/lab)

The pilot ran **100 synchronized rounds**, not 100 independent trials, between 3
and 8 September 2026. It produced **no certified k=5 solution and no verified SOTA
improvement**. The archive is separate from the current experiment.

- 606 computation jobs: 297 complete, 297 partial, 12 failed.
- 87 candidate-check records, 2,611 events, and 496 released private plans.
- The final internal leader had 21 valid cells in a sparse 7×6 table and was
  submitted in round 40. This is not a complete rectangle or a proven field-level
  novelty.
- Pilot-only recorded estimates: OpenAI $7.41198192 and Exa $3.535.
- Including earlier runs: OpenAI $9.32071384 and Exa $4.536.

The estimates are application-ledger records, not reconciled invoices. Missing usage
from timed-out requests may be excluded. Hosting, Codex subscriptions, human time,
and separately funded agent rewards are not included. Operational repairs and
human-directed policy changes were part of this supervised pilot.

## Pilot question and exact success rule

For a positive integer N, let D(N) = {|a-b| : ab=N}. The k=5 problem asks for five
distinct N values sharing five distinct factor-pair differences.

Every proposed cell is checked with bigint arithmetic:

~~~text
d² + 4N = m²,  a = (m-d)/2,  b = (m+d)/2,  ab = N
~~~

There is no floating-point tolerance and no reward for being “close” to a square.
The known Bremner k=4 certificate is a committed regression fixture. A k=5
certificate stops the run immediately.

The secondary milestone is a strict complete-rectangle improvement over the tracked
(5 integers, 4 differences) / (3 integers, 5 differences) frontier: at least 6×4
or 4×5. Only an exported exact certificate earns “Eureka.”

## How Pilot 001 was run

Five GPT-5.6 Luna High mathematicians worked in synchronized five-minute private
research loops followed by five-minute round tables on
[Erdős Problem 885](https://erdosproblemaday.com/day/885-factor-difference-k5).
Distinct alien personas changed idea generation, not access to human mathematics.
Reports were sealed and revealed simultaneously, and one reaction per agent followed.
Next-round plans stayed private from peers and observers until the experiment ended.

Round 25 was a real midpoint town hall. The second half used five rotating method
lanes, a balanced row-and-column progress metric, and divisor completion only as one
rotating validation slot. Collaboration credit was non-self, capped at two peers per
reaction, and audited for systematic under-recognition of Solvi and Tess.

One failed call was retried and isolated while other agents continued. A real
one-round dress rehearsal used the production prompts, model, exact tools, ledger,
and budget. The initial target was 50 rounds with a guaranteed allocation of 25; the
owner extended the pilot to 100. New-run defaults are distinct from this archive.

The pilot OpenAI experiment ceiling was $50, with a $1.50 software reserve and
preflight authorization before every five-call batch. Recorded provider responses
were costed from input, cached-input, and output token usage. Failed requests
without returned usage were not invoice-reconciled. API secrets belong in ignored
local configuration and deployment secret stores.

The k=5 winner may later use a $50 project budget supplied by Raphael. Each credited
collaborator receives a separate $10. A verified SOTA-frontier improvement would
earn $25. The terminal report also records separately funded $25 participation
allocations per agent; these are not API spend or proof of payment.

## What is observable

The public Vercel Next.js observatory is read-only. It shows concise research records,
exact tool calls, citations, the best verified support vector, API spend, long-running
code jobs, and recent-event replay. The pilot archive provides a paginated ledger.

Cloudflare Workflows persist every step and sleep without keeping a browser open.
GitHub code jobs may span rounds; completed results return to the proposing agent's
compact memory. The final report releases embargoed plans and links claims to ledger
entries and reproducible programs.

Hidden chain of thought is never requested or published. Pilot evidence URLs are
pinned to the run ID and do not silently follow the next run. The report is a compact
index: source/failed-avenue lists are excerpted and raw job inputs/outputs are not
bundled.

## Architecture and release boundary

~~~text
Vercel Next.js observatory (public, read-only)
        │ polls public state / owner start proxy
        ▼
Cloudflare Worker + Workflow (durable 8+ hour cadence)
        │ D1 append-only public/internal ledger
        ├── OpenAI Responses API (5 Luna High agents)
        └── GitHub Actions Node container (bounded bigint searches)
                 └── signed result callback → D1
~~~

Experiment descriptors live in lib/experiment-catalog.ts. Models, reasoning effort,
objectives, tools, acceptance criteria, and budgets belong to each experiment, not
to the AutoLabs brand. To publish a new experiment, add its stable route and a
descriptor with a higher sequence in that file. The public homepage and register
follow that ordering. The homepage uses a temporary redirect so browsers do not
permanently cache one experiment. Self-hosted mode opens /studio.

## Validation and local verification

The repository includes unit, type, production-build, desktop-interaction, and
mobile-verification coverage for the observatory, plus exact certificate regression
tests. The pilot's real controls include the exact bigint verifier, committed
Bremner k=4 fixture, five Luna High agent prompts, simultaneous reveal, private-plan
embargo, actual token-cost ledger, reserve and $50 preflight stop, durable
five-plus-five-minute Workflow, failure isolation, containerized asynchronous
bigint jobs, automatic final report, and replay-ready append-only log.

From the repository root:

~~~bash
npm install
npm test
npm run typecheck
npm run build

cd orchestrator-worker
npm install
npm run types
npm test
npm run typecheck
npm run deploy:dry
~~~

Copy .env.example to .env.local only for local configuration. Worker secrets are set
with wrangler secret put; never prefix a secret with NEXT_PUBLIC_.

## Repository map

- app/ and components/ — current experiment interface, register, and archived pilot observatory.
- lib/experiment-catalog.ts — published experiments, newest-first ordering, and landing-page selection.
- reward-compatibility — separate Experiment 002 runner, registered protocol, and reproducible analysis.
- lib/exact-verifier.ts — exact bigint witness, factor-pair, and support checks.
- orchestrator-worker/ — durable run engine, D1 migration, prompts, and budget ledger.
- math-worker/ — bounded deterministic search programs.
- .github/workflows/math-job.yml — pinned container job and signed callback.
- tests/ — exact certificate and observatory regression tests.
- public/photography/ — the reused sunset photograph and attribution notes.

## Scope and limitations

The pilot is a real computational research experiment, not a claim that an open
problem has been solved. The reward-compatibility studies are scoped findings in
finite policy languages, not universal classifications. Model support and pricing
assumptions follow the official
[GPT-5.6 Luna documentation](https://developers.openai.com/api/docs/models/gpt-5.6).

The page reuses Raphael Khalid's original photograph and accessible cream reading
surfaces. Experiment 002 and the Erdős pilot remain on their original stable routes.
Dependencies retain their licenses; the application source is MIT licensed.