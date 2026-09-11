# AutoLabs · Observable agent experiments

AutoLabs is a public laboratory for observable agent experiments. The
[homepage](https://autolabs-ebon.vercel.app) opens the newest published experiment;
the [experiment register](https://autolabs-ebon.vercel.app/experiments) is ordered
newest first. Each experiment retains its own stable page, protocol and evidence.
Study-specific research runners remain separate; arbitrary scientific evaluators
are not generated automatically.
The separate [self-hosted early release](selfhost/README.md) adds a creator,
OpenRouter models, no-cost mock tests, two starter evaluators and local run exports.
It is not connected to the public pilot's controls. See its documented limitations
before use. Source code is available under the MIT license; dependencies retain
their respective licenses.

## Experiment 002.1 — Testing finite reward compatibility

- [Live study and public ledger](https://autolabs-ebon.vercel.app/experiments/reward-compatibility-21)
- [Research implementation and protocol](https://github.com/RaphaelKhalid/reward-compatibility/tree/main/v21)
- 80 development cases and 320 held-out cases across coin tracking and a restricted
  Backdoor-Easy-inspired affine-trigger language. Three comparison methods use
  independent Luna calls: description judgment, unguided search and verifier-guided search.
- Up to eight concurrent API calls, fixed sample size, read-only public status and
  five-record ledger pages. Held-out responses stay sealed until completion.
- The **$40 cap is shared with Experiment 002**, including its prior commitments;
  the interface separates old commitments, current spend and outstanding reservations.
- This finite-language validation does not establish all three aligned / orthogonal /
  in-conflict categories for unrestricted tasks. A missing witness is not a conflict proof.
- The checker interprets bounded JSON data, not arbitrary generated Python or JavaScript.
  The durable cloud runner remains in the research repository; the browser never starts
  API calls or receives private credentials. Closing the page does not stop execution.
- The page reuses Raphael Khalid's original photograph and accessible cream reading
  surfaces. Experiment 002 and the Erdős pilot remain on their original stable routes.

## Experiment 002 — Measuring reward compatibility (complete)

- [Live study and research ledger](https://autolabs-ebon.vercel.app/experiments/reward-compatibility)
- [Research repository, registered protocol and analysis](https://github.com/RaphaelKhalid/reward-compatibility)
- Background: Raphael Khalid's sunset photograph, with a cream overlay and
  separate reading surfaces. [Source and image reuse notes](public/photography/README.md).
- The metric guide distinguishes exact outcome scoring, reasoning rewards and
  monitoring. Public call labels translate grader ratings into per-sample rewards;
  they do not reveal sealed records or represent aggregate research findings.
- Luna actor with reasoning effort none; fresh Luna High evaluators and reporter.
- Exact coin-task scoring, matched reward conditions, three repeats, and sealed
  held-out evaluation. The final figure appears after the frozen run completes.
- Separate Cloudflare Durable Object runner, $4 feasibility gate and $40 inclusive
  OpenAI cap. The research repository owns its runner, protocol and figure export;
  this repository owns the AutoLabs interface. No private reasoning traces are exposed.

To publish the next experiment, add its stable route and a descriptor with a higher
`sequence` in `lib/experiment-catalog.ts`. The public homepage and register both
follow that ordering. The homepage uses a temporary redirect so browsers do not
permanently cache one experiment. Self-hosted mode still opens `/studio`.

## Pilot 001 — completed

The first pilot completed **100 synchronized rounds**, not 100 independent trials,
between 3 and 8 September 2026. It produced **no certified k=5 solution and no
verified SOTA improvement**. See the [results and paginated ledger](https://autolabs-ebon.vercel.app/experiments/erdos-885).
The original [animated pilot archive](https://autolabs-ebon.vercel.app/experiments/erdos-885/lab)
remains available separately; it is not the current experiment.

- 606 computation jobs: 297 complete, 297 partial, 12 failed.
- 87 candidate-check records, 2,611 events, 496 released private plans.
- Final internal leader: 21 valid cells in a sparse 7×6 table, submitted in round
  40. This is not a complete rectangle or a proven field-level novelty.
- Pilot-only recorded estimates: OpenAI $7.41198192; Exa $3.535.
- Including earlier runs: OpenAI $9.32071384; Exa $4.536.

Usage estimates are application-ledger records, not reconciled invoices; missing
usage from timed-out requests may be excluded. Hosting, Codex subscriptions,
human time and separately funded agent rewards are not included. Operational
repairs and human-directed policy changes were part of this supervised pilot.

Experiment descriptors live in `lib/experiment-catalog.ts`. Models, reasoning
effort, objectives, tools, acceptance criteria and budgets belong to each
experiment, not to the AutoLabs brand. Pilot evidence URLs are pinned to its run
ID and will not silently follow the next run. The report is a compact index:
source/failed-avenue lists are excerpted and raw job inputs/outputs are not bundled.

## Historical pilot methodology

Five GPT-5.6 Luna High mathematicians work in synchronized five-minute private
research loops and five-minute round tables on [Erdős Problem 885](https://erdosproblemaday.com/day/885-factor-difference-k5).
The public observatory shows concise research records, exact tool calls, citations,
the best verified support vector, API spend, long-running code jobs, and a recent-event
replay. The pilot archive provides a paginated ledger. Hidden chain-of-thought is never requested or published.

## The mathematical target

For a positive integer `N`, let `D(N) = {|a-b| : ab=N}`. The k=5 problem asks
for five distinct `N` values sharing five distinct factor-pair differences.
Every proposed cell is checked with bigint arithmetic:

```text
d² + 4N = m²,  a = (m-d)/2,  b = (m+d)/2,  ab = N
```

There is no floating-point tolerance and no reward for being “close” to a square.
The known Bremner k=4 certificate is a committed regression fixture. A k=5
certificate stops the run immediately. The secondary milestone is a strict
complete-rectangle improvement over the currently tracked `(5 integers, 4
differences)` / `(3 integers, 5 differences)` frontier: at least `6×4` or `4×5`.

## Historical pilot covenant

- Exactly five unrestricted expert mathematicians, all using `gpt-5.6-luna`
  with `high` reasoning effort.
- Distinct alien personas alter idea-generation, not access to human mathematics.
- Reports are sealed and revealed simultaneously; one reaction per agent follows.
- Round 25 is a real midpoint town hall. The second half uses five rotating method
  lanes, a balanced row-and-column progress metric, and divisor completion only
  as one rotating validation slot rather than a shared default.
- Collaboration credit is non-self, capped at two peers per reaction, and audited
  for systematic under-recognition of Solvi and Tess.
- Next-round plans stay private from peers and observers until the experiment ends.
- One failed call is retried and isolated; the other agents continue.
- A real one-round dress rehearsal uses the production prompts, model, exact tools,
  ledger and budget.
- Initial competition target: 50 rounds; guaranteed allocation: 25 rounds.
  The owner extended the pilot to 100. New-run defaults are distinct from this archive.
- Pilot OpenAI experiment ceiling: `$50`, with a `$1.50` software reserve and
  preflight authorization before every five-call batch.
- Recorded provider responses are costed from input, cached-input and output
  token usage. Failed requests without returned usage are not invoice-reconciled.
  API secrets belong in ignored local configuration and deployment secret stores.

The k=5 winner may later use a `$50` project budget supplied by Raphael. Each
credited collaborator receives a separate `$10`. A verified SOTA-frontier
improvement would earn `$25`. The terminal report also records separately funded
`$25` participation allocations per agent; these are not API spend or proof of payment.

## Pilot architecture

```text
Vercel Next.js observatory (public, read-only)
        │ polls public state / owner start proxy
        ▼
Cloudflare Worker + Workflow (durable 8+ hour cadence)
        │ D1 append-only public/internal ledger
        ├── OpenAI Responses API (5 Luna High agents)
        └── GitHub Actions Node container (bounded bigint searches)
                 └── signed result callback → D1
```

Cloudflare Workflows persist every step and sleep without keeping a browser open.
GitHub code jobs may span rounds; completed results are delivered to the proposing
agent’s compact memory. The final report releases embargoed plans and links every
claim to ledger entries and reproducible programs.

## Local verification

```bash
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
```

Copy `.env.example` to `.env.local` only for local configuration. Worker secrets
are set with `wrangler secret put`; never prefix a secret with `NEXT_PUBLIC_`.

## Repository map

- `app/`, `components/` — current experiment interface, register and archived pilot observatory.
- `lib/experiment-catalog.ts` — published experiments, newest-first ordering and landing-page selection.
- [reward-compatibility](https://github.com/RaphaelKhalid/reward-compatibility) — separate Experiment 002 runner, registered protocol and reproducible analysis.
- `lib/exact-verifier.ts` — exact bigint witness, factor-pair and support checks.
- `orchestrator-worker/` — durable run engine, D1 migration, prompts and budget ledger.
- `math-worker/` — bounded deterministic search programs.
- `.github/workflows/math-job.yml` — pinned container job and signed callback.
- `tests/` — exact certificate regression tests.

## Research baseline

The prompt registry records the published search regions and fixed saturated
specializations from the [Problem 885 research log](https://erdosproblemaday.com/day/885-factor-difference-k5)
so agents do not spend the run rediscovering known negative results. Model support
and pricing assumptions follow the official [GPT-5.6 Luna documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna).

This is a real computational research experiment, not a claim that an open problem
has been solved. Only an exported exact certificate earns “Eureka”.
