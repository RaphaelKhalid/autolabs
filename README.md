# AutoLabs · Observable agent experiments

AutoLabs is evolving into a reusable public laboratory. The animated lab remains
the homepage, with an [experiment register](https://autolabs-ebon.vercel.app/experiments)
and stable per-experiment evidence links. The current execution engine is still
specialized for Erdős 885; arbitrary-problem launching is not implemented yet.

## Pilot 001 — completed

The first pilot completed **100 synchronized rounds**, not 100 independent trials,
between 3 and 8 September 2026. It produced **no certified k=5 solution and no
verified SOTA improvement**. See the [results and paginated ledger](https://autolabs-ebon.vercel.app/experiments/erdos-885).

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

## Experiment covenant

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
- Current OpenAI experiment ceiling: `$50`, with a `$1.50` software reserve and
  preflight authorization before every five-call batch.
- Recorded provider responses are costed from input, cached-input and output
  token usage. Failed requests without returned usage are not invoice-reconciled.
  API secrets belong in ignored local configuration and deployment secret stores.

The k=5 winner may later use a `$50` project budget supplied by Raphael. Each
credited collaborator receives a separate `$10`. A verified SOTA-frontier
improvement would earn `$25`. The terminal report also records separately funded
`$25` participation allocations per agent; these are not API spend or proof of payment.

## Architecture

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

- `app/`, `components/` — the public observatory and owner-only ribbon control.
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
