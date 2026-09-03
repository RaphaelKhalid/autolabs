# Security and transparency boundary

## Secrets

`OPENAI_API_KEY`, `ADMIN_TOKEN`, `GITHUB_TOKEN` and `CALLBACK_SECRET` are runtime
secrets. They are ignored locally, never returned from an endpoint, and never
included in prompts, browser bundles, events or logs. Public configuration is
limited to the Worker origin and repository name.

## Public access

Experiment state, visible events and completed reports are read-only. Start actions
require the private owner key at the Vercel route and again at the Worker. Secret
comparisons hash to fixed length and use constant-time equality. Code-job callbacks
require a distinct secret. CORS is restricted to the production observatory and
local development origins.

## Research visibility

Public records contain hypotheses, claims, equations, citations, exact tool inputs
and outputs, failed avenues, reactions and conclusions. The system neither requests
nor publishes hidden chain-of-thought. Private next-round plans are stored in an
internal D1 column and released only in the final report.

## Computation

All witness checks use `bigint`. Long searches are bounded by an explicit check
limit and execute in an isolated Node container with read-only repository access.
The container can only post a result to the signed callback; it receives no OpenAI
credential or owner key.
