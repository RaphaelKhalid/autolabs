# AutoLabs runtime credentials

## Short answer

You should not make AutoLabs use your Codex ChatGPT login as its production runtime credential. Codex sign-in is for the Codex CLI/app workflow; it is not a general-purpose API token to place in a public harness. OpenAI’s Codex help page distinguishes ChatGPT sign-in from API-key authentication ([official explanation](https://help.openai.com/en/articles/11381614-api-codex-cli-and-sign-in-with-chatgpt)).

For production AutoLabs calls, use a server-side OpenAI API key in Vercel/Worker secrets, with a separate restricted project or budget if available. Never put it in browser JavaScript, a public repository, a Kaggle notebook, a D1 row, or a client-visible error.

## Recommended separation

- Codex account: development, code review, protocol teaching, and owner approval.
- Vercel server route: authenticated control plane and safe proxy; no provider key reaches the browser.
- Orchestrator Worker: ledger, idempotency, budget state, and durable status.
- Kaggle owner-local relay: private GPU submission using a Kaggle credential stored only on the owner machine.
- OpenAI API: only when a registered protocol explicitly permits it; 3A’s manifest currently sets paid API spend to zero.

The Experiment 3A button now queues a request. A relay must claim and verify it before pushing Kaggle. That queue is the “real harness” boundary: the user can click once, but the long-running GPU process is handled by the appropriate execution environment.

