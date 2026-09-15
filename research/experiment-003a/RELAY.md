# Experiment 3A relay setup

The one-click owner control queues a request. A local relay must be configured before the button can produce a Kaggle run. This keeps Kaggle credentials and any future provider credentials off Vercel and out of the browser.

## Relay contract

1. Authenticate to the worker using the private relay token stored in the owner machine’s secret store.
2. Poll `GET /api/persona-3a/next` no more frequently than once per minute.
3. When a request is returned, verify the study ID, phase, maximum runtime, and manifest hash against the checked-in files.
4. Push the private Kaggle notebook with the pinned metadata and a timeout no greater than 6,600 seconds.
5. Report `started`, then `completed`, `failed`, or `cancelled` through the authenticated worker update endpoint.
6. If the hash or phase does not match, refuse to run and report a failed hand-off.

The existing `afterlight/research/persona-discovery/relay-kaggle.py` is the starting point for the owner-local relay. Adapt it to this queue only after reviewing its current token and status behavior. Do not put `KAGGLE_API_TOKEN`, an OpenAI key, or the relay token in a committed file.

## Dry-run expectation

The first verification should claim no request and run only manifest/hash validation. The first real test should be an owner-approved Kaggle launch with `phase=development`, `maxRuntimeSeconds=6600`, and `apiBudgetUsd=0`. There is no confirmation launch path in this endpoint.

