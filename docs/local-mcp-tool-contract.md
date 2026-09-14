# AutoLabs local MCP tool contract

This document proposes the smallest MCP surface for the existing self-hosted
AutoLabs runner. It is an adapter contract: the runner remains the sole owner
of configuration validation, provider calls, budgets, and local run records.

## Transport and trust boundary

The adapter uses line-delimited JSON-RPC 2.0 over standard input and output.
`AUTOLABS_RUNNER_URL` defaults to `http://127.0.0.1:8788`; every request to the
runner carries `Authorization: Bearer $AUTOLABS_LOCAL_TOKEN`. Provider keys
are loaded only by the runner process from its private environment. They are
never tool arguments, manifest fields, logs, or MCP responses.

The adapter should be run as the same local owner who started the runner. The
runner stays loopback-bound and the adapter must reject a non-HTTP(S) runner
URL. A client must not expose this owner interface to an untrusted network.

## Tools

### `autolabs_start`

Input is exactly one of:

```json
{"manifest": {"schemaVersion": 1, "title": "…", "objective": "…", "template": "integer-search-v1", "provider": "mock", "agents": []}}
```

or:

```json
{"manifestPath": "manifests/example.json"}
```

Inline manifests are forwarded to `POST /runs`; the runner validates and
immutably snapshots the configuration. Manifest paths are restricted to the
adapter working directory, valid JSON, and a small size limit. Paths resembling
`.env`, token, secret, or key files are rejected. An Afterlight frozen contract
manifest containing only contract identity and hash is not an AutoLabs local
configuration and must not be started by this tool.

### `autolabs_status`

With no arguments, call `GET /runs` and return the local run list. With
`{"runId":"run-…"}`, call `GET /runs/{runId}`. Accept only the runner's
`run-[a-z0-9-]+` identifier grammar; never interpolate arbitrary paths.

### `autolabs_artifact`

Input is `{"runId":"run-…"}`. Call `GET /runs/{runId}/export` and return the
JSON run export as MCP text content. The export contains public assistant
output, configuration, evaluations, usage, and interventions; it may contain
private research data and is not uploaded or published by the adapter.

## Error and lifecycle behavior

Tool failures return an MCP tool result with `isError: true` and a concise
message. JSON-RPC framing errors use standard error codes. The adapter does not
retry, pause, resume, or mutate a run. Starting a run remains subject to the
existing single-active-run, budget, and provider-key checks. MCP clients should
poll `autolabs_status` and use `autolabs_artifact` after completion.

This contract intentionally leaves the Afterlight deployed API and its frozen
experiment protocol untouched. A future Afterlight adapter would need its own
validated contract identity/hash input and owner endpoint; it must not reuse
this local AutoLabs configuration schema.
