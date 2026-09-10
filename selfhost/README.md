# Self-hosted AutoLabs — early release

This runner is separate from the historical Cloudflare Erdős pilot. It has no
access to that pilot's controls or credentials. It supports 1–8 configurable
agents, OpenRouter model IDs, mock tests, research/discussion rounds, local
records, pause/resume, configuration import/export and complete run exports.

## Start without Docker

Requires Node.js 24 and npm. In the repository root:

```sh
npm ci
node selfhost/setup.ts
```

Edit the ignored `.env.selfhost.local` file to set `OPENROUTER_API_KEY`. Leave it
empty for mock runs. The setup generates a random `AUTOLABS_LOCAL_TOKEN`; use
that owner token to unlock the creator. Do not put provider keys in prompts,
configuration exports or browser storage. Local files use restrictive POSIX
permissions where supported; on Windows, use a private user directory and check
the directory ACLs. No secrets are copied into the container build context.

Run these in two terminals:

```sh
node --env-file=.env.selfhost.local selfhost/server.ts
node selfhost/web.ts dev --hostname 127.0.0.1 --port 3000
```

Open http://localhost:3000/studio. Connect, run a mock test, then explicitly
choose OpenRouter for a paid run. A connection check issues one text request
with 16 maximum output tokens and rejects estimated reservations over $0.01.

For a native production build, use `node selfhost/web.ts build`, then
`node selfhost/web.ts start --hostname 127.0.0.1 --port 3000` alongside the runner.
The helper loads environment values without forwarding `--env-file` into Next.js workers.

## Container deployment

After setup, use Docker Compose:

```sh
docker compose --env-file .env.selfhost.local up --build -d
```

Only the web interface is published, on loopback. The runner is on the private
container network, and records persist in the `autolabs-data` volume. Do not
remove that volume if you want to retain experiments.

For a remote server, keep loopback binding and connect through an SSH tunnel:

```sh
ssh -L 3000:127.0.0.1:3000 user@your-server
```

The remote runner continues when the browser or tunnel closes. A runner on your
laptop cannot continue while the laptop is asleep or off. Do not expose this
early-release owner interface directly to the public internet. Multi-user
accounts, workspace isolation and arbitrary endpoint support are not implemented.

## Templates and evaluation

- `integer-search-v1`: deterministic known-answer test minimizing
  `(x-17)^2+(y+9)^2`, with integer `x,y` in `[-100,100]`. A zero score is a template
  success, not a scientific discovery. Editing the prompt does not edit this evaluator.
- `research-notes-v1`: research notes with human review, no automated truth,
  novelty or success verdict. No external retrieval is attached.

The template IDs are versioned. To install a new trusted evaluator, extend the
configuration validator and evaluator in `selfhost/config.ts`, and add tests.
This version does not accept uploaded programs, arbitrary shell commands, MCP
tools or remote evaluator URLs. Tool isolation and a plugin interface are later
work; adding an arbitrary code runner without isolation is not supported.

Requests are serialized in this release. Research prompts have only the agent's
own previous report; meetings receive the team reports and preceding reactions.
The configured phase interval is a delay after a phase, not a promise of
continuous model reasoning for that duration. Each agent makes one request per
phase. Provider reasoning settings are not yet configurable.

## Budgets, recovery and evidence

Each launch creates a validated configuration snapshot and SHA-256 configuration
hash. Existing runs are not edited when the creator configuration changes.
The configured model ID is always explicit; returned model IDs are logged.

Prices come from OpenRouter's model catalogue. Conservative reservations precede
requests and are reconciled with returned usage when available. Missing usage
keeps the reservation. Reservations are safeguards, not a billing guarantee;
set a provider-side key limit too. Connection checks have a separate local ledger.
Each run has its own budget: starting another run creates a new allocation.

Requests interrupted by a process restart pause for review; they are not silently
retried. Resuming after an error authorizes another attempt. In-flight requests
may still finish after a pause. Run exactly one runner per data directory.
The server keeps atomic JSON snapshots under `.autolabs/` (or `/data` in Docker).
This is a single-owner, single-runner implementation, not distributed scheduling.

Run exports contain public assistant output, configuration, evaluations, usage
and interventions—not hidden reasoning. They may contain private research data.
Nothing is automatically uploaded or published. Review exports before sharing.

## Tests

```sh
npm test
npm run typecheck
npm run build
```

The automated runner tests use a mock provider and spend no API credit. Docker
packaging needs verification on a machine with Docker installed.
