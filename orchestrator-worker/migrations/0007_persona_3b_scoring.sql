CREATE TABLE IF NOT EXISTS persona_3b_runs (
  id TEXT PRIMARY KEY,
  study_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'awaiting_adjudication', 'synthesizing', 'complete', 'failed', 'cancelled')),
  phase TEXT NOT NULL,
  manifest_hash TEXT NOT NULL,
  source_artifact_hash TEXT NOT NULL,
  total_records INTEGER NOT NULL CHECK (total_records = 780),
  primary_records INTEGER NOT NULL CHECK (primary_records = 780),
  primary_comparisons INTEGER NOT NULL CHECK (primary_comparisons = 768),
  planned_judgments INTEGER NOT NULL CHECK (planned_judgments = 885),
  repeat_records INTEGER NOT NULL CHECK (repeat_records = 117),
  disagreement_records INTEGER NOT NULL CHECK (disagreement_records = 117),
  call_ceiling INTEGER NOT NULL CHECK (call_ceiling = 1014),
  call_count INTEGER NOT NULL DEFAULT 0 CHECK (call_count >= 0 AND call_count <= call_ceiling),
  budget_usd REAL NOT NULL CHECK (budget_usd >= 0 AND budget_usd <= 10),
  reserved_usd REAL NOT NULL CHECK (reserved_usd >= 0),
  spent_usd REAL NOT NULL DEFAULT 0 CHECK (spent_usd >= 0),
  idempotency_key TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT,
  error_message TEXT,
  analysis_digest TEXT,
  artifact_hash TEXT,
  shortlist_summary_json TEXT,
  confirmation_status TEXT CHECK (confirmation_status IS NULL OR confirmation_status = 'pending')
);

CREATE TABLE IF NOT EXISTS persona_3b_shards (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES persona_3b_runs(id),
  phase TEXT NOT NULL CHECK (phase IN ('primary', 'repeat', 'disagreement', 'synthesis')),
  shard_index INTEGER NOT NULL CHECK (shard_index BETWEEN 0 AND 4),
  agent_id TEXT NOT NULL,
  expected_records INTEGER NOT NULL CHECK (expected_records >= 0),
  expected_calls INTEGER NOT NULL CHECK (expected_calls >= 0),
  status TEXT NOT NULL CHECK (status IN ('blocked', 'queued', 'claimed', 'running', 'complete', 'failed')),
  completed_records INTEGER NOT NULL DEFAULT 0 CHECK (completed_records >= 0),
  attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
  call_count INTEGER NOT NULL DEFAULT 0 CHECK (call_count >= 0),
  input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
  cached_input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (cached_input_tokens >= 0),
  output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
  claim_token_hash TEXT,
  claimed_at TEXT,
  updated_at TEXT NOT NULL,
  completed_at TEXT,
  error_message TEXT,
  UNIQUE (run_id, phase, shard_index)
);

CREATE TABLE IF NOT EXISTS persona_3b_events (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES persona_3b_runs(id),
  shard_id TEXT,
  phase TEXT,
  agent_id TEXT,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  visible INTEGER NOT NULL DEFAULT 1 CHECK (visible IN (0, 1)),
  at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_persona_3b_runs_status_created ON persona_3b_runs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_persona_3b_shards_run_status ON persona_3b_shards(run_id, status, phase, shard_index);
CREATE INDEX IF NOT EXISTS idx_persona_3b_events_run_at ON persona_3b_events(run_id, at);
