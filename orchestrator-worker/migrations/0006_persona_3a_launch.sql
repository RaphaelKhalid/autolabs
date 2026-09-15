CREATE TABLE IF NOT EXISTS persona_launch_requests (
  id TEXT PRIMARY KEY,
  study_id TEXT NOT NULL,
  phase TEXT NOT NULL CHECK (phase IN ('preflight', 'development')),
  max_runtime_seconds INTEGER NOT NULL CHECK (max_runtime_seconds BETWEEN 900 AND 6600),
  status TEXT NOT NULL CHECK (status IN ('queued', 'claimed', 'started', 'completed', 'failed', 'cancelled')),
  idempotency_key TEXT NOT NULL UNIQUE,
  manifest_hash TEXT NOT NULL,
  requested_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  error_message TEXT
);

CREATE INDEX IF NOT EXISTS persona_launch_requests_status_requested_idx
  ON persona_launch_requests(status, requested_at);

