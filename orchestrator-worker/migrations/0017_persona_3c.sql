-- Experiment 3C: live GPU-pod progress reporting into the AutoLabs harness.
CREATE TABLE IF NOT EXISTS persona_3c_runs (
  id TEXT PRIMARY KEY,
  study_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'complete', 'failed', 'stopped')),
  stage TEXT NOT NULL,
  manifest_hash TEXT NOT NULL,
  budget_usd REAL NOT NULL CHECK (budget_usd > 0 AND budget_usd <= 20),
  spent_usd REAL NOT NULL DEFAULT 0 CHECK (spent_usd >= 0),
  gpu_hours REAL NOT NULL DEFAULT 0 CHECK (gpu_hours >= 0),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT,
  error_message TEXT,
  last_record_id TEXT,
  idempotency_key TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS persona_3c_records (
  run_id TEXT NOT NULL REFERENCES persona_3c_runs(id),
  stage TEXT NOT NULL,
  record_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (run_id, stage, record_id)
);

CREATE TABLE IF NOT EXISTS persona_3c_progress (
  run_id TEXT NOT NULL REFERENCES persona_3c_runs(id),
  stage TEXT NOT NULL,
  done INTEGER NOT NULL DEFAULT 0 CHECK (done >= 0),
  total INTEGER NOT NULL DEFAULT 0 CHECK (total >= 0),
  updated_at TEXT NOT NULL,
  PRIMARY KEY (run_id, stage)
);

CREATE TABLE IF NOT EXISTS persona_3c_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES persona_3c_runs(id),
  at TEXT NOT NULL,
  stage TEXT,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_persona_3c_runs_created ON persona_3c_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_persona_3c_records_run ON persona_3c_records(run_id, stage);
CREATE INDEX IF NOT EXISTS idx_persona_3c_progress_run ON persona_3c_progress(run_id);
CREATE INDEX IF NOT EXISTS idx_persona_3c_events_run ON persona_3c_events(run_id, id);
