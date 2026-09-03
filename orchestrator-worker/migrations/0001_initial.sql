PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  workflow_id TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL,
  mode TEXT NOT NULL,
  phase TEXT NOT NULL,
  round INTEGER NOT NULL DEFAULT 0,
  target_rounds INTEGER NOT NULL,
  minimum_rounds INTEGER NOT NULL,
  phase_minutes INTEGER NOT NULL,
  phase_ends_at TEXT,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  budget_usd REAL NOT NULL,
  reserve_usd REAL NOT NULL,
  spent_usd REAL NOT NULL DEFAULT 0,
  calls INTEGER NOT NULL DEFAULT 0,
  best_support_json TEXT NOT NULL DEFAULT '[0,0,0,0,0]',
  best_label TEXT NOT NULL DEFAULT 'No candidate verified yet',
  best_verified INTEGER NOT NULL DEFAULT 0,
  sota_improved INTEGER NOT NULL DEFAULT 0,
  public_state_json TEXT NOT NULL,
  report_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
  run_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  at TEXT NOT NULL,
  round INTEGER NOT NULL,
  phase TEXT NOT NULL,
  agent_id TEXT,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  visible INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (run_id, seq),
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS agent_memory (
  run_id TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  public_summary_json TEXT NOT NULL DEFAULT '{}',
  private_plan_json TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL,
  PRIMARY KEY (run_id, agent_id),
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS usage (
  response_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  round INTEGER NOT NULL,
  phase TEXT NOT NULL,
  input_tokens INTEGER NOT NULL,
  cached_input_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  cost_usd REAL NOT NULL,
  at TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  round INTEGER NOT NULL,
  job_type TEXT NOT NULL,
  params_json TEXT NOT NULL,
  status TEXT NOT NULL,
  result_json TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  completed_at TEXT,
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_jobs_run_status ON jobs(run_id, status);
CREATE INDEX IF NOT EXISTS idx_usage_run ON usage(run_id);
