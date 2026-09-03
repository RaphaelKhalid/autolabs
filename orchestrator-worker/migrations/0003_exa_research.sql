CREATE TABLE IF NOT EXISTS exa_usage (
  request_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  round INTEGER NOT NULL,
  query TEXT NOT NULL,
  cost_usd REAL NOT NULL,
  sources_json TEXT NOT NULL,
  at TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_exa_usage_run ON exa_usage(run_id);
