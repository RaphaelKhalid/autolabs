CREATE TABLE IF NOT EXISTS persona_3b_review_records (
  run_id TEXT NOT NULL REFERENCES persona_3b_runs(id),
  record_id TEXT NOT NULL,
  plan_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (run_id, record_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_persona_3b_review_plan ON persona_3b_review_records(run_id, plan_hash);