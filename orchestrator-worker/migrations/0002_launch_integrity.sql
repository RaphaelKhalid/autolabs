CREATE TABLE IF NOT EXISTS private_plans (
  run_id TEXT NOT NULL,
  round INTEGER NOT NULL,
  agent_id TEXT NOT NULL,
  plan_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (run_id, round, agent_id),
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_single_active
  ON runs((1))
  WHERE status = 'running';

CREATE INDEX IF NOT EXISTS idx_private_plans_run_round
  ON private_plans(run_id, round, agent_id);
