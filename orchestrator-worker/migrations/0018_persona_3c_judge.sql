-- Experiment 3C DESCRIBE stage (judge pass one): blinded pairwise
-- difference judgments run inside the Worker so no laptop process is
-- needed. Mirrors the 3B scoring budget-reservation pattern (reserve
-- worst-case before the provider call, settle to actual on success, charge
-- worst-case on failure) but scoped to a run's own judge_* counters rather
-- than a separate runs table.
ALTER TABLE persona_3c_runs ADD COLUMN judge_budget_usd REAL NOT NULL DEFAULT 0 CHECK (judge_budget_usd >= 0);
ALTER TABLE persona_3c_runs ADD COLUMN judge_spent_usd REAL NOT NULL DEFAULT 0 CHECK (judge_spent_usd >= 0);
ALTER TABLE persona_3c_runs ADD COLUMN judge_reserved_usd REAL NOT NULL DEFAULT 0 CHECK (judge_reserved_usd >= 0);
ALTER TABLE persona_3c_runs ADD COLUMN judge_calls INTEGER NOT NULL DEFAULT 0 CHECK (judge_calls >= 0);
ALTER TABLE persona_3c_runs ADD COLUMN judge_call_ceiling INTEGER NOT NULL DEFAULT 0 CHECK (judge_call_ceiling >= 0);

-- One row per blinded (direction, scenario, orderSwap) pair. job_id is the
-- sha256 of the canonical JSON of {directionKey, scenario, orderSwap}, so
-- re-submitting the same plan (append mode, batches of <=500 pairs) is a
-- no-op via INSERT OR IGNORE.
CREATE TABLE IF NOT EXISTS persona_3c_judge (
  run_id TEXT NOT NULL REFERENCES persona_3c_runs(id),
  job_id TEXT NOT NULL,
  direction_key TEXT NOT NULL,
  scenario TEXT NOT NULL,
  order_swap INTEGER NOT NULL CHECK (order_swap IN (0, 1)),
  prompt_sha256 TEXT,
  response_json TEXT,
  provider_response_id TEXT,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  cached_input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  cost_usd REAL NOT NULL DEFAULT 0,
  status TEXT NOT NULL CHECK (status IN ('queued', 'in_flight', 'complete', 'failed')),
  attempts INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (run_id, job_id)
);

-- Blinded texts, kept out of persona_3c_judge itself so the public results
-- route can never accidentally select them; never returned by any route.
CREATE TABLE IF NOT EXISTS persona_3c_judge_inputs (
  run_id TEXT NOT NULL REFERENCES persona_3c_runs(id),
  job_id TEXT NOT NULL,
  text_a TEXT NOT NULL,
  text_b TEXT NOT NULL,
  PRIMARY KEY (run_id, job_id)
);

CREATE INDEX IF NOT EXISTS idx_persona_3c_judge_run_status ON persona_3c_judge(run_id, status);
CREATE INDEX IF NOT EXISTS idx_persona_3c_judge_run_created ON persona_3c_judge(run_id, created_at);
