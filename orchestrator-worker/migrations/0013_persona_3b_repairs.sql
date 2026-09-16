-- Append-only same-run repair metadata. Original score rows remain unchanged.
ALTER TABLE persona_3b_scores ADD COLUMN repair_index INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_persona_3b_score_repair ON persona_3b_scores(run_id, shard_id, record_id, repeat_index, repair_index);
CREATE TABLE IF NOT EXISTS persona_3b_amendments (
  run_id TEXT PRIMARY KEY,
  amendment_hash TEXT NOT NULL,
  original_call_ceiling INTEGER NOT NULL,
  effective_call_ceiling INTEGER NOT NULL,
  hard_budget_usd REAL NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);