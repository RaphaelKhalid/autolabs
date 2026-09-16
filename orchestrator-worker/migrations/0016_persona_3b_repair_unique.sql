-- Allow append-only repair attempts to coexist with each original score key.
PRAGMA foreign_keys=OFF;
CREATE TABLE persona_3b_scores_repair_v2 (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES persona_3b_runs(id),
  shard_id TEXT NOT NULL REFERENCES persona_3b_shards(id),
  record_id TEXT NOT NULL,
  repeat_index INTEGER NOT NULL CHECK (repeat_index IN (0, 1, 2)),
  response_json TEXT NOT NULL,
  response_sha256_a TEXT NOT NULL,
  response_sha256_b TEXT NOT NULL,
  order_swap INTEGER NOT NULL CHECK (order_swap IN (0, 1)),
  provider_response_id TEXT,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  cached_input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  repair_index INTEGER NOT NULL DEFAULT 0,
  UNIQUE (shard_id, record_id, repeat_index, repair_index)
);
INSERT INTO persona_3b_scores_repair_v2 (id,run_id,shard_id,record_id,repeat_index,response_json,response_sha256_a,response_sha256_b,order_swap,provider_response_id,input_tokens,cached_input_tokens,output_tokens,created_at,repair_index)
SELECT id,run_id,shard_id,record_id,repeat_index,response_json,response_sha256_a,response_sha256_b,order_swap,provider_response_id,input_tokens,cached_input_tokens,output_tokens,created_at,repair_index
FROM persona_3b_scores;
DROP TABLE persona_3b_scores;
ALTER TABLE persona_3b_scores_repair_v2 RENAME TO persona_3b_scores;
CREATE INDEX IF NOT EXISTS idx_persona_3b_scores_run ON persona_3b_scores(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_persona_3b_score_repair ON persona_3b_scores(run_id, shard_id, record_id, repeat_index, repair_index);
PRAGMA foreign_keys=ON;