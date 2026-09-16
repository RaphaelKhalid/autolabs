CREATE TABLE IF NOT EXISTS persona_3b_scores (
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
  UNIQUE (shard_id, record_id, repeat_index)
);

CREATE INDEX IF NOT EXISTS idx_persona_3b_scores_run ON persona_3b_scores(run_id, created_at);
