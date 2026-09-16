CREATE TABLE IF NOT EXISTS persona_3b_attempts (
  run_id TEXT NOT NULL REFERENCES persona_3b_runs(id),
  shard_id TEXT NOT NULL REFERENCES persona_3b_shards(id),
  record_id TEXT NOT NULL,
  repeat_index INTEGER NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('in_flight','complete','failed')),
  created_at TEXT NOT NULL,
  PRIMARY KEY (shard_id, record_id, repeat_index)
);
