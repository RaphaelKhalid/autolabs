CREATE TABLE IF NOT EXISTS checkpoint_email_outbox (
  run_id TEXT NOT NULL,
  checkpoint_round INTEGER NOT NULL CHECK (checkpoint_round IN (35, 40, 45, 50)),
  round_start INTEGER NOT NULL,
  recipient TEXT NOT NULL,
  subject TEXT NOT NULL,
  body_text TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sending', 'sent', 'failed')),
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at TEXT NOT NULL,
  sent_at TEXT,
  PRIMARY KEY (run_id, checkpoint_round),
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_checkpoint_email_outbox_status
  ON checkpoint_email_outbox(status, created_at);
