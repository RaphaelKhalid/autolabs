CREATE TABLE IF NOT EXISTS event_campaigns (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  max_grants INTEGER NOT NULL CHECK (max_grants = 5),
  claimed_grants INTEGER NOT NULL DEFAULT 0 CHECK (claimed_grants >= 0 AND claimed_grants <= max_grants),
  status TEXT NOT NULL CHECK (status IN ('preview', 'open', 'closed')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

INSERT OR IGNORE INTO event_campaigns(id,name,max_grants,claimed_grants,status,created_at,updated_at)
VALUES ('autolabs-50-2026','AutoLabs $50 research event',5,0,'preview',datetime('now'),datetime('now'));

CREATE TABLE IF NOT EXISTS event_grants (
  id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL REFERENCES event_campaigns(id),
  protocol_id TEXT NOT NULL,
  email TEXT NOT NULL,
  full_name TEXT,
  attribution_url TEXT,
  publish_consent INTEGER NOT NULL CHECK (publish_consent IN (0, 1)),
  idempotency_key TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'complete', 'partial', 'failed', 'budget-safe-stop')),
  reserved_cents INTEGER NOT NULL DEFAULT 1000 CHECK (reserved_cents = 1000),
  provider_spend_cents INTEGER NOT NULL DEFAULT 0 CHECK (provider_spend_cents >= 0),
  uncertain_spend_cents INTEGER NOT NULL DEFAULT 0 CHECK (uncertain_spend_cents >= 0),
  call_ceiling INTEGER NOT NULL,
  call_count INTEGER NOT NULL DEFAULT 0 CHECK (call_count >= 0 AND call_count <= call_ceiling),
  sample_target INTEGER NOT NULL,
  model TEXT NOT NULL,
  token_limit INTEGER NOT NULL,
  custom_json TEXT NOT NULL DEFAULT '{}',
  public_state_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  confirmed_at TEXT,
  completed_at TEXT,
  UNIQUE (campaign_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_event_grants_campaign_status
  ON event_grants(campaign_id, status, created_at);

CREATE INDEX IF NOT EXISTS idx_event_grants_protocol
  ON event_grants(protocol_id, created_at);
