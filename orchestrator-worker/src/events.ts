import { EVENT_CAMPAIGN_ID, EVENT_GRANT_CENTS, eventProtocolById } from '../../lib/events-catalog';

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MAX_BODY_BYTES = 32_768;

function json(value: unknown, init: ResponseInit = {}, corsHeaders: Record<string, string>) {
  const headers = new Headers(init.headers);
  headers.set('content-type', 'application/json; charset=utf-8');
  headers.set('cache-control', 'no-store');
  for (const [key, item] of Object.entries(corsHeaders)) headers.set(key, item);
  return new Response(JSON.stringify(value), { ...init, headers });
}

function publicGrant(row: {
  id: string;
  protocol_id: string;
  status: string;
  call_ceiling: number;
  call_count: number;
  sample_target: number;
  model: string;
  token_limit: number;
  public_state_json: string;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}) {
  let state: Record<string, unknown> = {};
  try { state = JSON.parse(row.public_state_json) as Record<string, unknown>; } catch { /* retain a minimal safe record */ }
  return {
    id: row.id,
    protocolId: row.protocol_id,
    status: row.status,
    callCeiling: row.call_ceiling,
    callCount: row.call_count,
    sampleTarget: row.sample_target,
    model: row.model,
    tokenLimit: row.token_limit,
    state,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    completedAt: row.completed_at,
  };
}

export async function campaign(env: Env, corsHeaders: Record<string, string>) {
  const row = await env.DB.prepare(`SELECT id,name,max_grants AS maxGrants,claimed_grants AS claimedGrants,status
    FROM event_campaigns WHERE id=?`).bind(EVENT_CAMPAIGN_ID).first<{
      id: string;
      name: string;
      maxGrants: number;
      claimedGrants: number;
      status: 'preview' | 'open' | 'closed';
    }>();
  if (!row) return json({ error: 'Event campaign is not initialized.' }, { status: 503 }, corsHeaders);
  const launchEnabled = String(env.EVENTS_LAUNCH_ENABLED) === 'true' && row.status === 'open';
  return json({
    campaign: {
      id: row.id,
      name: row.name,
      maxGrants: row.maxGrants,
      claimedGrants: row.claimedGrants,
      remaining: Math.max(0, row.maxGrants - row.claimedGrants),
      status: launchEnabled ? row.status : 'preview',
      launchEnabled,
      grantCents: EVENT_GRANT_CENTS,
    },
  }, {}, corsHeaders);
}

function stringField(value: unknown, max: number) {
  return typeof value === 'string' && value.length <= max ? value.trim() : '';
}

function httpsUrl(value: unknown) {
  if (value === undefined || value === null || value === '') return null;
  if (typeof value !== 'string' || value.length > 320) return undefined;
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'https:' ? parsed.toString() : undefined;
  } catch { return undefined; }
}

function customization(value: unknown): Record<string, string> | null {
  if (value === undefined || value === null) return {};
  if (typeof value !== 'object' || Array.isArray(value)) return null;
  const allowed = new Set(['scenarioFamily', 'cueWording', 'domainSubset', 'secondaryAnalysis', 'notes']);
  const normalized: Record<string, string> = {};
  for (const [key, item] of Object.entries(value)) {
    if (!allowed.has(key) || typeof item !== 'string' || item.length > 600) return null;
    normalized[key] = item.trim();
  }
  return normalized;
}
export async function claimEvent(request: Request, env: Env, corsHeaders: Record<string, string>) {
  const contentLength = Number(request.headers.get('content-length') ?? 0);
  if (Number.isFinite(contentLength) && contentLength > MAX_BODY_BYTES) return json({ error: 'Payload too large.' }, { status: 413 }, corsHeaders);
  if (String(env.EVENTS_LAUNCH_ENABLED) !== 'true') return json({ error: 'The sponsored event is not open yet.' }, { status: 503 }, corsHeaders);

  const rawText = await request.text();
  if (rawText.length > MAX_BODY_BYTES) return json({ error: 'Payload too large.' }, { status: 413 }, corsHeaders);
  let raw: Record<string, unknown> | null = null;
  try {
    const parsed = JSON.parse(rawText) as unknown;
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) raw = parsed as Record<string, unknown>;
  } catch { /* handled by validation below */ }
  if (!raw) return json({ error: 'Invalid claim payload.' }, { status: 400 }, corsHeaders);

  const protocolId = stringField(raw.protocolId, 80);
  const protocol = eventProtocolById(protocolId);
  const email = stringField(raw.email, 254).toLowerCase();
  const fullName = stringField(raw.fullName, 100) || null;
  const attributionUrl = httpsUrl(raw.attributionUrl);
  const idempotencyKey = stringField(raw.idempotencyKey, 100);
  const publishConsent = raw.publishConsent === true;
  const custom = customization(raw.customization);

  if (!protocol || custom === null || !EMAIL_PATTERN.test(email) || !idempotencyKey || idempotencyKey.length < 16 || attributionUrl === undefined || !publishConsent) {
    return json({ error: 'A valid protocol, email, optional HTTPS attribution link, idempotency key, and publication consent are required.' }, { status: 400 }, corsHeaders);
  }
  const customJson = JSON.stringify(custom);
  if (customJson.length > 4_096) return json({ error: 'Customization payload is too large.' }, { status: 400 }, corsHeaders);

  const existing = await env.DB.prepare(`SELECT id,protocol_id,status,call_ceiling,call_count,sample_target,model,token_limit,public_state_json,created_at,updated_at,completed_at
    FROM event_grants WHERE campaign_id=? AND idempotency_key=?`).bind(EVENT_CAMPAIGN_ID, idempotencyKey).first<Parameters<typeof publicGrant>[0]>();
  if (existing) return json({ accepted: true, idempotent: true, grant: publicGrant(existing) }, { status: 200 }, corsHeaders);

  const grantId = `event-${crypto.randomUUID()}`;
  const now = new Date().toISOString();
  const publicState = {
    protocolId,
    status: 'queued',
    sampleTarget: protocol.sampleTarget,
    callCeiling: protocol.callCeiling,
    callCount: 0,
    budgetState: 'reserved',
    message: 'Grant reserved. A monitor review is required before execution.',
  };

  try {
    const results = await env.DB.batch([
      env.DB.prepare(`INSERT INTO event_grants(id,campaign_id,protocol_id,email,full_name,attribution_url,publish_consent,idempotency_key,status,reserved_cents,call_ceiling,sample_target,model,token_limit,custom_json,public_state_json,created_at,updated_at,confirmed_at)
        SELECT ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,? FROM event_campaigns
        WHERE id=? AND status='open' AND claimed_grants < max_grants`)
        .bind(grantId, EVENT_CAMPAIGN_ID, protocolId, email, fullName, attributionUrl, 1, idempotencyKey, 'queued', EVENT_GRANT_CENTS, protocol.callCeiling, protocol.sampleTarget, protocol.model, protocol.tokenLimit, customJson, JSON.stringify(publicState), now, now, now, EVENT_CAMPAIGN_ID),
      env.DB.prepare(`UPDATE event_campaigns SET claimed_grants=claimed_grants+1,updated_at=?
        WHERE id=? AND status='open' AND claimed_grants < max_grants
        AND EXISTS (SELECT 1 FROM event_grants WHERE id=?)`)
        .bind(now, EVENT_CAMPAIGN_ID, grantId),
    ]);
    if (Number(results[0]?.meta.changes ?? 0) !== 1 || Number(results[1]?.meta.changes ?? 0) !== 1) {
      return json({ error: 'All five sponsored grants have been reserved or the event is not open.' }, { status: 409 }, corsHeaders);
    }
  } catch {
    const raced = await env.DB.prepare(`SELECT id,protocol_id,status,call_ceiling,call_count,sample_target,model,token_limit,public_state_json,created_at,updated_at,completed_at
      FROM event_grants WHERE campaign_id=? AND idempotency_key=?`).bind(EVENT_CAMPAIGN_ID, idempotencyKey).first<Parameters<typeof publicGrant>[0]>();
    if (raced) return json({ accepted: true, idempotent: true, grant: publicGrant(raced) }, { status: 200 }, corsHeaders);
    return json({ error: 'The grant could not be reserved safely. Please retry with the same idempotency key.' }, { status: 409 }, corsHeaders);
  }

  const grant = await env.DB.prepare(`SELECT id,protocol_id,status,call_ceiling,call_count,sample_target,model,token_limit,public_state_json,created_at,updated_at,completed_at
    FROM event_grants WHERE id=?`).bind(grantId).first<Parameters<typeof publicGrant>[0]>();
  return json({ accepted: true, idempotent: false, grant: grant ? publicGrant(grant) : { id: grantId, protocolId, status: 'queued' } }, { status: 202 }, corsHeaders);
}

export async function eventRun(env: Env, grantId: string, corsHeaders: Record<string, string>) {
  const row = await env.DB.prepare(`SELECT id,protocol_id,status,call_ceiling,call_count,sample_target,model,token_limit,public_state_json,created_at,updated_at,completed_at
    FROM event_grants WHERE campaign_id=? AND id=?`).bind(EVENT_CAMPAIGN_ID, grantId).first<Parameters<typeof publicGrant>[0]>();
  if (!row) return json({ error: 'Run not found.' }, { status: 404 }, corsHeaders);
  return json({ grant: publicGrant(row) }, {}, corsHeaders);
}
