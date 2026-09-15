import { createHash, timingSafeEqual } from 'node:crypto';
import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const STUDY_ID = 'experiment-003a-v1';
const PHASE = 'development';
const MAX_RUNTIME_SECONDS = 6_600;

function secretEquals(left: string, right: string) {
  const a = createHash('sha256').update(left).digest();
  const b = createHash('sha256').update(right).digest();
  return timingSafeEqual(a, b);
}

async function requestBody(request: Request) {
  const contentLength = Number(request.headers.get('content-length') ?? 0);
  if (!Number.isFinite(contentLength) || contentLength > 2_048) return {};
  const text = await request.text();
  if (new TextEncoder().encode(text).byteLength > 2_048 || text.trim() === '') return {};
  try {
    const value = JSON.parse(text) as unknown;
    return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

export async function POST(request: Request) {
  const ownerToken = process.env.AUTOLABS_3A_START_TOKEN ?? process.env.AUTOLABS_ADMIN_TOKEN;
  const workerToken = process.env.PERSONA_3A_WORKER_TOKEN ?? process.env.AUTOLABS_ADMIN_TOKEN;
  const workerUrl = process.env.ORCHESTRATOR_URL ?? process.env.NEXT_PUBLIC_ORCHESTRATOR_URL;
  if (!ownerToken || !workerToken || !workerUrl) return NextResponse.json({ error: 'Experiment 3A control is not configured.' }, { status: 503 });

  const supplied = request.headers.get('x-autolabs-owner-key') ?? '';
  if (!supplied || !secretEquals(supplied, ownerToken)) return NextResponse.json({ error: 'Owner key required.' }, { status: 401 });

  const raw = await requestBody(request);
  const suppliedKey = raw.idempotencyKey;
  const idempotencyKey = typeof suppliedKey === 'string' && suppliedKey.length >= 16 && suppliedKey.length <= 120
    ? suppliedKey
    : crypto.randomUUID();

  try {
    const response = await fetch(`${workerUrl.replace(/\/$/, '')}/api/persona-3a/start`, {
      method: 'POST',
      signal: AbortSignal.timeout(10_000),
      headers: { authorization: `Bearer ${workerToken}`, 'content-type': 'application/json' },
      body: JSON.stringify({ studyId: STUDY_ID, phase: PHASE, maxRuntimeSeconds: MAX_RUNTIME_SECONDS, idempotencyKey }),
    });
    const payload = await response.json().catch(() => ({}));
    return NextResponse.json(payload, { status: response.status, headers: { 'cache-control': 'no-store' } });
  } catch {
    return NextResponse.json({ error: 'The orchestrator is unavailable; no launch result was assumed.' }, { status: 502 });
  }
}

