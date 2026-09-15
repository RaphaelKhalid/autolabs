import { createHash, timingSafeEqual } from 'node:crypto';
import { NextResponse } from 'next/server';
import { SPONSORED_RUN_CAP_USD, validateSponsoredBudget } from '../../../lib/sponsored-budget';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const TEST_MODEL = 'gpt-5.6-luna';
const TEST_PROMPT = 'Reply with exactly the word OK.';
const MAX_OUTPUT_TOKENS = 32;
const TIMEOUT_MS = 15_000;
const MAX_REQUEST_BYTES = 2_048;

interface ResponsesPayload {
  id?: string;
  output?: Array<{ content?: Array<{ type?: string; text?: string }> }>;
  usage?: { input_tokens?: number; output_tokens?: number };
  error?: { message?: string };
}

function secretEquals(left: string, right: string) {
  const a = createHash('sha256').update(left).digest();
  const b = createHash('sha256').update(right).digest();
  return timingSafeEqual(a, b);
}

function responseText(payload: ResponsesPayload) {
  return (payload.output ?? [])
    .flatMap((item) => item.content ?? [])
    .find((item) => item.type === 'output_text' && typeof item.text === 'string')?.text;
}

function authorized(request: Request, ownerToken: string) {
  const ownerKey = request.headers.get('x-autolabs-owner-key');
  const bearer = request.headers.get('authorization');
  return Boolean(
    (ownerKey && secretEquals(ownerKey, ownerToken)) ||
    (bearer && secretEquals(bearer, `Bearer ${ownerToken}`)),
  );
}

async function validEmptyBody(request: Request) {
  const contentLength = Number(request.headers.get('content-length') ?? 0);
  if (!Number.isFinite(contentLength) || contentLength > MAX_REQUEST_BYTES) return false;
  const raw = await request.text();
  if (new TextEncoder().encode(raw).byteLength > MAX_REQUEST_BYTES || raw.trim() === '') return true;
  try {
    const value = JSON.parse(raw) as unknown;
    return Boolean(value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 0);
  } catch {
    return false;
  }
}

export async function POST(request: Request) {
  const ownerToken = process.env.AUTOLABS_ADMIN_TOKEN;
  const apiKey = process.env.OPENAI_API_KEY;
  if (!ownerToken || !apiKey) {
    return NextResponse.json({ error: 'The sponsored test is not configured.' }, { status: 503 });
  }
  if (!authorized(request, ownerToken)) {
    return NextResponse.json({ error: 'Owner key required.' }, { status: 401 });
  }
  if (!(await validEmptyBody(request))) {
    return NextResponse.json({ error: 'The test request accepts no prompt or model overrides.' }, { status: 400 });
  }

  // The reservation is deliberately conservative and is reported separately
  // from provider usage. A timeout may still have incurred an unknown charge.
  const budget = validateSponsoredBudget({ reservedUsd: 0.01 });
  if (!budget.withinCap) {
    return NextResponse.json({ error: 'The sponsored test budget is unavailable.' }, { status: 409 });
  }

  try {
    const response = await fetch('https://api.openai.com/v1/responses', {
      method: 'POST',
      signal: AbortSignal.timeout(TIMEOUT_MS),
      headers: { authorization: `Bearer ${apiKey}`, 'content-type': 'application/json' },
      body: JSON.stringify({
        model: TEST_MODEL,
        input: TEST_PROMPT,
        max_output_tokens: MAX_OUTPUT_TOKENS,
        store: false,
      }),
    });
    const payload = await response.json().catch(() => ({})) as ResponsesPayload;
    if (!response.ok) {
      return NextResponse.json({ error: 'The provider rejected the sponsored test request.' }, { status: 502 });
    }
    const output = responseText(payload);
    if (!output) {
      return NextResponse.json({ error: 'The provider returned no test output.' }, { status: 502 });
    }
    return NextResponse.json({
      ok: true,
      model: TEST_MODEL,
      output,
      usage: {
        inputTokens: payload.usage?.input_tokens ?? null,
        outputTokens: payload.usage?.output_tokens ?? null,
        reported: Boolean(payload.usage),
      },
      budget: {
        capUsd: SPONSORED_RUN_CAP_USD,
        reservationUsd: 0.01,
        accounting: payload.usage ? 'provider-reported usage; reconcile the provider invoice later' : 'usage unavailable; charge remains uncertain',
      },
    }, { headers: { 'cache-control': 'no-store' } });
  } catch (error) {
    const timedOut = error instanceof DOMException && error.name === 'TimeoutError';
    return NextResponse.json({ error: timedOut ? 'The sponsored test timed out; any provider charge is uncertain.' : 'The sponsored test provider is unavailable.' }, { status: timedOut ? 504 : 502 });
  }
}


