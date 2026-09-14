import { createHash, timingSafeEqual } from 'node:crypto';
import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MODEL = 'gpt-4.1-mini';
const MAX_OUTPUT_TOKENS = 64;
const MAX_CALLS = 1;

function secretEquals(left: string, right: string) {
  const a = createHash('sha256').update(left).digest();
  const b = createHash('sha256').update(right).digest();
  return timingSafeEqual(a, b);
}

export async function POST(request: Request) {
  const configuredOwnerKey = process.env.AUTOLABS_ADMIN_TOKEN;
  const openaiKey = process.env.OPENAI_API_KEY;
  if (!configuredOwnerKey || !openaiKey) {
    return NextResponse.json({ error: 'The test runner is not configured.' }, { status: 503 });
  }
  const suppliedOwnerKey = request.headers.get('x-autolabs-owner-key');
  if (!suppliedOwnerKey || !secretEquals(suppliedOwnerKey, configuredOwnerKey)) {
    return NextResponse.json({ error: 'Owner key required.' }, { status: 401 });
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20_000);
  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      signal: controller.signal,
      headers: { authorization: `Bearer ${openaiKey}`, 'content-type': 'application/json' },
      body: JSON.stringify({
        model: MODEL,
        temperature: 0,
        max_tokens: MAX_OUTPUT_TOKENS,
        messages: [
          { role: 'system', content: 'You are the AutoLabs wiring check. Answer in one short sentence.' },
          { role: 'user', content: 'Return exactly: AutoLabs test call passed.' },
        ],
      }),
    });
    const payload = await response.json().catch(() => null) as { choices?: Array<{ message?: { content?: unknown } }>; id?: unknown; error?: { message?: string } } | null;
    if (!response.ok) {
      return NextResponse.json({ error: payload?.error?.message || 'The provider rejected the test call.', budgetGuard: { maxCalls: MAX_CALLS, maxOutputTokens: MAX_OUTPUT_TOKENS } }, { status: 502 });
    }
    const text = typeof payload?.choices?.[0]?.message?.content === 'string' ? payload.choices[0].message.content.slice(0, 500) : '';
    if (!text) return NextResponse.json({ error: 'The provider returned no text.', budgetGuard: { maxCalls: MAX_CALLS, maxOutputTokens: MAX_OUTPUT_TOKENS } }, { status: 502 });
    return NextResponse.json({ text, model: MODEL, requestId: typeof payload?.id === 'string' ? payload.id : undefined, budgetGuard: { maxCalls: MAX_CALLS, maxOutputTokens: MAX_OUTPUT_TOKENS, maxRunUsd: 0.01 } });
  } catch (error) {
    const message = error instanceof Error && error.name === 'AbortError' ? 'The provider timed out.' : 'The provider could not be reached.';
    return NextResponse.json({ error: message, budgetGuard: { maxCalls: MAX_CALLS, maxOutputTokens: MAX_OUTPUT_TOKENS, maxRunUsd: 0.01 } }, { status: 504 });
  } finally {
    clearTimeout(timeout);
  }
}
