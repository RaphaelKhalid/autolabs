import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  const workerUrl = process.env.ORCHESTRATOR_URL ?? process.env.NEXT_PUBLIC_ORCHESTRATOR_URL;
  if (!workerUrl) return NextResponse.json({ run: null, error: 'Experiment 3C status is not configured.' }, { status: 503 });
  const runId = new URL(request.url).searchParams.get('runId');
  const suffix = runId ? `?runId=${encodeURIComponent(runId)}` : '';
  try {
    const response = await fetch(`${workerUrl.replace(/\/$/, '')}/api/persona-3c/status${suffix}`, { cache: 'no-store', signal: AbortSignal.timeout(10_000) });
    const payload = await response.json().catch(() => ({ run: null }));
    return NextResponse.json(payload, { status: response.status, headers: { 'cache-control': 'no-store' } });
  } catch {
    return NextResponse.json({ run: null, error: 'The Experiment 3C orchestrator is unavailable.' }, { status: 502 });
  }
}
