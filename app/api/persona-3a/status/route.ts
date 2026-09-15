import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const TELEMETRY_URL = 'https://afterlight-api.raphaelbahadurkhan.workers.dev/api/studies/persona-discovery/status';

export async function GET() {
  const workerUrl = process.env.ORCHESTRATOR_URL ?? process.env.NEXT_PUBLIC_ORCHESTRATOR_URL;
  if (!workerUrl) return NextResponse.json({ error: 'Experiment 3A status is not configured.' }, { status: 503 });
  try {
    const [response, telemetryResponse] = await Promise.all([
      fetch(`${workerUrl.replace(/\/$/, '')}/api/persona-3a/status`, { cache: 'no-store', signal: AbortSignal.timeout(10_000) }),
      fetch(TELEMETRY_URL, { cache: 'no-store', signal: AbortSignal.timeout(10_000) }),
    ]);
    const payload = await response.json().catch(() => ({})) as { launch?: { requestedAt?: string } };
    const telemetry = await telemetryResponse.json().catch(() => null) as Record<string, unknown> | null;
    const requestedAt = payload.launch?.requestedAt ? Date.parse(payload.launch.requestedAt) : 0;
    const telemetryAt = typeof telemetry?.updatedAt === 'string' ? Date.parse(telemetry.updatedAt) : 0;
    return NextResponse.json({ ...payload, telemetry: telemetryResponse.ok && telemetryAt >= requestedAt ? telemetry : null }, { status: response.status, headers: { 'cache-control': 'no-store' } });
  } catch {
    return NextResponse.json({ error: 'Experiment 3A status is unavailable.' }, { status: 502 });
  }
}