import { getState, globalSpend, nowIso } from './db';
import type { PublicEvent } from './types';

const CHECKPOINT_ROUNDS = [35, 40, 45, 50] as const;
const CHECKPOINT_RECIPIENT = 'raphaelbahadurkhan@gmail.com';

export function checkpointRoundsThrough(completedRound: number) {
  return CHECKPOINT_ROUNDS.filter((round) => round <= completedRound);
}

export async function queueCheckpointSummary(db: D1Database, runId: string, checkpointRound: number) {
  if (!CHECKPOINT_ROUNDS.includes(checkpointRound as (typeof CHECKPOINT_ROUNDS)[number])) return false;
  const roundStart = checkpointRound - 4;
  const [state, spent, eventRows, jobRows] = await Promise.all([
    getState(db, runId),
    globalSpend(db),
    db.prepare(`SELECT round,kind,title,summary FROM events
        WHERE run_id=? AND visible=1 AND round BETWEEN ? AND ?
        ORDER BY seq`)
      .bind(runId, roundStart, checkpointRound)
      .all<{ round: number; kind: PublicEvent['kind']; title: string; summary: string }>(),
    db.prepare(`SELECT status,COUNT(*) AS count FROM jobs WHERE run_id=? GROUP BY status ORDER BY status`)
      .bind(runId)
      .all<{ status: string; count: number }>(),
  ]);
  const publicEvents = eventRows.results;
  const failures = publicEvents.filter((event) => event.kind === 'error');
  const notable = publicEvents
    .filter((event) => event.kind === 'candidate' || event.kind === 'research' || event.kind === 'tool')
    .slice(-8);
  const bestSupport = Array.isArray(state.bestSupport) ? state.bestSupport.map(Number) : [];
  const bestResult = Boolean(state.bestVerified)
    ? `${String(state.bestLabel ?? 'Verified result')} · support ${JSON.stringify(bestSupport)}`
    : 'No verified result yet.';
  const notableText = notable.length
    ? notable.map((event) => `- R${event.round} ${event.title}: ${event.summary}`).join('\n')
    : '- No new public research finding was recorded in this interval.';
  const failureText = failures.length
    ? failures.map((event) => `- R${event.round} ${event.title}: ${event.summary}`).join('\n')
    : '- No public failure or exhausted-retry event was recorded in this interval.';
  const queueText = jobRows.results.length
    ? jobRows.results.map((row) => `${row.status}=${Number(row.count)}`).join(', ')
    : 'no jobs recorded';
  const subject = `Autolabs Erdős 885 · rounds ${roundStart}–${checkpointRound}`;
  const body = [
    subject,
    `Run: ${runId}`,
    `Best verified result: ${bestResult}`,
    `SOTA frontier improved: ${Boolean(state.sotaImproved) ? 'yes' : 'no'}`,
    `OpenAI spend: $${spent.toFixed(4)}`,
    `Compute queue: ${queueText}`,
    '',
    'Notable public methods/findings:',
    notableText,
    '',
    'Failures/retries:',
    failureText,
  ].join('\n');
  const inserted = await db.prepare(`INSERT OR IGNORE INTO checkpoint_email_outbox
      (run_id,checkpoint_round,round_start,recipient,subject,body_text,status,attempts,created_at)
      VALUES(?,?,?,?,?,?,'pending',0,?)`)
    .bind(runId, checkpointRound, roundStart, CHECKPOINT_RECIPIENT, subject, body, nowIso())
    .run();
  return Number(inserted.meta.changes ?? 0) === 1;
}

export async function queueCheckpointSummariesThrough(db: D1Database, runId: string, completedRound: number) {
  for (const checkpointRound of checkpointRoundsThrough(completedRound)) {
    await queueCheckpointSummary(db, runId, checkpointRound);
  }
}
