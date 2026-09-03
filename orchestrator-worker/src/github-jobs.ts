import type { AgentId, ResearchReport } from './types';
import { addEvent, nowIso } from './db';
import { secondHalfPolicy } from './second-half-policy';

const MAX_ACTIVE_JOBS = 8;
// At eight dispatched jobs per round, a 50-round competition can legitimately
// need 400 jobs. Keep concurrency bounded while allowing every round to compute.
const MAX_RUN_JOBS = 400;
const AGENT_ORDER: Record<AgentId, number> = { mira: 0, pip: 1, orum: 2, solvi: 3, tess: 4 };

async function resolveSourceRevision(repository: string, githubToken: string) {
  const response = await fetch(`https://api.github.com/repos/${repository}/commits/main`, {
    headers: {
      authorization: `Bearer ${githubToken}`,
      accept: 'application/vnd.github+json',
      'user-agent': 'autolabs-orchestrator',
      'x-github-api-version': '2022-11-28',
    },
  });
  if (!response.ok) throw new Error(`GitHub source revision ${response.status}`);
  const body = await response.json() as { sha?: string };
  if (!body.sha || !/^[0-9a-f]{40}$/.test(body.sha)) throw new Error('GitHub returned an invalid source revision.');
  return body.sha;
}

export async function reapStaleJobs(db: D1Database, runId: string) {
  await db.prepare(`UPDATE jobs
    SET status='failed', error='Compute lease expired before a callback was received.', completed_at=?
    WHERE run_id=? AND status IN ('queued','dispatching','running')
      AND julianday(created_at) < julianday('now','-55 minutes')`)
    .bind(nowIso(), runId)
    .run();
}

export async function scheduleJobs(options: {
  db: D1Database;
  githubToken: string;
  repository: string;
  runId: string;
  round: number;
  agentId: AgentId;
  reports: ResearchReport['proposedJobs'];
}) {
  const counts = await options.db.prepare(`SELECT
      SUM(CASE WHEN status IN ('queued','dispatching','running') THEN 1 ELSE 0 END) AS active,
      COUNT(*) AS total
    FROM jobs WHERE run_id=?`)
    .bind(options.runId)
    .first<{ active: number | null; total: number }>();
  const activeCapacity = Math.max(0, MAX_ACTIVE_JOBS - Number(counts?.active ?? 0));
  const totalCapacity = Math.max(0, MAX_RUN_JOBS - Number(counts?.total ?? 0));
  const permitted = options.round < 26
    ? options.reports
    : options.reports.filter((job, index, reports) => {
      if (job.jobType !== 'divisor_completion') return true;
      const policy = secondHalfPolicy(options.agentId, options.round);
      return policy.designatedDivisorVerifier === options.agentId
        && reports.findIndex((candidate) => candidate.jobType === 'divisor_completion') === index;
    });
  const jobs = permitted.slice(0, Math.min(3, activeCapacity, totalCapacity));
  const sourceSha = jobs.length ? await resolveSourceRevision(options.repository, options.githubToken) : null;

  for (let index = 0; index < jobs.length; index += 1) {
    const job = jobs[index];
    const id = `${options.runId}-r${options.round}-${options.agentId}-${index}`;
    await options.db.prepare(`INSERT OR IGNORE INTO jobs(id,run_id,agent_id,round,job_type,params_json,status,created_at) VALUES(?,?,?,?,?,?,?,?)`)
      .bind(id, options.runId, options.agentId, options.round, job.jobType, JSON.stringify(job.params), 'queued', nowIso())
      .run();

    const lease = await options.db.prepare(`UPDATE jobs SET status='dispatching' WHERE id=? AND status='queued'`)
      .bind(id)
      .run();
    if (Number(lease.meta.changes ?? 0) !== 1) continue;

    try {
      const response = await fetch(`https://api.github.com/repos/${options.repository}/dispatches`, {
        method: 'POST',
        headers: {
          authorization: `Bearer ${options.githubToken}`,
          accept: 'application/vnd.github+json',
          'content-type': 'application/json',
          'user-agent': 'autolabs-orchestrator',
          'x-github-api-version': '2022-11-28',
        },
        body: JSON.stringify({
          event_type: 'autolabs_math_job',
          client_payload: {
            id,
            runId: options.runId,
            agentId: options.agentId,
            round: options.round,
            jobType: job.jobType,
            sourceSha,
            params: job.params,
          },
        }),
      });

      if (!response.ok) {
        await options.db.prepare(`UPDATE jobs SET status='failed',error=?,completed_at=? WHERE id=? AND status='dispatching'`)
          .bind(`GitHub dispatch ${response.status}`, nowIso(), id)
          .run();
        continue;
      }

      await options.db.prepare(`UPDATE jobs SET status='running' WHERE id=? AND status='dispatching'`).bind(id).run();
      await addEvent(options.db, options.runId, options.round * 1000 + 300 + AGENT_ORDER[options.agentId] * 10 + index, {
        at: nowIso(),
        round: options.round,
        phase: 'meeting',
        agentId: options.agentId,
        kind: 'tool',
        title: `Code job queued · ${job.jobType}`,
        summary: job.reason,
        payload: { id, jobType: job.jobType, sourceSha, params: job.params, status: 'running' },
        visible: true,
      });
    } catch (error) {
      await options.db.prepare(`UPDATE jobs SET status='failed',error=?,completed_at=? WHERE id=? AND status='dispatching'`)
        .bind(error instanceof Error ? error.message : 'GitHub dispatch failed', nowIso(), id)
        .run();
    }
  }
}
