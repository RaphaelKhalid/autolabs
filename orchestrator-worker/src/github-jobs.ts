import type { AgentId, ResearchReport } from './types';
import { addEvent, nowIso } from './db';

export async function scheduleJobs(options: {
  db: D1Database;
  githubToken: string;
  repository: string;
  callbackUrl: string;
  callbackSecret: string;
  runId: string;
  round: number;
  agentId: AgentId;
  reports: ResearchReport['proposedJobs'];
}) {
  const jobs = options.reports.slice(0, 3);
  for (let index = 0; index < jobs.length; index += 1) {
    const job = jobs[index];
    const id = `${options.runId}-r${options.round}-${options.agentId}-${index}`;
    await options.db.prepare(`INSERT OR IGNORE INTO jobs(id,run_id,agent_id,round,job_type,params_json,status,created_at) VALUES(?,?,?,?,?,?,?,?)`)
      .bind(id, options.runId, options.agentId, options.round, job.jobType, JSON.stringify(job.params), 'queued', nowIso()).run();
    const existing = await options.db.prepare('SELECT status FROM jobs WHERE id=?').bind(id).first<{ status: string }>();
    if (existing?.status !== 'queued') continue;
    const response = await fetch(`https://api.github.com/repos/${options.repository}/dispatches`, {
      method: 'POST',
      headers: { authorization: `Bearer ${options.githubToken}`, accept: 'application/vnd.github+json', 'content-type': 'application/json', 'user-agent': 'autolabs-orchestrator' },
      body: JSON.stringify({ event_type: 'autolabs_math_job', client_payload: { id, runId: options.runId, agentId: options.agentId, round: options.round, jobType: job.jobType, params: job.params, callbackUrl: options.callbackUrl } }),
    });
    if (!response.ok) {
      await options.db.prepare(`UPDATE jobs SET status='failed',error=?,completed_at=? WHERE id=?`).bind(`GitHub dispatch ${response.status}`, nowIso(), id).run();
      continue;
    }
    await options.db.prepare(`UPDATE jobs SET status='running' WHERE id=?`).bind(id).run();
    await addEvent(options.db, options.runId, options.round * 1000 + 300 + AGENT_ORDER[options.agentId] * 10 + index, {
      at: nowIso(), round: options.round, phase: 'research', agentId: options.agentId, kind: 'tool', title: `Code job queued · ${job.jobType}`, summary: job.reason, visible: true,
    });
  }
}

const AGENT_ORDER: Record<AgentId, number> = { mira: 0, pip: 1, orum: 2, solvi: 3, tess: 4 };
