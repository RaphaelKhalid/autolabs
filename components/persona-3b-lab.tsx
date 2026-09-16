'use client';

import Link from 'next/link';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { ArrowUpRight, CircleDollarSign, FlaskConical, Radio, RefreshCw, ShieldCheck } from 'lucide-react';
import { AlienForm } from '@/components/autolabs-observatory';
import { demoAgents, type ResearchAgent } from '@/lib/experiment';
import {
  fetchPersona3BStatus, persona3bContract, persona3bPhaseLabel, phaseDisplay, startPersona3B,
  type Persona3BRun, type Persona3BShard, type Persona3BStatus,
} from '@/lib/persona-3b';

const agentPositions = [
  { x: 14, y: 35 }, { x: 32, y: 68 }, { x: 51, y: 31 }, { x: 70, y: 68 }, { x: 87, y: 37 },
];
const pigments = ['#f27a4f', '#a6d879', '#b39af4', '#42d6df', '#f47b91'];

function isActive(shard: Persona3BShard | undefined) {
  return shard?.status === 'claimed' || shard?.status === 'running';
}

function statusTone(status: string) {
  if (status === 'complete') return 'is-complete';
  if (status === 'failed') return 'is-failed';
  if (status === 'blocked') return 'is-blocked';
  return 'is-live';
}

function number(value: number) { return new Intl.NumberFormat('en-US').format(value); }
function dollars(value: number) { return `$${value.toFixed(2)}`; }

function shardForAgent(shards: Persona3BShard[], agentId: string, run: Persona3BRun | null) {
  const own = shards.filter((shard) => shard.agentId === agentId);
  const phaseOrder = run?.status === 'awaiting_adjudication'
    ? ['disagreement', 'repeat', 'primary', 'synthesis']
    : run?.status === 'synthesizing'
      ? ['synthesis', 'disagreement', 'repeat', 'primary']
      : ['primary', 'repeat', 'disagreement', 'synthesis'];
  for (const phase of phaseOrder) {
    const match = own.find((shard) => shard.phase === phase && shard.status !== 'blocked');
    if (match) return match;
  }
  return own[0];
}

function phaseTotals(shards: Persona3BShard[], phase: Persona3BShard['phase']) {
  const rows = shards.filter((shard) => shard.phase === phase);
  return { done: rows.reduce((sum, shard) => sum + shard.completedRecords, 0), total: rows.reduce((sum, shard) => sum + shard.expectedRecords, 0), active: rows.filter((shard) => isActive(shard)).length };
}

function AgentCard({ agent, index, shard }: { agent: ResearchAgent; index: number; shard?: Persona3BShard }) {
  const active = isActive(shard);
  const position = agentPositions[index];
  const label = shard ? `${phaseDisplay(shard.phase)} · ${shard.status}` : 'Awaiting assignment';
  return <article className={`persona3b-agent ${active ? 'is-active' : ''} ${shard ? statusTone(shard.status) : 'is-idle'}`} style={{ left: `${position.x}%`, top: `${position.y}%`, '--agent-color': pigments[index] } as React.CSSProperties} aria-label={`${agent.name}: ${label}`}>
    <div className="persona3b-agent-aura" aria-hidden="true" />
    <AlienForm agent={{ id: agent.id, name: agent.name, color: pigments[index] }} index={index} meeting={shard?.phase === 'disagreement'} />
    <div className="persona3b-agent-card"><b>{agent.name}</b><span>{phaseDisplay(shard?.phase ?? 'primary')}</span><small>{shard ? `${number(shard.completedRecords)} / ${number(shard.expectedRecords)} records` : 'No shard leased'}</small></div>
  </article>;
}

function PhaseCard({ label, phase, shards }: { label: string; phase: Persona3BShard['phase']; shards: Persona3BShard[] }) {
  const totals = phaseTotals(shards, phase);
  return <article className={`persona3b-phase persona3b-phase--${phase}`}>
    <div className="persona3b-phase-top"><span>{label}</span><strong>{phase === 'disagreement' ? (totals.total ? `CONDITIONAL · ${number(totals.done)} / ${number(totals.total)}` : 'CONDITIONAL') : totals.total ? `${number(totals.done)} / ${number(totals.total)}` : 'GATED'}</strong></div>
    <div className="persona3b-phase-track" role="progressbar" aria-label={`${label} progress`} aria-valuemin={0} aria-valuemax={totals.total || 1} aria-valuenow={totals.done}><i style={{ width: totals.total ? `${Math.min(100, totals.done / totals.total * 100)}%` : '0%' }} /></div>
    <small>{totals.active ? `${totals.active} agent${totals.active === 1 ? '' : 's'} working` : phase === 'primary' ? 'Ready to lease' : phase === 'disagreement' ? 'Uses shared retry / adjudication reserve' : 'Unlocks after the prior pass'}</small>
  </article>;
}

export function Persona3BLab() {
  const [status, setStatus] = useState<Persona3BStatus>({ run: null, shards: [], events: [] });
  const [connection, setConnection] = useState<'loading' | 'ready' | 'error'>('loading');
  const [ownerKey, setOwnerKey] = useState('');
  const [startMessage, setStartMessage] = useState('');
  const [starting, setStarting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const reload = useCallback(async () => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 12_000);
    try { setStatus(await fetchPersona3BStatus(controller.signal)); setConnection('ready'); }
    catch { setConnection('error'); }
    finally { window.clearTimeout(timeout); }
  }, []);

  useEffect(() => { void reload(); }, [reload]);
  useEffect(() => {
    const terminal = status.run && ['complete', 'failed', 'cancelled'].includes(status.run.status);
    const timer = window.setInterval(() => { if (!terminal) void reload(); }, terminal ? 15_000 : 3_000);
    return () => window.clearInterval(timer);
  }, [reload, status.run]);

  const run = status.run;
  const primary = useMemo(() => phaseTotals(status.shards, 'primary'), [status.shards]);
  const agents = useMemo(() => demoAgents.map((agent) => ({ agent, shard: shardForAgent(status.shards, agent.id, run) })), [run, status.shards]);
  const displayEvents = status.events.slice(-8).reverse();
  const progress = run ? Math.min(100, run.callCount / run.callCeiling * 100) : 0;

  async function onStart(event: React.FormEvent) {
    event.preventDefault();
    if (!ownerKey.trim() || starting) return;
    setStarting(true); setStartMessage('');
    try { const next = await startPersona3B(ownerKey.trim()); setStatus((current) => ({ ...current, ...next })); setOwnerKey(''); setStartMessage('Scoring run queued.'); }
    catch (error) { setStartMessage(error instanceof Error ? error.message : 'The run could not be started.'); }
    finally { setStarting(false); }
  }

  return <main className="persona3b-page">
    <section className="persona3b-monitor" aria-live="polite" aria-label="Experiment 3B run status">
      <div className="persona3b-monitor-top"><div><p className="persona-eyebrow">AUTOLABS RUN RECEIPT</p><h2>{run ? persona3bPhaseLabel(run) : 'Awaiting the owner start'}</h2></div><span className={`persona3b-status ${run ? statusTone(run.status) : 'is-idle'}`}><i />{run?.status ?? (connection === 'error' ? 'unavailable' : 'no run')}</span></div>
      <div className="persona3b-metrics"><div><span>SOURCE OUTPUTS</span><strong>{number(persona3bContract.totalRecords)} rows</strong></div><div><span>PAIRED COMPARISONS</span><strong>{number(primary.done)} / {number(persona3bContract.pairedComparisons)}</strong></div><div><span>GRADING CALLS</span><strong>{number(run?.callCount ?? 0)} / {number(run?.callCeiling ?? persona3bContract.callCeiling)}</strong>{run?.amendment ? <small>amended ceiling · manifest {number(run.amendment.originalCallCeiling)}</small> : null}</div><div><span>SPEND / CAP</span><strong>{dollars(run?.spentUsd ?? 0)} / {dollars(run?.budgetUsd ?? persona3bContract.budgetUsd)}</strong></div><div><span>MODEL</span><strong>{run?.model ?? persona3bContract.model} · high</strong></div></div>
      <div className="persona3b-progress"><div><span>BOUND CALL PROGRESS</span><strong>{Math.round(progress)}%</strong></div><div className="persona3b-progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}><i style={{ width: `${progress}%` }} /></div><small>Budget guard is enforced by the worker before every provider request.</small></div>
      {run?.error && <p className="persona3b-error" role="alert">{run.error}</p>}
    </section>

    <section className="persona3b-room" aria-label="Animated scoring room"><div className="persona3b-room-grid" aria-hidden="true" /><div className="persona3b-room-caption"><span><Radio size={12} /> LIVE TELEMETRY</span><small>{connection === 'ready' ? 'worker connected · polling 3 s' : connection === 'loading' ? 'connecting to worker' : 'worker unavailable'}</small></div>{agents.map(({ agent, shard }, index) => <AgentCard key={agent.id} agent={agent} index={index} shard={shard} />)}<div className="persona3b-room-center"><FlaskConical size={18} /><span>BLINDED<br />SCORING</span></div></section>

    <section className="persona3b-header persona3b-context"><div><p className="persona-eyebrow">EXPERIMENT 3B · LIVE SCORING ROOM</p><h1>Do any of the 780 existing responses carry a persona relevant direction?</h1><p className="persona3b-lede">Five AutoLabs researchers coordinate blinded Luna High judging over the completed Experiment 3A screen. The room reports work as it happens; the scientific conclusion comes only after unblinding and analysis.</p></div><div className="persona3b-header-links"><a href="https://github.com/RaphaelKhalid/autolabs/tree/main/research/experiment-003b" target="_blank" rel="noreferrer">Protocol + provenance <ArrowUpRight size={13} /></a></div></section>
    <section className="persona3b-phases" aria-label="Scoring passes"><div className="persona3b-section-head"><div><p className="persona-eyebrow">PARALLEL WORKFLOW</p><h2>Independent passes, then judgment</h2></div><span>{run ? persona3bPhaseLabel(run) : 'Not started'}</span></div><div className="persona3b-phase-grid"><PhaseCard label="Primary · 768 paired comparisons" phase="primary" shards={status.shards} /><PhaseCard label="Repeat · 117 rows" phase="repeat" shards={status.shards} /><PhaseCard label="Disagreement review · conditional" phase="disagreement" shards={status.shards} /><PhaseCard label="Final analyst" phase="synthesis" shards={status.shards} /></div><p className="persona3b-method-note"><ShieldCheck size={15} /> Primary scoring uses 768 paired comparisons; the independent repeat pass uses 117 judgments. Disagreement review is conditional and shares the remaining 129 attempts with bounded retries. Feature IDs, signs and candidate names stay in the private mapping.</p></section>

    <section className="persona3b-ledger"><div className="persona3b-section-head"><div><p className="persona-eyebrow">EVENT LEDGER</p><h2>What the lab has recorded</h2></div><button type="button" onClick={async () => { setRefreshing(true); await reload(); setRefreshing(false); }} disabled={refreshing}><RefreshCw size={14} className={refreshing ? 'is-spinning' : ''} /> Refresh</button></div>{displayEvents.length ? <div className="persona3b-events">{displayEvents.map((event, index) => <article key={`${event.id ?? event.at}-${index}`}><time dateTime={event.at}>{new Date(event.at).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}</time><div><b>{event.title}</b><p>{event.summary}</p></div><span>{event.agentId ?? event.phase ?? 'system'}</span></article>)}</div> : <p className="persona3b-empty">No public events yet. A queued run will appear here after the worker records its first lease.</p>}</section>

    <section className="persona3b-outcome"><div><p className="persona-eyebrow">INTERPRETATION GATE</p><h2>{run?.status === 'complete' ? 'Scoring is complete; analysis is the next record.' : 'Shortlist: pending scoring'}</h2><p>At most three candidates may advance, and zero is a valid result. Confirmation remains separately gated and cannot launch from this room.</p></div><div className="persona3b-outcome-states"><span><i className={run?.status === 'complete' ? 'is-on' : ''} />SHORTLIST {run?.status === 'complete' ? 'PENDING ANALYSIS' : 'PENDING'}</span><span><i />CONFIRMATION PENDING</span></div></section>

    <section className="persona3b-control"><div><p className="persona-eyebrow">OWNER CONTROL</p><h2>Start the frozen scoring run</h2><p>One start creates an idempotent run with the existing 780 responses, 768 paired comparisons, 117 repeat judgments, a shared 129-attempt retry/adjudication allowance, and the approved $10 hard cap. No GPU generation or response regeneration is attached.</p></div>{run ? <div className="persona3b-control-receipt"><strong>{run.id}</strong><span>Owner start received · {run.status}</span></div> : <form onSubmit={onStart}><label htmlFor="persona3b-owner-key">Owner key</label><div><input id="persona3b-owner-key" type="password" autoComplete="off" value={ownerKey} onChange={(event) => setOwnerKey(event.target.value)} placeholder="Required to start" /><button type="submit" disabled={starting || !ownerKey.trim()}>{starting ? 'Queueing…' : 'Start scoring'}</button></div>{startMessage && <p role="status">{startMessage}</p>}</form>}</section>

    <footer className="persona3b-footer"><span>Source artifact {persona3bContract.sourceArtifactHash.slice(0, 12)}… · {persona3bContract.totalRecords} records · {persona3bContract.model} high</span><Link href="/experiments/persona-discovery">Read the distinct 3A development record ↗</Link></footer>
  </main>;
}
