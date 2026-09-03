'use client';

import {
  Activity, ArrowUpRight, BookOpen, BrainCircuit, ChevronLeft, ChevronRight,
  CircleDollarSign, FlaskConical, Github, Pause, Play, Radio, ShieldCheck,
  Sparkles, TimerReset, Trophy, Volume2, VolumeX, X, Zap,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { fetchExperiment } from '@/lib/api';
import {
  formatCountdown, initialExperiment, supportLabel,
  type ExperimentEvent, type ExperimentState, type ResearchAgent, type ScientificReport,
} from '@/lib/experiment';

declare global {
  interface Document {
    modelContext?: {
      registerTool(tool: {
        name: string;
        title?: string;
        description: string;
        inputSchema: object;
        annotations?: { readOnlyHint?: boolean; untrustedContentHint?: boolean };
        execute(input: unknown): unknown | Promise<unknown>;
      }, options?: { signal?: AbortSignal }): void | Promise<void>;
    };
  }
}

const pigments = ['#9a4f36', '#657a4e', '#60528b', '#27747a', '#a14f59'];
const researchPositions = [
  { x: 13, y: 34 }, { x: 32, y: 66 }, { x: 51, y: 31 },
  { x: 71, y: 66 }, { x: 87, y: 36 },
];
const meetingPositions = [
  { x: 34, y: 47 }, { x: 42, y: 67 }, { x: 51, y: 31 },
  { x: 60, y: 67 }, { x: 68, y: 47 },
];

function phaseTitle(state: ExperimentState) {
  if (state.phase === 'research') return 'Private research interval';
  if (state.phase === 'meeting') return 'Simultaneous public colloquium';
  if (state.phase === 'ribbon') return 'Opening ceremony';
  if (state.phase === 'eureka') return 'Exact certificate discovered';
  if (state.phase === 'complete') return 'Experiment concluded';
  if (state.phase === 'budget-stop') return 'Budget reserve reached';
  if (state.phase === 'error') return 'Engine paused by an error';
  return 'Awaiting the ribbon';
}

function EventIcon({ event }: { event: ExperimentEvent }) {
  if (event.kind === 'candidate') return <Sparkles size={14} />;
  if (event.kind === 'meeting') return <BrainCircuit size={14} />;
  if (event.kind === 'tool') return <FlaskConical size={14} />;
  if (event.kind === 'budget') return <CircleDollarSign size={14} />;
  return <Activity size={14} />;
}


function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function asItems(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function readable(value: unknown) {
  if (typeof value === 'string') return value;
  try { return JSON.stringify(value, null, 2); } catch { return String(value); }
}

function supportFrom(value: unknown): number[] | null {
  if (!Array.isArray(value) || value.length !== 5 || !value.every((item) => typeof item === 'number' && Number.isFinite(item))) return null;
  return value as number[];
}

function strongerSupport(left: number[], right: number[]) {
  const leftSorted = [...left].sort((a, b) => a - b);
  const rightSorted = [...right].sort((a, b) => a - b);
  for (let index = 0; index < 5; index += 1) {
    if (leftSorted[index] !== rightSorted[index]) return leftSorted[index] > rightSorted[index];
  }
  return false;
}

function EvidenceList({ label, values }: { label: string; values: unknown[] }) {
  if (!values.length) return null;
  return <section><span>{label}</span><ul>{values.map((value, index) => <li key={`${label}-${index}`}>{readable(value)}</li>)}</ul></section>;
}

function EventEvidence({ event }: { event: ExperimentEvent }) {
  const payload = asRecord(event.payload);
  if (!Object.keys(payload).length) return null;
  const structured = event.kind === 'research' || event.kind === 'meeting';
  return (
    <details className="event-evidence">
      <summary>Open evidence</summary>
      {structured ? <div>
        <EvidenceList label="CLAIMS" values={asItems(payload.claims)} />
        <EvidenceList label="EQUATIONS" values={asItems(payload.equations)} />
        <EvidenceList label="SOURCE ANCHORS" values={asItems(payload.citations)} />
        <EvidenceList label="FAILED AVENUES" values={asItems(payload.failedAvenues)} />
        <EvidenceList label="EXACT CHECKS" values={asItems(payload.verifiedCandidates)} />
        <EvidenceList label="COMPUTE JOBS" values={asItems(payload.proposedJobs)} />
        <EvidenceList label="NEXT QUESTIONS" values={asItems(payload.nextQuestions)} />
        <EvidenceList label="AGREEMENTS" values={asItems(payload.agreements)} />
        <EvidenceList label="OBJECTIONS" values={asItems(payload.objections)} />
        <EvidenceList label="COLLABORATION CREDIT" values={asItems(payload.collaborationCredits)} />
      </div> : <pre>{readable(payload)}</pre>}
    </details>
  );
}

function ScientificReportView({ report, workerUrl }: { report: ScientificReport; workerUrl?: string }) {
  const outcome = report.result.k5Proved ? 'k = 5 proved' : report.result.sotaImproved ? 'frontier improved' : report.outcome === 'budget-stop' ? 'budget-safe stop' : 'search completed';
  const base = workerUrl?.replace(/\/$/, '');
  return (
    <section id="report" className="scientific-report">
      <div className="record-heading">
        <p className="section-index">03 / TERMINAL REPORT</p>
        <h2>The sealed record,<br /><em>opened in full.</em></h2>
        <p>Every public claim, exact certificate, failed path, compute job and formerly private plan is retained as a reproducible scientific object.</p>
      </div>
      <div className="report-outcome">
        <span>OUTCOME</span><strong>{outcome}</strong>
        <p>{report.roundsCompleted} of {report.targetRounds} rounds completed · best support {supportLabel(report.result.bestSupport)}</p>
        <div><i className={report.result.k5Proved ? 'is-yes' : ''}>k = 5 {report.result.k5Proved ? 'certified' : 'not certified'}</i><i className={report.result.sotaImproved ? 'is-yes' : ''}>SOTA {report.result.sotaImproved ? 'improved' : 'unchanged'}</i></div>
      </div>
      <div className="report-grid">
        <section><span>SOURCE ANCHORS</span><strong>{report.scientificRecord.citations.length}</strong><ul>{report.scientificRecord.citations.map((item, index) => <li key={index}>{item}</li>)}</ul></section>
        <section><span>FAILED AVENUES</span><strong>{report.scientificRecord.failedAvenues.length}</strong><ul>{report.scientificRecord.failedAvenues.map((item, index) => <li key={index}>{item}</li>)}</ul></section>
        <section><span>EXACT CERTIFICATES</span><strong>{report.result.candidateCertificates.length}</strong><details><summary>Inspect certificates</summary><pre>{readable(report.result.candidateCertificates)}</pre></details></section>
        <section><span>DETERMINISTIC JOBS</span><strong>{report.codeJobs.length}</strong><details><summary>Inspect job record</summary><pre>{readable(report.codeJobs)}</pre></details></section>
      </div>
      <div className="report-agents">
        <header><span>RESEARCHER</span><span>PROPOSED PROJECT</span><span>CREDIT</span></header>
        {report.agents.map((agent) => <div key={agent.id}><b>{agent.name}</b><p>{agent.proposedPrizeProject}</p><strong>{agent.collaborationCredits}</strong></div>)}
      </div>
      <details className="report-disclosure"><summary>Release all private next-round plans ({report.privatePlansReleased.length})</summary><pre>{readable(report.privatePlansReleased)}</pre></details>
      <details className="report-disclosure"><summary>OpenAI usage ledger ({report.usage.length} entries)</summary><pre>{readable(report.usage)}</pre></details>
      <div className="report-links">
        {base && <a href={`${base}/api/experiments/${report.runId}/report`} target="_blank" rel="noreferrer">Scientific report JSON <ArrowUpRight size={12} /></a>}
        {base && <a href={`${base}${report.scientificRecord.completeEventLedger}`} target="_blank" rel="noreferrer">Complete event ledger <ArrowUpRight size={12} /></a>}
        <a href="https://github.com/RaphaelKhalid/autolabs" target="_blank" rel="noreferrer">Verifier source <ArrowUpRight size={12} /></a>
      </div>
    </section>
  );
}
export function AlienForm({ agent, index, meeting, compact = false }: {
  agent: ResearchAgent;
  index: number;
  meeting: boolean;
  compact?: boolean;
}) {
  const filter = `matter-${agent.id}-${compact ? 'small' : 'field'}`;
  const color = pigments[index % pigments.length];
  const common = { vectorEffect: 'non-scaling-stroke' as const };

  return (
    <svg
      className={`alien-form alien-form--${index + 1} ${meeting ? 'is-convening' : ''} ${compact ? 'is-compact' : ''}`}
      viewBox="0 0 200 180"
      role="img"
      aria-label={`${agent.name}, an abstract research phenomenon`}
    >
      <defs>
        <filter id={filter} x="-35%" y="-35%" width="170%" height="170%">
          <feTurbulence type="fractalNoise" baseFrequency=".008 .025" numOctaves="2" seed={index + 7} result="noise" />
          <feDisplacementMap in="SourceGraphic" in2="noise" scale={meeting ? 11 : 17} xChannelSelector="R" yChannelSelector="B" />
        </filter>
        <radialGradient id={`wash-${filter}`} cx="42%" cy="38%" r="65%">
          <stop offset="0" stopColor={color} stopOpacity=".66" />
          <stop offset=".54" stopColor={color} stopOpacity=".22" />
          <stop offset="1" stopColor={color} stopOpacity="0" />
        </radialGradient>
      </defs>

      <ellipse className="alien-form__aura" cx="100" cy="92" rx="78" ry="62" fill={`url(#wash-${filter})`} />
      {index === 0 && <>
        <g className="alien-form__drift" filter={`url(#${filter})`}>
          <path d="M18 91C43 36 80 27 105 55c26-28 59-17 77 36-25-18-42-17-52 11-11 34-49 41-65 5C54 82 38 79 18 91Z" fill={color} fillOpacity=".18" stroke={color} strokeWidth="1.35" {...common} />
          <path d="M34 91c27-23 46-22 66 4 20-28 41-28 66-4-23 5-37 17-44 39-16-9-30-9-45 0-7-22-21-34-43-39Z" fill="none" stroke={color} strokeOpacity=".7" strokeWidth=".8" {...common} />
          <path d="M57 71c30 16 58 16 87 0M65 112c23-13 46-13 70 0" fill="none" stroke={color} strokeOpacity=".42" strokeWidth=".7" {...common} />
        </g>
      </>}
      {index === 1 && <>
        <g className="alien-form__orbit">
          {[0, 45, 90, 135].map((rotation) => <ellipse key={rotation} cx="100" cy="90" rx="24" ry="68" transform={`rotate(${rotation} 100 90)`} fill="none" stroke={color} strokeOpacity=".56" strokeWidth="1" {...common} />)}
        </g>
        <path className="alien-form__pulse" d="M100 29c17 25 38 43 64 60-27 13-49 33-65 61-15-29-36-49-63-61 27-16 48-35 64-60Z" fill={color} fillOpacity=".11" stroke={color} strokeWidth="1.1" filter={`url(#${filter})`} {...common} />
      </>}
      {index === 2 && <>
        <g className="alien-form__counterturn" filter={`url(#${filter})`}>
          <path d="M35 101C46 29 131 19 159 69c26 48-34 91-80 70-39-17-31-65 1-78 35-15 74 18 54 47-17 25-58 15-56-13 2-21 31-31 45-13" fill="none" stroke={color} strokeWidth="5.4" strokeLinecap="round" strokeOpacity=".26" {...common} />
          <path d="M35 101C46 29 131 19 159 69c26 48-34 91-80 70-39-17-31-65 1-78 35-15 74 18 54 47-17 25-58 15-56-13 2-21 31-31 45-13" fill="none" stroke={color} strokeWidth=".85" strokeLinecap="round" {...common} />
        </g>
        <circle cx="100" cy="91" r="52" fill="none" stroke={color} strokeOpacity=".25" strokeDasharray="1 8" {...common} />
      </>}
      {index === 3 && <>
        <g className="alien-form__breathe" filter={`url(#${filter})`}>
          <path d="M25 90C48 53 67 35 100 27c33 8 52 26 75 63-26 37-47 55-75 63-28-8-49-26-75-63Z" fill={color} fillOpacity=".12" stroke={color} strokeWidth="1.2" {...common} />
          {[0, 1, 2, 3].map((n) => <path key={n} d={`M${42 + n * 12} 90C${59 + n * 7} ${55 - n * 4} ${95 + n * 4} ${45 + n * 6} ${157 - n * 12} 90C${139 - n * 7} ${126 + n * 3} ${104 - n * 3} ${137 - n * 5} ${42 + n * 12} 90Z`} fill="none" stroke={color} strokeOpacity={.22 + n * .11} strokeWidth=".75" {...common} />)}
        </g>
      </>}
      {index === 4 && <>
        <g className="alien-form__eclipse" filter={`url(#${filter})`}>
          <path d="M29 107c18-69 89-94 145-43-27-2-44 8-50 30-8 31-51 49-95 13Z" fill={color} fillOpacity=".17" stroke={color} strokeWidth="1.15" {...common} />
          <path d="M29 107c30 10 58 7 83-10 25-16 42-27 62-33" fill="none" stroke={color} strokeWidth=".8" strokeOpacity=".68" {...common} />
          <path d="M42 119c43 18 87 3 119-36" fill="none" stroke={color} strokeWidth="6" strokeOpacity=".08" {...common} />
        </g>
        <g className="alien-form__sparks" fill={color}>{[['44','63'],['67','39'],['142','123'],['166','100'],['94','143']].map(([cx,cy], n) => <circle key={n} cx={cx} cy={cy} r={n % 2 ? 1.2 : 2} />)}</g>
      </>}
      <circle className="alien-form__datum" cx="100" cy="90" r="2.4" fill={color} />
    </svg>
  );
}

export function AutolabsObservatory() {
  const [state, setState] = useState<ExperimentState>(initialExperiment);
  const [selected, setSelected] = useState<ResearchAgent | null>(null);
  const [tab, setTab] = useState<'stream' | 'profile' | 'tools'>('stream');
  const [now, setNow] = useState(Date.now());
  const [sound, setSound] = useState(false);
  const [replay, setReplay] = useState(false);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [replayIndex, setReplayIndex] = useState(state.events.length - 1);
  const [controlOpen, setControlOpen] = useState(false);
  const [ownerKey, setOwnerKey] = useState('');
  const [startMessage, setStartMessage] = useState('');
  const [live, setLive] = useState(Boolean(process.env.NEXT_PUBLIC_ORCHESTRATOR_URL));

  const reload = useCallback(async () => {
    try {
      const next = await fetchExperiment();
      setState(next);
      setLive(Boolean(process.env.NEXT_PUBLIC_ORCHESTRATOR_URL));
    } catch { setLive(false); }
  }, []);

  useEffect(() => {
    void reload();
    const clock = window.setInterval(() => setNow(Date.now()), 1000);
    const poll = window.setInterval(() => void reload(), 5000);
    return () => { window.clearInterval(clock); window.clearInterval(poll); };
  }, [reload]);


  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    try {
      void Promise.resolve(context.registerTool({
        name: 'read_autolabs_status', title: 'Read Autolabs status',
        description: 'Read the current public phase, round, exact best support vector, and API budget usage. This never starts or changes the experiment.',
        inputSchema: { type: 'object', properties: {}, additionalProperties: false },
        annotations: { readOnlyHint: true, untrustedContentHint: false },
        execute: () => ({ phase: state.phase, round: state.round, targetRounds: state.targetRounds, bestSupport: state.bestSupport, spentUsd: state.spentUsd, budgetUsd: state.budgetUsd }),
      }, { signal: lifecycle.signal })).catch(() => undefined);
    } catch { /* Browser does not implement WebMCP. */ }
    return () => lifecycle.abort();
  }, [state]);

  const visibleEvents = useMemo(() => state.events.filter((event) => event.visible), [state.events]);
  const replayMax = Math.max(0, visibleEvents.length - 1);
  const replayPosition = Math.min(Math.max(0, replayIndex), replayMax);
  const replayEvent = visibleEvents[replayPosition];
  const phase = replay && replayEvent ? replayEvent.phase : state.phase;
  const isMeeting = phase === 'meeting';
  const displayRound = replay && replayEvent ? replayEvent.round : state.round;
  const displaySupport = useMemo(() => {
    if (!replay) return state.bestSupport;
    let best = [0, 0, 0, 0, 0];
    for (const event of visibleEvents.slice(0, replayPosition + 1)) {
      const support = supportFrom(asRecord(event.payload).support);
      if (support && strongerSupport(support, best)) best = support;
    }
    return best;
  }, [replay, replayPosition, state.bestSupport, visibleEvents]);
  const selectedToolEvents = useMemo(() => selected ? visibleEvents.filter((event) => event.agentId === selected.id && (event.kind === 'tool' || asItems(asRecord(event.payload).proposedJobs).length > 0)) : [], [selected, visibleEvents]);
  const workerUrl = process.env.NEXT_PUBLIC_ORCHESTRATOR_URL;
  const budgetPercent = Math.min(100, (state.spentUsd / state.budgetUsd) * 100);
  const supportPercent = Math.min(100, (displaySupport.reduce((sum, n) => sum + n, 0) / 25) * 100);

  useEffect(() => {
    if (!selected) return;
    const fresh = state.agents.find((agent) => agent.id === selected.id);
    if (fresh && fresh !== selected) setSelected(fresh);
  }, [selected, state.agents]);

  useEffect(() => {
    if (!replay) setReplayIndex(replayMax);
  }, [replay, replayMax]);

  useEffect(() => {
    if (!replay || !replayPlaying || visibleEvents.length === 0) return;
    if (replayPosition >= replayMax) {
      setReplayPlaying(false);
      return;
    }
    const stepMs = Math.max(150, Math.min(1200, Math.floor(300_000 / visibleEvents.length)));
    const timer = window.setTimeout(() => setReplayIndex((value) => Math.min(replayMax, value + 1)), stepMs);
    return () => window.clearTimeout(timer);
  }, [replay, replayMax, replayPlaying, replayPosition, visibleEvents.length]);

  async function startRun(mode: 'rehearsal' | 'competition') {
    setStartMessage('Opening the laboratory…');
    try {
      const response = await fetch('/api/control/start', {
        method: 'POST', headers: { 'content-type': 'application/json', 'x-autolabs-owner-key': ownerKey },
        body: JSON.stringify({ mode, targetRounds: 50, minimumRounds: 25, phaseMinutes: 5, budgetUsd: 50 }),
      });
      const payload = await response.json() as { error?: string };
      if (!response.ok) throw new Error(payload.error ?? 'Could not start the laboratory.');
      setStartMessage('Ribbon cut. Research is in motion.');
      setOwnerKey('');
      await reload();
    } catch (error) { setStartMessage(error instanceof Error ? error.message : 'Could not start the laboratory.'); }
  }

  return (
    <main className={`observatory ${isMeeting ? 'is-meeting' : ''}`}>
      <div className="paper-noise" aria-hidden="true" />
      <header className="masthead">
        <a className="wordmark" href="/" aria-label="Return to the living lab">
          <span className="wordmark__sigil">A</span>
          <span><b>AUTOLABS</b><small>Computational mathematics, observed</small></span>
        </a>
        <nav className="masthead__nav" aria-label="Primary">
          <a href="/">Lab</a>
          <a href="#ledger">Ledger</a>
          {state.report && <a href="#report">Report</a>}
          <a href="https://github.com/RaphaelKhalid/autolabs" target="_blank" rel="noreferrer">Source <ArrowUpRight size={11} /></a>
        </nav>
        <div className="masthead__actions">
          <span className={`live-signal ${live ? '' : 'is-preview'}`}><i /> {live ? (state.phase === 'idle' ? 'Engine online · idle' : 'Live experiment') : 'Preview state'}</span>
          <button className="bare-button" onClick={() => setSound(!sound)} aria-label={sound ? 'Mute atmosphere' : 'Enable atmosphere'}>{sound ? <Volume2 size={15} /> : <VolumeX size={15} />}</button>
          <button className="control-link" onClick={() => setControlOpen(true)}>Owner control</button>
        </div>
      </header>

      <section className="score-ribbon" aria-label="Current experiment score">
        <div className="score-ribbon__title"><span>ERDŐS PROBLEM</span><strong>885</strong></div>
        <div className="score-ribbon__measure">
          <span>BEST EXACT SUPPORT</span>
          <b>{supportLabel(displaySupport)}</b>
          <div className="score-line"><i style={{ width: `${supportPercent}%` }} /></div>
          <small>target&nbsp; (5, 5, 5, 5, 5)</small>
        </div>
        <div className="score-ribbon__datum"><span>INTERVAL</span><b>{phaseTitle({ ...state, phase })}</b></div>
        <div className="score-ribbon__datum"><span>ROUND</span><b>{String(displayRound).padStart(2, '0')}<em> / {state.targetRounds}</em></b></div>
        <div className="score-ribbon__datum score-ribbon__clock"><span>NEXT TRANSITION</span><b>{formatCountdown(replay ? null : state.phaseEndsAt, now)}</b></div>
      </section>

      <section id="field" className="research-intro">
        <p className="section-index">01 / LIVE FIELD</p>
        <div>
          <h1>An exact search,<br /><em>observed in motion.</em></h1>

        </div>
        <button className="replay-link" onClick={() => { setReplayPlaying(false); if (replay) setReplay(false); else { setReplay(true); setReplayIndex(0); } }}><TimerReset size={14} /> {replay ? 'Return to the present' : 'Replay the experiment'}</button>
      </section>

      <section className="research-field" aria-label="Five live autonomous researchers">
        <svg className="field-lines" viewBox="0 0 1200 650" preserveAspectRatio="none" aria-hidden="true">
          <path d="M30 326H1170M600 22V628" />
          <ellipse cx="600" cy="325" rx="375" ry="238" />
          <ellipse cx="600" cy="325" rx="180" ry="112" />
          <path d="M225 325C370 150 826 146 975 325M225 325c145 177 601 179 750 0" />
        </svg>
        <div className="field-equation" aria-hidden="true"><span>d² + 4N</span><i>=</i><span>m²</span></div>
        <div className="meeting-nucleus" aria-hidden={!isMeeting}>
          <i /><i /><i />
          <span>SIMULTANEOUS<br />REVEAL</span>
        </div>

        {state.agents.map((agent, index) => {
          const position = isMeeting ? meetingPositions[index] : researchPositions[index];
          return (
            <button
              key={agent.id}
              className={`researcher researcher--${index + 1} ${selected?.id === agent.id ? 'is-selected' : ''}`}
              style={{ '--x': `${position.x}%`, '--y': `${position.y}%`, '--pigment': pigments[index] } as React.CSSProperties}
              onClick={() => { setSelected(agent); setTab('stream'); }}
              aria-label={`Open ${agent.name}'s research record`}
            >
              <span className="research-note"><i>0{index + 1}</i><span>{phase === 'idle' ? 'Dormant geometry; waiting for the ribbon.' : isMeeting ? 'Report sealed; entering shared review.' : agent.bubble}</span></span>
              <AlienForm agent={agent} index={index} meeting={isMeeting} />
              <span className="researcher__identity"><b>{agent.name}</b><small>{agent.epithet}</small></span>
              <span className="researcher__state"><i /> {phase === 'idle' ? 'READY' : isMeeting ? 'CONVENING' : formatCountdown(state.phaseEndsAt, now)}</span>
            </button>
          );
        })}

        <aside className="field-caption">
          <span>THE EXACT CONDITION</span>
          <p>For every row integer N and every column difference d, the quantity d² + 4N must be a perfect square.</p>
          <b>No approximation enters the score.</b>
        </aside>
      </section>

      <section className="telemetry" aria-label="Experiment telemetry">
        <div><span>MODEL</span><b>5 × GPT-5.6 Luna High</b></div>
        <div><span>CADENCE</span><b>5 min private / 5 min shared</b></div>
        <div><span>VERIFIER</span><b><ShieldCheck size={13} /> Exact bigint</b></div>
        <div className="budget-meter"><span>OPENAI USE</span><b>${state.spentUsd.toFixed(2)} <em>/ ${state.budgetUsd.toFixed(2)}</em></b><i><u style={{ width: `${budgetPercent}%` }} /></i></div>
      </section>

      <section id="ledger" className="record-section">
        <div className="record-heading">
          <p className="section-index">02 / PUBLIC RECORD</p>
          <h2>Nothing claimed<br />without a certificate.</h2>
          <p>The ledger is append-only. Reports reveal together; private next-round plans remain sealed until the experiment concludes.</p>
        </div>
        <div className="event-ledger">
          <header><span>EVENT</span><span>RESEARCH RECORD</span><span>TIME</span></header>
          {visibleEvents.length === 0 && <div className="ledger-empty"><Radio size={15} /><p><b>The instrument is ready.</b><small>The public ledger begins at the ribbon cutting.</small></p></div>}
          {visibleEvents.slice().reverse().map((event) => (
            <button key={event.seq} onClick={() => event.agentId && setSelected(state.agents.find((agent) => agent.id === event.agentId) ?? null)}>
              <span className={`event-glyph event-glyph--${event.kind}`}><EventIcon event={event} /></span>
              <span><b>{event.title}</b><small>{event.summary}</small></span>
              <time>R{event.round}<br />{new Date(event.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time>
            </button>
          ))}
        </div>
      </section>

      {state.report && <ScientificReportView report={state.report} workerUrl={workerUrl} />}

      <section className="covenant">
        <p className="section-index">{state.report ? '04' : '03'} / COVENANT</p>
        <div className="covenant__statement"><span>δ</span><p>Human mathematics,<br /><em>exact evidence.</em></p></div>
        <ol>
          <li><span>01</span>Five unrestricted expert mathematicians</li>
          <li><span>02</span>Simultaneous reports; private plans embargoed</li>
          <li><span>03</span>Immediate stop on an exact k = 5 certificate</li>
          <li><span>04</span>$50 API ceiling with a protected reserve</li>
        </ol>
        <div className="prize-note"><Trophy size={19} /><p><b>$50 victory project</b><small>$25 for every credited collaborator · supplied later by Raphael</small></p></div>
      </section>

      <footer><span>AUTOLABS / EXPERIMENT 885</span><p>Public observation. Deterministic verification. Reproducible research.</p><a href="https://github.com/RaphaelKhalid/autolabs" target="_blank" rel="noreferrer"><Github size={14} /> Inspect the source</a></footer>

      {controlOpen && (
        <div className="modal-backdrop" onMouseDown={() => setControlOpen(false)}>
          <section className="control-modal" onMouseDown={(event) => event.stopPropagation()} aria-modal="true" role="dialog" aria-labelledby="control-title">
            <button className="modal-close" onClick={() => setControlOpen(false)} aria-label="Close control room"><X size={18} /></button>
            <span className="modal-kicker"><Zap size={13} /> PRIVATE OWNER CONTROL</span>
            <h2 id="control-title">Cut the ribbon.</h2>
            <p>The dress rehearsal is real: identical agents, tools, exact verification, public ledger and budget accounting. Once opened, the durable engine continues without this browser.</p>
            <label>OWNER KEY<input type="password" value={ownerKey} onChange={(event) => setOwnerKey(event.target.value)} autoComplete="off" placeholder="Private launch key" /></label>
            <div className="modal-actions">
              <button onClick={() => void startRun('rehearsal')}>Real dress rehearsal</button>
              <button className="primary" onClick={() => void startRun('competition')}>Start competition</button>
            </div>
            {startMessage && <output aria-live="polite">{startMessage}</output>}
            <small>50 target rounds · 25 guaranteed · 5+5 minute cadence · $50 OpenAI ceiling including rehearsal.</small>
          </section>
        </div>
      )}

      {selected && (() => {
        const index = Math.max(0, state.agents.findIndex((agent) => agent.id === selected.id));
        return (
          <div className="drawer-backdrop" onMouseDown={() => setSelected(null)}>
            <aside className="research-drawer" onMouseDown={(event) => event.stopPropagation()} style={{ '--pigment': pigments[index] } as React.CSSProperties}>
              <header>
                <div className="drawer-form"><AlienForm agent={selected} index={index} meeting={isMeeting} compact /></div>
                <div><span>{selected.epithet}</span><h2>{selected.name}</h2><p>Research phenomenon 0{index + 1}</p></div>
                <button onClick={() => setSelected(null)} aria-label="Close research drawer"><X size={19} /></button>
              </header>
              <nav>{(['stream', 'profile', 'tools'] as const).map((item) => <button key={item} className={tab === item ? 'active' : ''} onClick={() => setTab(item)}>{item}</button>)}</nav>
              {tab === 'stream' && <div className="drawer-content research-stream">
                <div className="transparency-note"><BrainCircuit size={15} /><p><b>Auditable research record.</b> Hypotheses, evidence, tool inputs, outputs and conclusions are shown here—not hidden chain-of-thought.</p></div>
                {visibleEvents.filter((event) => !event.agentId || event.agentId === selected.id).slice().reverse().map((event) => <article key={event.seq}><time>ROUND {event.round} · {event.kind.toUpperCase()}</time><h3>{event.title}</h3><p>{event.summary}</p>{event.kind === 'candidate' && <code>support = {supportLabel(supportFrom(asRecord(event.payload).support) ?? selected.bestSupport)} · EXACT</code>}<EventEvidence event={event} /></article>)}
              </div>}
              {tab === 'profile' && <div className="drawer-content profile-view"><span>COGNITIVE GEOMETRY</span><h3>{selected.approach}</h3><p>Every researcher may use every branch of human mathematics. Its morphology expresses how it generates, selects and attacks ideas—not a limitation on knowledge.</p><span>PROPOSED PRIZE PROJECT</span><blockquote>{selected.project}</blockquote><span>CURRENT VERIFIED SUPPORT</span><strong>{supportLabel(selected.bestSupport)}</strong></div>}
              {tab === 'tools' && <div className="drawer-content tool-view"><span>EXACT TOOLKIT</span>{selected.tools.map((tool) => <div key={tool}><FlaskConical size={15} /><p><b>{tool}</b><small>Inputs and outputs retained in the ledger.</small></p><ShieldCheck size={14} /></div>)}<div><BookOpen size={15} /><p><b>{selected.citations} source anchors</b><small>Known searches checked before compute is scheduled.</small></p><ShieldCheck size={14} /></div><span>RECORDED TOOL EVENTS</span>{selectedToolEvents.length === 0 ? <p className="tool-empty">No deterministic job has been revealed for this researcher yet.</p> : selectedToolEvents.slice().reverse().map((event) => <article className="tool-event" key={event.seq}><b>{event.title}</b><small>{event.summary}</small><EventEvidence event={event} /></article>)}</div>}
            </aside>
          </div>
        );
      })()}

      {replay && (
        <section className="replay-console">
          <button onClick={() => setReplayPlaying(!replayPlaying)} aria-label={replayPlaying ? 'Pause replay' : 'Play replay'}>{replayPlaying ? <Pause size={16} /> : <Play size={16} />}</button>
          <button onClick={() => setReplayIndex(Math.max(0, replayIndex - 1))} aria-label="Previous event"><ChevronLeft size={16} /></button>
          <input type="range" min="0" max={replayMax} value={replayPosition} onChange={(event) => setReplayIndex(Number(event.target.value))} aria-label="Replay timeline" />
          <button onClick={() => setReplayIndex(Math.min(replayMax, replayIndex + 1))} aria-label="Next event"><ChevronRight size={16} /></button>
          <div><span>EVENT {visibleEvents.length ? replayPosition + 1 : 0} / {visibleEvents.length}</span><b>{replayEvent?.title ?? 'No events yet'}</b></div>
          <button className="replay-close" onClick={() => setReplay(false)} aria-label="Close replay"><X size={16} /></button>
        </section>
      )}
    </main>
  );
}
