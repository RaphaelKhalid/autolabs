'use client';

import { AnimatePresence, motion } from 'framer-motion';
import {
  ArrowUpRight, ChevronLeft, ChevronRight, CircleDollarSign, Clock3,
  FlaskConical, Github, Pause, Play, Radio, ShieldCheck, TimerReset, X, Zap,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { fetchExperiment } from '@/lib/api';
import {
  formatCountdown, initialExperiment, supportLabel,
  type ComputeJob, type ExperimentEvent, type ExperimentState,
} from '@/lib/experiment';
import { AlienForm } from './autolabs-observatory';

const pigments = ['#9a4f36', '#657a4e', '#60528b', '#27747a', '#a14f59'];
const stations = [
  { x: 16, y: 31, rotate: -3 }, { x: 30, y: 70, rotate: 2 },
  { x: 50, y: 26, rotate: -1 }, { x: 70, y: 70, rotate: -2 },
  { x: 85, y: 34, rotate: 3 },
];
const tessellation = [
  { x: 45, y: 46, rotate: -31 }, { x: 48, y: 55, rotate: 41 },
  { x: 50, y: 42, rotate: 2 }, { x: 53, y: 55, rotate: -42 },
  { x: 56, y: 46, rotate: 31 },
];

function phaseLabel(phase: ExperimentState['phase']) {
  if (phase === 'research') return 'PRIVATE RESEARCH';
  if (phase === 'meeting') return 'SHARED COLLOQUIUM';
  if (phase === 'ribbon') return 'RIBBON CUTTING';
  if (phase === 'eureka') return 'EUREKA';
  if (phase === 'complete') return 'COMPLETE';
  if (phase === 'budget-stop') return 'BUDGET-SAFE STOP';
  if (phase === 'error') return 'ENGINE PAUSED';
  return 'AWAITING THE RIBBON';
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function supportFrom(value: unknown): number[] | null {
  return Array.isArray(value) && value.length === 5 && value.every((item) => typeof item === 'number') ? value as number[] : null;
}

function stronger(left: number[], right: number[]) {
  const a = [...left].sort((x, y) => x - y);
  const b = [...right].sort((x, y) => x - y);
  return a.some((value, index) => value !== b[index] && a.slice(0, index).every((prior, i) => prior === b[i]) && value > b[index]);
}

function Evidence({ event }: { event: ExperimentEvent }) {
  const payload = asRecord(event.payload);
  if (!Object.keys(payload).length) return null;
  return <details><summary>Evidence</summary><pre>{JSON.stringify(payload, null, 2)}</pre></details>;
}

function JobRow({ job, onOpen }: { job: ComputeJob; onOpen(): void }) {
  const active = ['queued', 'dispatching', 'running'].includes(job.status);
  return (
    <button className="queue-row" onClick={onOpen}>
      <span className={`queue-status ${active ? 'is-active' : `is-${job.status}`} `}><i />{job.status}</span>
      <span><b>{job.jobType.replaceAll('_', ' ')}</b><small>{job.agentId} · round {job.round}</small></span>
      <ArrowUpRight size={12} />
    </button>
  );
}

export function LivingLab() {
  const [state, setState] = useState<ExperimentState>(initialExperiment);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());
  const [replay, setReplay] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [replayIndex, setReplayIndex] = useState(0);
  const [controlOpen, setControlOpen] = useState(false);
  const [ownerKey, setOwnerKey] = useState('');
  const [startMessage, setStartMessage] = useState('');
  const [queueOpen, setQueueOpen] = useState(true);
  const [live, setLive] = useState(Boolean(process.env.NEXT_PUBLIC_ORCHESTRATOR_URL));

  const reload = useCallback(async () => {
    try {
      setState(await fetchExperiment());
      setLive(Boolean(process.env.NEXT_PUBLIC_ORCHESTRATOR_URL));
    } catch { setLive(false); }
  }, []);

  useEffect(() => {
    void reload();
    const clock = window.setInterval(() => setNow(Date.now()), 1_000);
    const poll = window.setInterval(() => void reload(), 5_000);
    return () => { window.clearInterval(clock); window.clearInterval(poll); };
  }, [reload]);

  const events = useMemo(() => state.events.filter((event) => event.visible), [state.events]);
  const replayMax = Math.max(0, events.length - 1);
  const replayPosition = Math.min(replayMax, Math.max(0, replayIndex));
  const replayEvent = events[replayPosition];
  const phase = replay && replayEvent ? replayEvent.phase : state.phase;
  const round = replay && replayEvent ? replayEvent.round : state.round;
  const meeting = phase === 'meeting';
  const selected = state.agents.find((agent) => agent.id === selectedId) ?? null;
  const agentEvents = selected ? events.filter((event) => !event.agentId || event.agentId === selected.id).slice().reverse() : [];
  const visibleJobs = useMemo(() => {
    const cutoff = replay && replayEvent ? new Date(replayEvent.at).getTime() : Number.POSITIVE_INFINITY;
    return (state.jobs ?? []).filter((job) => new Date(job.createdAt).getTime() <= cutoff);
  }, [replay, replayEvent, state.jobs]);
  const support = useMemo(() => {
    if (!replay) return state.bestSupport;
    let best = [0, 0, 0, 0, 0];
    for (const event of events.slice(0, replayPosition + 1)) {
      const candidate = supportFrom(asRecord(event.payload).support);
      if (candidate && stronger(candidate, best)) best = candidate;
    }
    return best;
  }, [events, replay, replayPosition, state.bestSupport]);

  useEffect(() => {
    if (!replay || !playing || !events.length) return;
    if (replayPosition >= replayMax) { setPlaying(false); return; }
    const duration = Math.max(150, Math.min(1_200, Math.floor(300_000 / events.length)));
    const timer = window.setTimeout(() => setReplayIndex((value) => Math.min(replayMax, value + 1)), duration);
    return () => window.clearTimeout(timer);
  }, [events.length, playing, replay, replayMax, replayPosition]);

  useEffect(() => { if (!replay) setReplayIndex(replayMax); }, [replay, replayMax]);

  async function startRun(mode: 'rehearsal' | 'competition') {
    setStartMessage('Opening the laboratory…');
    try {
      const response = await fetch('/api/control/start', {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-autolabs-owner-key': ownerKey },
        body: JSON.stringify({ mode }),
      });
      const payload = await response.json() as { error?: string };
      if (!response.ok) throw new Error(payload.error ?? 'Could not open the laboratory.');
      setOwnerKey('');
      setStartMessage('Ribbon cut. Research is in motion.');
      await reload();
    } catch (error) { setStartMessage(error instanceof Error ? error.message : 'Could not open the laboratory.'); }
  }

  return (
    <main className={`living-lab phase-${phase} ${meeting ? 'is-tessellating' : ''}`}>
      <div className="world-sky" aria-hidden="true"><i /><i /><i /></div>
      <div className="world-rain" aria-hidden="true" />
      <div className="world-canopy world-canopy--left" aria-hidden="true" />
      <div className="world-canopy world-canopy--right" aria-hidden="true" />
      <div className="world-window" aria-hidden="true"><i /><i /><i /><i /></div>
      <div className="world-floor" aria-hidden="true" />

      <header className="lab-header">
        <a className="lab-mark" href="/"><i>A</i><span>AUTOLABS<small>EXPERIMENT 885</small></span></a>
        <nav><a href="/journal">Journal</a><a href="https://github.com/RaphaelKhalid/autolabs" target="_blank" rel="noreferrer">Source <ArrowUpRight size={10} /></a></nav>
        <div><span className={`lab-live ${live ? '' : 'is-offline'}`}><i />{live ? 'LIVE' : 'PREVIEW'}</span><button onClick={() => setControlOpen(true)}>Owner</button></div>
      </header>

      <section className="lab-score" aria-label="Live score">
        <div><span>ERDŐS 885</span><b>{supportLabel(support)}</b></div>
        <div className="lab-scoreline"><i style={{ width: `${Math.min(100, support.reduce((sum, n) => sum + n, 0) / 25 * 100)}%` }} /></div>
        <div><span>{phaseLabel(phase)}</span><b>R{String(round).padStart(2, '0')} · {replay ? '--:--' : formatCountdown(state.phaseEndsAt, now)}</b></div>
      </section>

      <section className="lab-world" aria-label="Autonomous mathematics laboratory">
        <svg className="world-architecture" viewBox="0 0 1400 800" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <linearGradient id="wood" x1="0" x2="1"><stop stopColor="#4d2e23" stopOpacity=".44"/><stop offset="1" stopColor="#8c5736" stopOpacity=".17"/></linearGradient>
          </defs>
          <path className="balcony" d="M0 501H1400V800H0Z" fill="url(#wood)" />
          <path className="rail" d="M0 504H1400M0 563H1400M90 504v108m168-108v108m168-108v108m168-108v108m168-108v108m168-108v108m168-108v108m168-108v108" />
          {stations.map((station, index) => <g key={index} className={`station station-${index + 1}`} transform={`translate(${station.x * 14} ${station.y * 8})`}>
            <path d="M-118 84h236l-28 38H-92Z" />
            <path d="M-86 122l-16 116m188-116 16 116" />
            <rect x="-47" y="26" width="94" height="58" rx="2" />
            <path d="M-35 40h70M-35 50h48M-35 60h57" />
            <circle cx="0" cy="106" r="3" />
          </g>)}
          <motion.path className="convergence-line" d="M220 250C470 80 520 365 700 400S930 690 1190 280" initial={false} animate={{ pathLength: meeting ? 1 : .12, opacity: meeting ? .7 : .14 }} transition={{ duration: 1.8 }} />
          <motion.ellipse className="colloquium-table" cx="700" cy="430" rx="170" ry="74" initial={false} animate={{ opacity: meeting ? .7 : .08, scale: meeting ? 1 : .65 }} transition={{ duration: 1.4 }} />
        </svg>

        <div className="world-lamp world-lamp--one" aria-hidden="true"><i /></div>
        <div className="world-lamp world-lamp--two" aria-hidden="true"><i /></div>
        <AnimatePresence>
          {state.agents.map((agent, index) => {
            const target = meeting ? tessellation[index] : stations[index];
            const liveBubble = phase === 'idle' ? 'Waiting for the ribbon.' : phase === 'research' && agent.bubble.startsWith('Dormant') ? 'Working privately—report sealed until the simultaneous reveal.' : agent.bubble;
            return (
              <motion.button
                key={agent.id}
                className={`living-researcher living-researcher--${index + 1} ${meeting ? 'is-meeting' : ''}`}
                style={{ '--pigment': pigments[index] } as React.CSSProperties}
                initial={false}
                animate={{ left: `${target.x}%`, top: `${target.y}%`, x: '-50%', y: '-50%', rotate: target.rotate, scale: meeting ? .82 : 1 }}
                transition={{ type: 'spring', stiffness: 42, damping: 17, mass: 1.2, delay: index * .055 }}
                whileHover={{ scale: meeting ? .9 : 1.06, zIndex: 12 }}
                onClick={() => setSelectedId(agent.id)}
                aria-label={`Open ${agent.name}'s complete research record`}
              >
                <motion.span className="living-bubble" layoutId={`bubble-${agent.id}`}>
                  <i>0{index + 1}</i><span>{liveBubble}</span>
                </motion.span>
                <AlienForm agent={agent} index={index} meeting={meeting} />
                <span className="living-name"><b>{agent.name}</b><small>{meeting ? 'TESSELLATING' : agent.epithet}</small></span>
              </motion.button>
            );
          })}
        </AnimatePresence>
        <div className="world-weather"><span>AUTUMN RAIN / RESEARCH WEATHER</span><i /></div>
      </section>

      <aside className={`compute-queue ${queueOpen ? 'is-open' : ''}`}>
        <button className="queue-toggle" onClick={() => setQueueOpen(!queueOpen)}><FlaskConical size={14}/><span>COMPUTE QUEUE</span><b>{visibleJobs.length}</b></button>
        {queueOpen && <div className="queue-body">
          <header><span>ASYNCHRONOUS EXACT JOBS</span><small>Jobs may outlive the round that proposed them.</small></header>
          {visibleJobs.length ? visibleJobs.slice(0, 7).map((job) => <JobRow key={job.id} job={job} onOpen={() => setSelectedId(job.agentId)} />) : <div className="queue-empty"><Radio size={14}/><p>{phase === 'research' ? 'Five research calls are running.' : 'No exact jobs proposed yet.'}<small>{phase === 'research' ? 'Calculator jobs appear after the sealed reports reveal.' : 'New jobs appear here as agents dispatch them.'}</small></p></div>}
          <footer><span><CircleDollarSign size={11}/> OPENAI ${state.spentUsd.toFixed(2)} / $50</span><span>EXA ${(state.exaSpentUsd ?? 0).toFixed(2)} / $40</span></footer>
        </div>}
      </aside>

      <div className="lab-actions">
        <button onClick={() => { if (replay) setReplay(false); else { setReplay(true); setReplayIndex(0); } setPlaying(false); }}><TimerReset size={13}/>{replay ? 'Present' : 'Replay'}</button>
        <a href="/journal"><ShieldCheck size={13}/>Full record</a>
      </div>

      {selected && <div className="lab-drawer-backdrop" onMouseDown={() => setSelectedId(null)}>
        <motion.aside className="lab-drawer" initial={{ x: 80, opacity: 0 }} animate={{ x: 0, opacity: 1 }} onMouseDown={(event) => event.stopPropagation()} style={{ '--pigment': selected.accent } as React.CSSProperties}>
          <header><div><AlienForm agent={selected} index={state.agents.indexOf(selected)} meeting={meeting} compact /></div><span>{selected.epithet}</span><h2>{selected.name}</h2><button onClick={() => setSelectedId(null)} aria-label="Close record"><X size={18}/></button></header>
          <section className="lab-drawer-profile"><p>{selected.approach}</p><b>PROPOSED PROJECT</b><blockquote>{selected.project}</blockquote></section>
          <section className="lab-chat-log">
            <h3>COMPLETE PUBLIC CHAT</h3>
            {agentEvents.length ? agentEvents.map((event) => <article key={event.seq}><time>ROUND {event.round} · {event.kind}</time><b>{event.title}</b><p>{event.summary}</p><Evidence event={event}/></article>) : <p className="drawer-empty">Its first public report will appear after the reveal.</p>}
          </section>
        </motion.aside>
      </div>}

      {replay && <section className="lab-replay">
        <button onClick={() => setPlaying(!playing)}>{playing ? <Pause size={15}/> : <Play size={15}/>}</button>
        <button onClick={() => setReplayIndex(Math.max(0, replayPosition - 1))}><ChevronLeft size={14}/></button>
        <input type="range" min="0" max={replayMax} value={replayPosition} onChange={(event) => setReplayIndex(Number(event.target.value))}/>
        <button onClick={() => setReplayIndex(Math.min(replayMax, replayPosition + 1))}><ChevronRight size={14}/></button>
        <span>{events.length ? `${replayPosition + 1} / ${events.length}` : '0 / 0'}<b>{replayEvent?.title ?? 'No events yet'}</b></span>
        <button onClick={() => setReplay(false)}><X size={15}/></button>
      </section>}

      {controlOpen && <div className="lab-control-backdrop" onMouseDown={() => setControlOpen(false)}>
        <motion.section className="lab-control" initial={{ y: 25, opacity: 0 }} animate={{ y: 0, opacity: 1 }} onMouseDown={(event) => event.stopPropagation()}>
          <button className="lab-control-close" onClick={() => setControlOpen(false)}><X size={18}/></button>
          <span><Zap size={12}/>OWNER CONTROL</span><h2>Cut the ribbon.</h2><p>The real rehearsal uses the same agents, retrieval, calculators, public record and stopping rules.</p>
          <label>OWNER KEY<input type="password" value={ownerKey} onChange={(event) => setOwnerKey(event.target.value)} autoComplete="off"/></label>
          <div><button onClick={() => void startRun('rehearsal')}>Real rehearsal</button><button onClick={() => void startRun('competition')}>Competition</button></div>
          {startMessage && <output>{startMessage}</output>}
          <small><Clock3 size={11}/>50 target · 25 guaranteed · 5+5 minute cadence</small>
        </motion.section>
      </div>}
    </main>
  );
}
