'use client';

import Image from 'next/image';
import {
  Activity, BookOpen, BrainCircuit, ChevronLeft, ChevronRight, CircleDollarSign,
  CloudRain, FlaskConical, Github, Maximize2, Pause, Play, Radio, Search,
  ShieldCheck, Sparkles, TimerReset, Trophy, Volume2, VolumeX, X, Zap,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { fetchExperiment } from '@/lib/api';
import {
  formatCountdown, initialExperiment, supportLabel,
  type ExperimentEvent, type ExperimentState, type ResearchAgent,
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

const spritePositions: Record<string, string> = {
  mira: '8% 22%', pip: '29% 22%', orum: '50% 22%', solvi: '71% 22%', tess: '92% 22%',
};

function phaseTitle(state: ExperimentState) {
  if (state.phase === 'research') return 'Private research loop';
  if (state.phase === 'meeting') return 'Round-table reveal';
  if (state.phase === 'ribbon') return 'Ribbon cutting';
  if (state.phase === 'eureka') return 'Eureka — exact witness found';
  if (state.phase === 'complete') return 'Experiment complete';
  return 'Awaiting launch';
}

function EventIcon({ event }: { event: ExperimentEvent }) {
  if (event.kind === 'candidate') return <Sparkles size={14} />;
  if (event.kind === 'meeting') return <BrainCircuit size={14} />;
  if (event.kind === 'tool') return <FlaskConical size={14} />;
  if (event.kind === 'budget') return <CircleDollarSign size={14} />;
  return <Activity size={14} />;
}

function AlienSprite({ agent, meeting }: { agent: ResearchAgent; meeting: boolean }) {
  return (
    <div className={`alien ${meeting ? 'alien--meeting' : ''}`} aria-hidden="true">
      <div
        className="alien__crop"
        style={{ backgroundPosition: spritePositions[agent.id], borderColor: agent.accent }}
      />
      <span className="alien__shadow" />
    </div>
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
  const [live, setLive] = useState(!process.env.NEXT_PUBLIC_ORCHESTRATOR_URL ? false : true);

  const reload = useCallback(async () => {
    try {
      const next = await fetchExperiment();
      setState(next);
      setLive(Boolean(process.env.NEXT_PUBLIC_ORCHESTRATOR_URL));
    } catch {
      setLive(false);
    }
  }, []);

  useEffect(() => {
    void reload();
    const clock = window.setInterval(() => setNow(Date.now()), 1000);
    const poll = window.setInterval(() => void reload(), 5000);
    return () => { window.clearInterval(clock); window.clearInterval(poll); };
  }, [reload]);

  useEffect(() => {
    if (!replayPlaying || !replay) return;
    const timer = window.setInterval(() => {
      setReplayIndex((value) => value >= state.events.length - 1 ? 0 : value + 1);
    }, 1200);
    return () => window.clearInterval(timer);
  }, [replay, replayPlaying, state.events.length]);

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
  const replayEvent = visibleEvents[Math.min(replayIndex, visibleEvents.length - 1)];
  const phase = replay && replayEvent ? replayEvent.phase : state.phase;
  const isMeeting = phase === 'meeting';
  const budgetPercent = Math.min(100, (state.spentUsd / state.budgetUsd) * 100);

  async function startRun(mode: 'rehearsal' | 'competition') {
    setStartMessage('Opening the lab…');
    try {
      const response = await fetch('/api/control/start', {
        method: 'POST', headers: { 'content-type': 'application/json', 'x-autolabs-owner-key': ownerKey },
        body: JSON.stringify({ mode, targetRounds: 50, minimumRounds: 25, phaseMinutes: 5, budgetUsd: 50 }),
      });
      const payload = await response.json() as { error?: string };
      if (!response.ok) throw new Error(payload.error ?? 'Could not start the lab.');
      setStartMessage('Ribbon cut. The researchers are moving.');
      setOwnerKey('');
      await reload();
    } catch (error) {
      setStartMessage(error instanceof Error ? error.message : 'Could not start the lab.');
    }
  }

  return (
    <main className={`observatory weather-${state.bestSupport[0] ?? 0}`}>
      <Image className="scene-bg" src="/art/autolabs-cabin-background.png" alt="Rainy autumn research cabin glowing above a wooded balcony" fill priority sizes="100vw" />
      <div className="scene-wash" />
      <div className="rain" aria-hidden="true" />
      <div className="vignette" aria-hidden="true" />

      <header className="topbar">
        <a className="brand" href="#lab" aria-label="Autolabs home">
          <span className="brand__mark"><FlaskConical size={16} /></span>
          <span><b>AUTOLABS</b><small>Luna High · Experiment 885</small></span>
        </a>

        <section className="scoreboard" aria-label="Best verified result">
          <span className="scoreboard__eyebrow"><ShieldCheck size={13} /> BEST EXACT SUPPORT</span>
          <strong>{supportLabel(state.bestSupport)}</strong>
          <span className="scoreboard__note">Goal (5,5,5,5,5)</span>
          <div className="score-track"><i style={{ width: `${(state.bestSupport.reduce((a, b) => a + b, 0) / 25) * 100}%` }} /></div>
        </section>

        <div className="top-actions">
          <span className={`live-pill ${live ? '' : 'live-pill--demo'}`}><Radio size={12} /> {live ? 'LIVE' : 'PREVIEW'}</span>
          <button className="icon-button" onClick={() => setSound(!sound)} aria-label={sound ? 'Mute atmosphere' : 'Enable atmosphere'}>{sound ? <Volume2 size={16} /> : <VolumeX size={16} />}</button>
          <a className="icon-button" href="https://github.com/RaphaelKhalid/autolabs" target="_blank" rel="noreferrer" aria-label="Open public GitHub repository"><Github size={17} /></a>
          <button className="launch-button" onClick={() => setControlOpen(true)}><Sparkles size={14} /> Control room</button>
        </div>
      </header>

      <section className="status-strip" aria-label="Experiment status">
        <div><span>PHASE</span><b>{phaseTitle({ ...state, phase })}</b></div>
        <div><span>ROUND</span><b>{String(state.round).padStart(2, '0')} <em>/ {state.targetRounds}</em></b></div>
        <div className="countdown"><span>NEXT REVEAL</span><b>{formatCountdown(state.phaseEndsAt, now)}</b></div>
        <div className="budget-readout"><span>OPENAI BUDGET</span><b>${state.spentUsd.toFixed(2)} <em>/ ${state.budgetUsd.toFixed(2)}</em></b><i><u style={{ width: `${budgetPercent}%` }} /></i></div>
        <div><span>VERIFICATION</span><b className="verified"><ShieldCheck size={14} /> exact only</b></div>
      </section>

      <section id="lab" className={`lab ${isMeeting ? 'lab--meeting' : ''}`}>
        <div className="lab-heading">
          <div><p>UNKNOWN GALAXY · CABIN 885</p><h1>Five minds, one stubborn rectangle.</h1></div>
          <button className="replay-button" onClick={() => { setReplay(!replay); setReplayPlaying(false); }}><TimerReset size={15} /> {replay ? 'Return live' : 'Open replay'}</button>
        </div>

        <div className="meeting-table" aria-hidden={!isMeeting}>
          <span>SIMULTANEOUS REVEAL</span><b>Round {state.round}</b><i />
        </div>

        <div className="cubicle-grid">
          {state.agents.map((agent, index) => (
            <button
              key={agent.id}
              className={`agent-station agent-station--${index + 1} ${selected?.id === agent.id ? 'is-selected' : ''}`}
              style={{ '--accent': agent.accent } as React.CSSProperties}
              onClick={() => { setSelected(agent); setTab('stream'); }}
            >
              <span className="thought-bubble">{isMeeting ? 'Report sealed. Ready to reveal.' : agent.bubble}<i /></span>
              <span className="desk"><span className="monitor"><i /><i /><i /></span><span className="mug" /></span>
              <AlienSprite agent={agent} meeting={isMeeting} />
              <span className="agent-tag"><b>{agent.name}</b><small>{agent.epithet}</small></span>
              <span className="agent-timer"><span className="pulse" /> {isMeeting ? 'AT TABLE' : formatCountdown(state.phaseEndsAt, now)}</span>
            </button>
          ))}
        </div>

        <aside className="field-note">
          <span>THE OBJECTIVE</span>
          <p>Find five integers whose factor-pair difference sets share five values.</p>
          <code>d² + 4N = m²</code>
          <small>Every green cell must be an exact square. No “almost”.</small>
        </aside>
      </section>

      <section className="lower-deck">
        <article className="ledger-card">
          <header><div><span><Activity size={14} /> PUBLIC EVENT LEDGER</span><small>Append-only · private plans unlock after completion</small></div><button aria-label="Expand event ledger"><Maximize2 size={15} /></button></header>
          <div className="event-list">
            {visibleEvents.slice().reverse().map((event) => (
              <button key={event.seq} onClick={() => event.agentId && setSelected(state.agents.find((a) => a.id === event.agentId) ?? null)}>
                <i className={`event-icon event-icon--${event.kind}`}><EventIcon event={event} /></i>
                <span><b>{event.title}</b><small>{event.summary}</small></span>
                <time>R{event.round} · {new Date(event.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time>
              </button>
            ))}
          </div>
        </article>

        <aside className="protocol-card">
          <header><ShieldCheck size={15} /><span><b>Research covenant</b><small>Committed before launch</small></span></header>
          <ul>
            <li><i /> Five unrestricted Luna High mathematicians</li>
            <li><i /> 5 min private research · 5 min meeting</li>
            <li><i /> Simultaneous reports; private plans embargoed</li>
            <li><i /> Stop instantly on an exact k=5 certificate</li>
            <li><i /> $50 API hard ceiling; reserve enforced</li>
          </ul>
          <div className="reward"><Trophy size={18} /><span><b>$50 victory project</b><small>$25 to every credited collaborator · supplied later by Raphael</small></span></div>
        </aside>
      </section>

      {controlOpen && (
        <div className="modal-backdrop" onMouseDown={() => setControlOpen(false)}>
          <section className="control-modal" onMouseDown={(event) => event.stopPropagation()} aria-modal="true" role="dialog" aria-labelledby="control-title">
            <button className="modal-close" onClick={() => setControlOpen(false)} aria-label="Close control room"><X size={18} /></button>
            <span className="modal-kicker"><Zap size={13} /> OWNER CONTROL · PUBLIC OBSERVERS ARE READ-ONLY</span>
            <h2 id="control-title">Cut the ribbon.</h2>
            <p>The rehearsal is real: the same five Luna High calls, exact tools, ledger and budget meter as competition mode. Once started, the durable engine continues online without this browser.</p>
            <label>OWNER KEY<input type="password" value={ownerKey} onChange={(event) => setOwnerKey(event.target.value)} autoComplete="off" placeholder="Enter the private launch key" /></label>
            <div className="modal-actions">
              <button onClick={() => void startRun('rehearsal')}><CloudRain size={14} /> Real dress rehearsal</button>
              <button className="primary" onClick={() => void startRun('competition')}><Sparkles size={14} /> Start competition</button>
            </div>
            {startMessage && <output aria-live="polite">{startMessage}</output>}
            <small>Competition target: 50 rounds · minimum: 25 · 5+5 minute cadence · $50 OpenAI ceiling including rehearsal.</small>
          </section>
        </div>
      )}

      {selected && (
        <div className="drawer-backdrop" onMouseDown={() => setSelected(null)}>
          <aside className="research-drawer" onMouseDown={(event) => event.stopPropagation()} style={{ '--accent': selected.accent } as React.CSSProperties}>
            <header>
              <div className="drawer-avatar"><div style={{ backgroundPosition: spritePositions[selected.id] }} /></div>
              <div><span>{selected.epithet}</span><h2>{selected.name}</h2><p>{selected.outfit}</p></div>
              <button onClick={() => setSelected(null)} aria-label="Close research drawer"><X size={19} /></button>
            </header>
            <nav>
              {(['stream', 'profile', 'tools'] as const).map((item) => <button key={item} className={tab === item ? 'active' : ''} onClick={() => setTab(item)}>{item}</button>)}
            </nav>
            {tab === 'stream' && <div className="drawer-content research-stream">
              <div className="transparency-note"><BrainCircuit size={15} /><p><b>Research record, not hidden chain-of-thought.</b> You see hypotheses, evidence, tool inputs, outputs and conclusions needed to audit the work.</p></div>
              {visibleEvents.filter((event) => !event.agentId || event.agentId === selected.id).slice().reverse().map((event) => <article key={event.seq}><time>ROUND {event.round} · {event.kind.toUpperCase()}</time><h3>{event.title}</h3><p>{event.summary}</p>{event.kind === 'candidate' && <code>support = {supportLabel(selected.bestSupport)} · EXACT</code>}</article>)}
            </div>}
            {tab === 'profile' && <div className="drawer-content profile-view"><span>COGNITIVE STYLE</span><h3>{selected.approach}</h3><p>All five researchers may use every branch of human mathematics. Their persona changes how they generate and challenge ideas—not what they are allowed to know.</p><span>PROPOSED PRIZE PROJECT</span><blockquote>{selected.project}</blockquote><span>CURRENT VERIFIED SUPPORT</span><strong>{supportLabel(selected.bestSupport)}</strong></div>}
            {tab === 'tools' && <div className="drawer-content tool-view"><span>EXACT TOOLKIT</span>{selected.tools.map((tool) => <div key={tool}><FlaskConical size={15} /><p><b>{tool}</b><small>Inputs and outputs are retained in the event ledger.</small></p><ShieldCheck size={14} /></div>)}<div><BookOpen size={15} /><p><b>{selected.citations} source anchors</b><small>Known searches are checked before compute is scheduled.</small></p><ShieldCheck size={14} /></div></div>}
          </aside>
        </div>
      )}

      {replay && (
        <section className="replay-console">
          <button onClick={() => setReplayPlaying(!replayPlaying)} aria-label={replayPlaying ? 'Pause replay' : 'Play replay'}>{replayPlaying ? <Pause size={17} /> : <Play size={17} />}</button>
          <button onClick={() => setReplayIndex(Math.max(0, replayIndex - 1))} aria-label="Previous event"><ChevronLeft size={17} /></button>
          <input type="range" min="0" max={Math.max(0, visibleEvents.length - 1)} value={Math.min(replayIndex, visibleEvents.length - 1)} onChange={(event) => setReplayIndex(Number(event.target.value))} aria-label="Replay timeline" />
          <button onClick={() => setReplayIndex(Math.min(visibleEvents.length - 1, replayIndex + 1))} aria-label="Next event"><ChevronRight size={17} /></button>
          <div><span>REPLAY EVENT {replayIndex + 1}/{visibleEvents.length}</span><b>{replayEvent?.title ?? 'No events yet'}</b></div>
          <button className="replay-close" onClick={() => setReplay(false)}><X size={16} /></button>
        </section>
      )}


    </main>
  );
}
