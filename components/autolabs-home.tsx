 'use client';

import Image from 'next/image';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { AlienForm } from '@/components/autolabs-observatory';
import { demoAgents, type ResearchAgent } from '@/lib/experiment';

type Position = { x: number; y: number; rotation: number };
const startPositions: Position[] = [
  { x: 12, y: 42, rotation: -5 }, { x: 31, y: 70, rotation: 4 }, { x: 50, y: 37, rotation: -2 }, { x: 69, y: 70, rotation: -4 }, { x: 88, y: 43, rotation: 5 },
];
const colors = ['#f7a26f', '#c7f29e', '#cab8ff', '#7deaf2', '#ff9caf'];

export function AutolabsHome() {
  const arenaRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ id: string; dx: number; dy: number } | null>(null);
  const [positions, setPositions] = useState<Record<string, Position>>(() => Object.fromEntries(demoAgents.map((agent, i) => [agent.id, startPositions[i]])));
  const [selected, setSelected] = useState<ResearchAgent>(demoAgents[2]);
  const [action, setAction] = useState<Record<string, string>>({});

  useEffect(() => {
    const move = (event: PointerEvent) => {
      const drag = dragRef.current;
      const arena = arenaRef.current;
      if (!drag || !arena) return;
      const rect = arena.getBoundingClientRect();
      setPositions((current) => ({ ...current, [drag.id]: { ...current[drag.id], x: Math.max(7, Math.min(93, ((event.clientX - rect.left) / rect.width) * 100 - drag.dx)), y: Math.max(16, Math.min(86, ((event.clientY - rect.top) / rect.height) * 100 - drag.dy)) } }));
    };
    const up = () => { dragRef.current = null; };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    return () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
  }, []);

  function select(agent: ResearchAgent) { setSelected(agent); }
  function trigger(agent: ResearchAgent, kind: string) {
    setSelected(agent);
    setAction((current) => ({ ...current, [agent.id]: kind }));
    window.setTimeout(() => setAction((current) => ({ ...current, [agent.id]: '' })), kind === 'is-tossed' ? 850 : 520);
    if (kind === 'is-tossed') setPositions((current) => ({ ...current, [agent.id]: { x: 15 + Math.random() * 70, y: 25 + Math.random() * 55, rotation: -18 + Math.random() * 36 } }));
  }
  function reset() { setPositions(Object.fromEntries(demoAgents.map((agent, i) => [agent.id, startPositions[i]]))); setAction({}); }

  return <main className="home-playground">
    <Image className="home-playground-photo" src="/photography/sunset-flight.jpg" alt="A hazy sunset over the Pacific Ocean" fill priority sizes="100vw" />
    <div className="home-playground-wash" aria-hidden="true" />
    <nav className="playground-nav" aria-label="Primary">
      <Link className="playground-brand" href="/" aria-label="AutoLabs home"><span>A</span><strong>AUTOLABS</strong></Link>
      <div><Link href="/experiments">Experiments</Link><Link href="/research">Research</Link><Link href="/journal">Journal</Link><Link href="/studio">Studio</Link><a href="https://github.com/RaphaelKhalid/autolabs" target="_blank" rel="noreferrer">Source ↗</a></div>
    </nav>
    <section className="playground-intro"><p>THE AUTOLABS FIELD</p><h1>Meet the researchers.</h1><span>Drag a form. Double-click to punch it. Select one to poke or throw it.</span></section>
    <section className="agent-arena" ref={arenaRef} aria-label="Interactive five-agent laboratory">
      <div className="arena-grid" aria-hidden="true" />
      <div className="arena-caption"><span>LIVE FIELD / 05 AGENTS</span><small>Personal playground · no experiment is started</small></div>
      {demoAgents.map((agent, index) => { const position = positions[agent.id]; return <div key={agent.id} className={'play-agent ' + (selected.id === agent.id ? 'is-selected ' : '') + (action[agent.id] || '')} style={{ left: position.x + '%', top: position.y + '%', '--agent-color': colors[index], '--agent-rotation': position.rotation + 'deg' } as React.CSSProperties} role="button" tabIndex={0} aria-label={'Select ' + agent.name} onPointerDown={(event) => { const rect = event.currentTarget.getBoundingClientRect(); const arena = arenaRef.current?.getBoundingClientRect(); if (arena) dragRef.current = { id: agent.id, dx: ((event.clientX - rect.left - rect.width / 2) / arena.width) * 100, dy: ((event.clientY - rect.top - rect.height / 2) / arena.height) * 100 }; select(agent); }} onClick={() => select(agent)} onDoubleClick={() => trigger(agent, 'is-punched')} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(agent); } }}>
        <div className="play-agent-note"><b>0{index + 1}</b><span>{agent.name}</span></div><AlienForm agent={{ id: agent.id, name: agent.name, color: colors[index] }} index={index} meeting={false} /><strong>{agent.name}</strong><small>{agent.epithet}</small>
      </div>; })}
    </section>
    <aside className="agent-console" aria-live="polite"><div className="agent-console-copy"><span>SELECTED RESEARCHER</span><h2>{selected.name}</h2><p>{action[selected.id] === 'is-punched' ? 'Punched. Recalibrating its hypothesis.' : action[selected.id] === 'is-tossed' ? 'Thrown across the field. Recovering.' : selected.bubble}</p></div><div className="agent-console-actions"><button onClick={() => trigger(selected, 'is-punched')}>Poke</button><button onClick={() => trigger(selected, 'is-tossed')}>Throw</button><button onClick={reset}>Reset field</button><Link href="/experiments">Enter the records ↗</Link></div></aside>
    <footer className="playground-footer"><span>AUTOLABS / OPEN RECORD</span><span>Five agents · one field · endless questions</span><Link href="/research">Open research ↗</Link></footer>
  </main>;
}
