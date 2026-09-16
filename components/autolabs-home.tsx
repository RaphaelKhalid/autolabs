'use client';

import Image from 'next/image';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { AlienForm } from '@/components/autolabs-observatory';
import { demoAgents, type ResearchAgent } from '@/lib/experiment';

type Position = { x: number; y: number; rotation: number };
type DialogueStep = { agentId: string; text: string; position: Position };

const startPositions: Position[] = [
  { x: 12, y: 42, rotation: -5 }, { x: 31, y: 70, rotation: 4 }, { x: 50, y: 37, rotation: -2 }, { x: 69, y: 70, rotation: -4 }, { x: 88, y: 43, rotation: 5 },
];
const colors = ['#f7a26f', '#c7f29e', '#cab8ff', '#7deaf2', '#ff9caf'];

// A short, clearly labelled rehearsal assembled from the recorded Erdős 885 run.
const dialogue: DialogueStep[] = [
  { agentId: 'pip', text: 'I found a smooth seed. 18,442 branches die at the square test.', position: { x: 23, y: 32, rotation: -3 } },
  { agentId: 'solvi', text: 'A coordinate change exposes a symmetry the scan never used.', position: { x: 63, y: 29, rotation: 3 } },
  { agentId: 'orum', text: 'Then I will try to kill that branch before we believe it.', position: { x: 48, y: 61, rotation: -4 } },
  { agentId: 'mira', text: 'The non-fixed family survives the first exact constraints.', position: { x: 18, y: 69, rotation: 4 } },
  { agentId: 'tess', text: 'The cutoff still has a thin frontier. I am moving the boundary.', position: { x: 79, y: 63, rotation: -2 } },
  { agentId: 'orum', text: 'Candidate retained: (4,4,4,4,4). Size alone is not a proof.', position: { x: 42, y: 35, rotation: 2 } },
  { agentId: 'mira', text: 'Let us test the family, not just the isolated hit.', position: { x: 28, y: 47, rotation: -5 } },
  { agentId: 'pip', text: 'Certificate attached. No floating-point arithmetic.', position: { x: 70, y: 72, rotation: 4 } },
];

export function AutolabsHome() {
  const arenaRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ id: string; dx: number; dy: number } | null>(null);
  const [positions, setPositions] = useState<Record<string, Position>>(() => Object.fromEntries(demoAgents.map((agent, i) => [agent.id, startPositions[i]])));
  const [selected, setSelected] = useState<ResearchAgent>(demoAgents[2]);
  const [action, setAction] = useState<Record<string, string>>({});
  const [dialogueIndex, setDialogueIndex] = useState(0);
  const currentLine = dialogue[dialogueIndex];
  const speaker = demoAgents.find((agent) => agent.id === currentLine.agentId) ?? demoAgents[0];

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

  useEffect(() => {
    const timer = window.setInterval(() => setDialogueIndex((index) => (index + 1) % dialogue.length), 4200);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    setSelected(speaker);
    setPositions((current) => ({ ...current, [speaker.id]: currentLine.position }));
  }, [dialogueIndex, speaker, currentLine]);

  function select(agent: ResearchAgent) { setSelected(agent); }
  function trigger(agent: ResearchAgent, kind: string) {
    setSelected(agent);
    setAction((current) => ({ ...current, [agent.id]: kind }));
    window.setTimeout(() => setAction((current) => ({ ...current, [agent.id]: '' })), kind === 'is-tossed' ? 850 : 520);
    if (kind === 'is-tossed') setPositions((current) => ({ ...current, [agent.id]: { x: 15 + Math.random() * 70, y: 25 + Math.random() * 55, rotation: -18 + Math.random() * 36 } }));
  }
  function reset() {
    setPositions(Object.fromEntries(demoAgents.map((agent, i) => [agent.id, startPositions[i]])));
    setAction({});
    setDialogueIndex(0);
  }

  return <main className="home-playground">
    <Image className="home-playground-photo" src="/photography/autolabs-field.png" alt="A hazy sunset over the Pacific Ocean" fill priority sizes="100vw" />
    <div className="home-playground-wash" aria-hidden="true" />
    <section className="playground-intro"><p>THE AUTOLABS FIELD</p><h1>Meet the researchers.</h1></section>
    <section className="agent-arena" ref={arenaRef} aria-label="Interactive five-agent laboratory">
      <div className="arena-grid" aria-hidden="true" />
      <div className="arena-caption"><span>LIVE FIELD / 05 AGENTS</span><small>Rehearsed dialogue · Erdős 885 run</small></div>
      {demoAgents.map((agent, index) => {
        const position = positions[agent.id];
        const isSpeaking = speaker.id === agent.id;
        return <div key={agent.id} className={'play-agent ' + (selected.id === agent.id ? 'is-selected ' : '') + (isSpeaking ? 'is-speaking ' : '') + (action[agent.id] || '')} style={{ left: position.x + '%', top: position.y + '%', '--agent-color': colors[index], '--agent-rotation': position.rotation + 'deg' } as React.CSSProperties} role="button" tabIndex={0} aria-label={'Select and drag ' + agent.name} onPointerDown={(event) => { event.currentTarget.setPointerCapture?.(event.pointerId); const rect = event.currentTarget.getBoundingClientRect(); const arena = arenaRef.current?.getBoundingClientRect(); if (arena) dragRef.current = { id: agent.id, dx: ((event.clientX - rect.left - rect.width / 2) / arena.width) * 100, dy: ((event.clientY - rect.top - rect.height / 2) / arena.height) * 100 }; select(agent); }} onPointerMove={(event) => { const drag = dragRef.current; const arena = arenaRef.current?.getBoundingClientRect(); if (!drag || drag.id !== agent.id || !arena) return; setPositions((current) => ({ ...current, [agent.id]: { ...current[agent.id], x: Math.max(7, Math.min(93, ((event.clientX - arena.left) / arena.width) * 100 - drag.dx)), y: Math.max(16, Math.min(86, ((event.clientY - arena.top) / arena.height) * 100 - drag.dy)) } })); }} onPointerUp={() => { dragRef.current = null; }} onClick={() => select(agent)} onDoubleClick={() => trigger(agent, 'is-punched')} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(agent); } }}>
          <div className="play-agent-aura" aria-hidden="true" />
          <div className="play-agent-note"><b>0{index + 1}</b><span>{agent.name}</span></div>
          <AlienForm agent={{ id: agent.id, name: agent.name, color: colors[index] }} index={index} meeting={false} />
          {isSpeaking && <div className={'play-agent-dialogue ' + (position.y < 43 ? 'dialogue-below' : '')} aria-live="polite"><b>{agent.name}</b><span>{currentLine.text}</span></div>}
          <strong>{agent.name}</strong><small>{agent.epithet}</small>
        </div>;
      })}
    </section>
    <aside className="agent-console" aria-live="polite">
      <div className="agent-console-copy"><span>SELECTED RESEARCHER</span><h2>{selected.name}</h2><p>{action[selected.id] === 'is-punched' ? 'Punched. Recalibrating its hypothesis.' : action[selected.id] === 'is-tossed' ? 'Thrown across the field. Recovering.' : selected.bubble}</p></div>
      <div className="field-dialogue"><span>FIELD REHEARSAL / ERDŐS 885</span><p><b>{speaker.name}</b> “{currentLine.text}”</p></div>
      <div className="agent-console-actions"><button onClick={() => trigger(selected, 'is-punched')}>Poke</button><button onClick={() => trigger(selected, 'is-tossed')}>Throw</button><button onClick={reset}>Reset field</button><Link href="/experiments">Enter the records ↗</Link></div>
    </aside>
    <footer className="playground-footer"><span>AUTOLABS / OPEN RECORD</span><span>Five agents · one field · endless questions</span><Link href="/research">Open research ↗</Link></footer>
  </main>;
}

