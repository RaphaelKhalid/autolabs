"use client";

import { useState } from "react";
import "./trace-mark-animation-preview.css";

type Phase = "trace" | "complete" | "settle";

const phases: { id: Phase; label: string; note: string }[] = [
  { id: "trace", label: "01 / trace", note: "draw the signal" },
  { id: "complete", label: "02 / complete", note: "let the structure emerge" },
  { id: "settle", label: "03 / settle", note: "keep what repeats" },
];

function Mark({ phase }: { phase: Phase }) {
  return (
    <svg className={`trace-mark trace-mark--${phase}`} viewBox="0 0 620 410" role="img" aria-label="Animated AutoLabs trace mark">
      <defs>
        <linearGradient id="trace-ink" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#d46b42" />
          <stop offset="0.58" stopColor="#e7e1d2" />
          <stop offset="1" stopColor="#6ca0a0" />
        </linearGradient>
        <filter id="trace-soft-glow" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="5" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      <g className="trace-grid" aria-hidden="true">
        <path d="M35 205H585" />
        <path d="M310 30V380" />
        <circle cx="310" cy="205" r="142" />
        <circle cx="310" cy="205" r="96" />
      </g>

      <g className="trace-orbit" fill="none" stroke="url(#trace-ink)">
        <ellipse cx="310" cy="205" rx="238" ry="68" transform="rotate(-18 310 205)" />
        <ellipse cx="310" cy="205" rx="238" ry="68" transform="rotate(38 310 205)" />
        <ellipse cx="310" cy="205" rx="200" ry="52" transform="rotate(86 310 205)" />
      </g>

      <g className="trace-a" fill="none" stroke="url(#trace-ink)" strokeLinecap="round" strokeLinejoin="round">
        <path d="M148 330C210 320 242 262 275 118C287 66 302 52 317 61C337 74 341 144 351 222C359 287 388 329 455 332" />
        <path d="M206 252C264 238 338 237 405 252" />
        <path d="M291 75C307 113 306 191 310 242" />
      </g>

      <g className="trace-nodes" filter="url(#trace-soft-glow)">
        <circle cx="442" cy="137" r="11" />
        <circle cx="442" cy="137" r="4" />
        <circle cx="198" cy="280" r="4" />
        <circle cx="310" cy="242" r="3" />
      </g>

      <g className="trace-caption">
        <text x="310" y="397" textAnchor="middle">SIGNAL / 03A / AUTOLABS</text>
      </g>
    </svg>
  );
}

export default function TraceMarkAnimationPreview() {
  const [phase, setPhase] = useState<Phase>("trace");
  const [running, setRunning] = useState(false);

  const advance = () => {
    setRunning(true);
    setPhase((current) => phases[(phases.findIndex((item) => item.id === current) + 1) % phases.length].id);
    window.setTimeout(() => setRunning(false), 900);
  };

  return (
    <main className="trace-demo">
      <header className="trace-demo__header">
        <span className="trace-demo__eyebrow">AutoLabs / mark study</span>
        <span className="trace-demo__status"><i /> live glyph system</span>
      </header>

      <section className="trace-demo__hero">
        <div className="trace-demo__copy">
          <p className="trace-demo__kicker">A readable mark with a changing interior.</p>
          <h1>Trace the signal.<br /><em>Keep the pattern.</em></h1>
          <p className="trace-demo__body">The logo keeps a stable silhouette while its linework moves through three research states: trace, complete, settle.</p>
          <button className="trace-demo__button" type="button" onClick={advance} aria-label="Advance the animated mark">
            <span>{running ? "evolving…" : "advance mark"}</span><b>↗</b>
          </button>
        </div>
        <div className="trace-demo__mark-wrap">
          <Mark phase={phase} />
        </div>
      </section>

      <section className="trace-demo__timeline" aria-label="Mark phases">
        {phases.map((item) => (
          <button className={item.id === phase ? "is-active" : ""} key={item.id} onClick={() => setPhase(item.id)} type="button">
            <span className="trace-demo__phase-number">{item.label.split(" /")[0]}</span>
            <span><strong>{item.label.split(" / ")[1]}</strong><small>{item.note}</small></span>
          </button>
        ))}
      </section>
    </main>
  );
}
