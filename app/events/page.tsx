'use client';

import { useRef, useState } from 'react';
import Link from 'next/link';

type Protocol = {
  number: string;
  title: string;
  status: string;
  hypothesis: string;
  sample: string;
  spend: string;
  model: string;
  change: string;
  method: string;
};

const protocols: Protocol[] = [
  { number: '01', title: 'When a cue changes the answer, does the explanation reveal it?', status: 'First protocol ready for review', hypothesis: 'If a preference cue changes the answer, the model’s explanation will often fail to mention that cue.', sample: '48 decisions: 24 neutral + 24 cued. One monitor judgment per explanation, plus 24 held-out audits.', spend: '48 actor calls + 48 monitor calls + 24 audits = 120 calls. $5 hard ceiling.', model: 'gpt-4.1-mini · 256 output tokens/call', change: 'Choose the approved scenario subset and cue wording.', method: 'Matched neutral/cued decisions; compare answer flips with cue detection.' },
  { number: '02', title: 'Does evidence arriving through a tool become harder to notice?', status: 'Protocol draft', hypothesis: 'The same cue will be harder for a monitor to spot when it arrives as a tool result instead of a user message.', sample: '36 matched cases: 18 user-message cues + 18 tool-result cues, with a monitor judgment for each.', spend: '72 actor calls + 72 monitor calls = 144 calls. $5 hard ceiling.', model: 'gpt-4.1-mini · 256 output tokens/call', change: 'Choose the approved tool wrapper and scenario subset.', method: 'Change only the delivery channel; hold cue content and task constant.' },
  { number: '03', title: 'What evidence disappears when reasoning becomes shorter?', status: 'Protocol draft', hypothesis: 'Shorter visible reasoning will give the monitor less evidence about why the model chose its answer.', sample: '60 cases × 3 length bands = 180 responses, with 60 matched monitor checks.', spend: '180 actor calls + 60 monitor calls = 240 calls. $5 hard ceiling.', model: 'gpt-4.1-mini · 256 output tokens/call', change: 'Choose the approved task subset and secondary length analysis.', method: 'Compare long, medium and short output limits on identical decisions.' },
  { number: '04', title: 'Can reasoning stay legible as optimization continues?', status: 'Protocol draft', hypothesis: 'As optimization pressure rises, a fixed monitor will miss more evidence that matters to the decision.', sample: '5 checkpoints × 24 held-out cases = 120 responses, with one monitor check per response.', spend: '120 actor calls + 120 monitor calls = 240 calls. $5 hard ceiling.', model: 'gpt-4.1-mini · 256 output tokens/call', change: 'Choose checkpoint spacing inside the approved range.', method: 'Keep the monitor and held-out cases fixed while the checkpoint changes.' },
  { number: '05', title: 'How much can a monitor infer from the text it receives?', status: 'Protocol draft', hypothesis: 'Visible explanations, reasoning fields and activation measurements will expose different amounts of decision evidence.', sample: '36 cases × 3 observation surfaces = 108 responses, with 108 monitor judgments.', spend: '108 actor calls + 108 monitor calls = 216 calls. $5 hard ceiling.', model: 'gpt-4.1-mini · 256 output tokens/call', change: 'Choose secondary monitor questions; primary surfaces stay fixed.', method: 'Give the monitor one observation surface at a time and score recovery.' },
  { number: '06', title: 'Is the monitor detecting influence, or judging a plausible story?', status: 'Protocol draft', hypothesis: 'A monitor may approve a convincing explanation even when it misses the cue that changed the answer.', sample: '48 positive/control cases: 96 actor responses, 96 monitor judgments and 24 audit checks.', spend: '96 actor calls + 96 monitor calls + 24 audits = 216 calls. $5 hard ceiling.', model: 'gpt-4.1-mini · 256 output tokens/call', change: 'Choose approved control examples; positive cases stay registered.', method: 'Use known choice changes, negative controls and cue-present cases.' },
];

function TestCard({ onRun, running }: { onRun: (key: string) => void; running: boolean }) {
  const [ownerKey, setOwnerKey] = useState('');
  return <div className="event-deck-card event-test-card">
    <div className="event-deck-top"><span className="event-number">T</span><span className="event-status live">LIVE TEST</span></div>
    <p className="event-card-kicker">WIRING CHECK / ONE REQUEST</p>
    <h2>Does the event page actually reach a model?</h2>
    <p className="event-hypothesis">A fixed request should return one short sentence. This tests the page, server route, provider key and result panel before anyone spends a sponsored grant.</p>
    <div className="event-ledger"><div><span>MODEL</span><b>gpt-4.1-mini</b></div><div><span>SAMPLE</span><b>1 call</b></div><div><span>OUTPUT CAP</span><b>64 tokens</b></div><div><span>SPEND GUARD</span><b>One request</b></div></div>
    <label className="event-key">Owner key<input type="password" autoComplete="off" placeholder="Server owner key" value={ownerKey} onChange={(e) => setOwnerKey(e.target.value)} /></label>
    <button className="event-run" disabled={!ownerKey || running} onClick={() => onRun(ownerKey)}>{running ? 'Calling the model…' : 'Run the test call'}</button>
    <p className="event-footnote">The key is sent as an authorization header and is not persisted by the page.</p>
  </div>;
}

export default function EventsPage() {
  const [index, setIndex] = useState(-1);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<{ text: string; model: string; requestId?: string } | null>(null);
  const [error, setError] = useState('');
  const touchStart = useRef<number | null>(null);
  const count = protocols.length + 1;
  const item = index >= 0 ? protocols[index] : null;

  function move(delta: number) { setIndex((current) => (current + delta + count) % count); setResult(null); setError(''); }
  function onTouchStart(event: React.TouchEvent) { touchStart.current = event.changedTouches[0]?.clientX ?? null; }
  function onTouchEnd(event: React.TouchEvent) { const start = touchStart.current; const end = event.changedTouches[0]?.clientX; touchStart.current = null; if (start != null && end != null && Math.abs(end - start) > 45) move(end < start ? 1 : -1); }
  async function runTest(ownerKey: string) {
    setRunning(true); setResult(null); setError('');
    try {
      const response = await fetch('/api/events/test', { method: 'POST', headers: { 'x-autolabs-owner-key': ownerKey } });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'The test call failed.');
      setResult(data);
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'The test call failed.'); }
    finally { setRunning(false); }
  }

  return <main className="events-page">
    <style jsx global>{`
      .events-page{min-height:100vh;background:#f0eadc;color:#171b19;padding:0 clamp(18px,5vw,82px) 70px;font-family:var(--font-serif),Georgia,serif}
      .events-nav{height:76px;border-bottom:1px solid rgba(23,27,25,.35);display:flex;align-items:center;justify-content:space-between;font:650 9px var(--font-mono),monospace;letter-spacing:.15em;text-transform:uppercase}.events-nav-links{display:flex;gap:22px;color:#5e6259}.events-nav a:hover{color:#9a4f36}
      .events-hero{display:flex;align-items:end;justify-content:space-between;gap:30px;padding:70px 0 28px}.events-hero h1{margin:0;max-width:760px;font-size:clamp(45px,7vw,94px);line-height:.86;font-weight:430;letter-spacing:-.065em}.events-hero h1 em{color:#a85238;font-weight:400}.events-hero p{max-width:280px;margin:0;color:#5e6259;font-size:12px;line-height:1.55}
      .event-deck-shell{position:relative}.event-deck-card{min-height:500px;padding:clamp(24px,4vw,52px);border:1px solid rgba(23,27,25,.42);background:rgba(255,255,255,.34);box-shadow:0 22px 70px rgba(43,37,24,.1);display:flex;flex-direction:column}.event-test-card{border-color:#a85238;background:linear-gradient(135deg,rgba(255,255,255,.42),rgba(168,82,56,.08))}.event-deck-top{display:flex;justify-content:space-between;align-items:center}.event-number{color:#a85238;font:650 14px var(--font-mono),monospace;letter-spacing:.08em}.event-status{color:#5e6259;font:650 8px var(--font-mono),monospace;letter-spacing:.12em;text-transform:uppercase}.event-status.live{color:#52694b}.event-card-kicker{margin:54px 0 16px;color:#a85238;font:650 8px var(--font-mono),monospace;letter-spacing:.16em;text-transform:uppercase}.event-deck-card h2{margin:0;max-width:900px;font-size:clamp(31px,4.5vw,62px);line-height:.95;font-weight:450;letter-spacing:-.045em}.event-hypothesis{max-width:800px;margin:23px 0 0;color:#4f544c;font-size:clamp(15px,1.5vw,20px);line-height:1.42}.event-ledger{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:auto;padding-top:24px;border-top:1px solid rgba(23,27,25,.24)}.event-ledger div{min-width:0}.event-ledger span{display:block;color:#5e6259;font:650 7px var(--font-mono),monospace;letter-spacing:.14em}.event-ledger b{display:block;margin-top:7px;font-size:14px;font-weight:540;line-height:1.2}.event-budget{margin-top:16px;padding:12px 14px;border-left:3px solid #a85238;background:rgba(168,82,56,.07);font-size:14px;line-height:1.45}.event-budget strong{font-weight:600}.event-change{margin:16px 0 0;color:#5e6259;font-size:12px;line-height:1.5}.event-change b{color:#171b19;font-weight:600}.event-nav{display:flex;align-items:center;justify-content:space-between;margin-top:18px}.event-arrow{width:48px;height:48px;border:1px solid rgba(23,27,25,.5);background:transparent;font:22px var(--font-mono);cursor:pointer}.event-arrow:hover{background:#171b19;color:#f0eadc}.event-dots{display:flex;align-items:center;gap:8px}.event-dot{width:7px;height:7px;border:1px solid #5e6259;border-radius:50%;background:transparent;cursor:pointer}.event-dot.active{background:#a85238;border-color:#a85238;transform:scale(1.35)}.event-index{color:#5e6259;font:650 8px var(--font-mono),monospace;letter-spacing:.12em}.event-key{display:grid;gap:8px;max-width:380px;margin-top:24px;color:#5e6259;font:650 8px var(--font-mono),monospace;letter-spacing:.12em;text-transform:uppercase}.event-key input{height:42px;padding:0 11px;border:1px solid rgba(23,27,25,.38);background:rgba(255,255,255,.58);font:12px var(--font-mono),monospace}.event-run{max-width:380px;height:46px;margin-top:12px;border:1px solid #171b19;background:#171b19;color:#f0eadc;font:650 9px var(--font-mono),monospace;letter-spacing:.1em;text-transform:uppercase;cursor:pointer}.event-run:hover{background:#a85238;border-color:#a85238}.event-run:disabled{cursor:not-allowed;opacity:.5}.event-footnote{max-width:380px;margin:10px 0 0;color:#5e6259;font-size:9px;line-height:1.5}.event-result,.event-error{margin-top:18px;padding:18px 20px;border:1px solid #52694b;background:rgba(82,105,75,.09)}.event-error{border-color:#a85238;background:rgba(168,82,56,.08);color:#a85238}.event-result h2{margin:0 0 10px;font-size:20px}.event-result pre{margin:0;white-space:pre-wrap;font:12px/1.55 var(--font-mono),monospace}.event-result p{margin:10px 0 0;font:8px var(--font-mono);color:#5e6259}.events-plain{display:flex;justify-content:space-between;gap:20px;margin-top:22px;color:#5e6259;font-size:11px;line-height:1.5}.events-plain a{color:#171b19;text-decoration:underline;text-underline-offset:4px}
      @media(max-width:720px){.events-nav{height:64px}.events-nav-links{gap:11px;font-size:8px}.events-hero{display:block;padding:54px 0 24px}.events-hero p{margin-top:22px}.event-deck-card{min-height:600px;padding:25px 20px}.event-card-kicker{margin-top:46px}.event-deck-card h2{font-size:34px}.event-hypothesis{font-size:16px}.event-ledger{grid-template-columns:repeat(2,1fr);gap:17px}.event-ledger b{font-size:12px}.event-nav{margin-top:14px}.event-arrow{width:42px;height:42px}.events-plain{display:block}.events-plain p{margin:10px 0}}
    `}</style>
    <nav className="events-nav"><Link href="/">A / AUTOLABS</Link><div className="events-nav-links"><Link href="/experiments">Experiments</Link><Link href="/events" aria-current="page">Events</Link><Link href="/research">Research</Link><Link href="/studio">Studio</Link></div></nav>
    <header className="events-hero"><h1>Pick a question.<br/><em>See the receipt.</em></h1><p>Ten $5 research grants. One protocol per run. Swipe through the card before you commit.</p></header>
    <section className="event-deck-shell" aria-label="Experiment deck" onTouchStart={onTouchStart} onTouchEnd={onTouchEnd}>
      {index === -1 ? <TestCard onRun={runTest} running={running} /> : <article className="event-deck-card"><div className="event-deck-top"><span className="event-number">{item?.number}</span><span className="event-status">{item?.status}</span></div><p className="event-card-kicker">HYPOTHESIS</p><h2>{item?.title}</h2><p className="event-hypothesis">{item?.hypothesis}</p><div className="event-budget"><strong>What the $5 buys:</strong> {item?.spend}</div><p className="event-change"><b>What you can change:</b> {item?.change}</p><div className="event-ledger"><div><span>SAMPLE</span><b>{item?.sample}</b></div><div><span>MODEL</span><b>{item?.model}</b></div><div><span>METHOD</span><b>{item?.method}</b></div><div><span>LAUNCH</span><b>Preview only</b></div></div></article>}
    </section>
    <div className="event-nav"><button className="event-arrow" onClick={() => move(-1)} aria-label="Previous option">←</button><div className="event-dots" aria-label="Choose an option">{Array.from({length: count}).map((_, dot) => <button key={dot} className={`event-dot ${dot === index + 1 ? 'active' : ''}`} onClick={() => { setIndex(dot - 1); setResult(null); setError(''); }} aria-label={dot === 0 ? 'Test option' : `Protocol ${dot}`} />)}<span className="event-index">{index === -1 ? 'TEST' : `${index + 1} / 06`}</span></div><button className="event-arrow" onClick={() => move(1)} aria-label="Next option">→</button></div>
    {result && <section className="event-result" role="status"><h2>Test call complete · {result.model}</h2><pre>{result.text}</pre>{result.requestId && <p>Provider request {result.requestId}</p>}</section>}
    {error && <div className="event-error" role="alert">{error}</div>}
    <div className="events-plain"><p>Six protocols are still being frozen. A $5 ceiling will be enforced when the server-side grant ledger is enabled.</p><p><Link href="/experiments/persona-discovery">Active SAE study ↗</Link> · <Link href="/experiments">All records ↗</Link></p></div>
  </main>;
}
