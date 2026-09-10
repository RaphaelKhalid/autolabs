'use client';
import Link from 'next/link';
import {useCallback,useEffect,useRef,useState} from 'react';
import {AlienForm} from './autolabs-observatory';
import {rewardEta} from '../lib/reward-eta';
const API='https://autolabs-reward-compatibility.raphaelbahadurkhan.workers.dev';
const REPO='https://github.com/RaphaelKhalid/reward-compatibility';
interface Status {status:string;stage:string;reason:string|null;updatedAt:string;execution?:{concurrency:number};ledger?:{storageBytes:number;softLimitBytes:number};progress:{done:number;total:number;current:{id:string;phase:string;kind:string}|null};budget:{spentUsd:number;reservedUsd:number;capUsd:number;calls:number};active:{id:string;effort:string}[];gate:{pass:boolean;checks:Record<string,boolean>}|null;recent:{id:number;time:string;type:string;data:Record<string,unknown>}[];}
interface Log {id:string;state:string;prompt:string;effort:string;charged:number;error:string|null;result:{text:string;inputTokens:number;outputTokens:number}|null;}
interface Row {config:string;family:string;witnessRate:number|null;additionalMonitorLoss:number|null;lossInterval:number[]|null;correctnessEffect:number|null;reasoningRewardAttainment:number|null;completeRepeats:number;}
const labels:Record<string,string>={gate:'Feasibility checks',diagnostic:'Measuring compatibility',baseline:'Sealed baseline evaluation',train:'In-context optimization',evaluation:'Sealed final evaluation',report:'Researcher summary'};
const pct=(n:number|null)=>n===null?'—':`${(n*100).toFixed(0)}%`;
export function RewardLab(){
  const [status,setStatus]=useState<Status|null>(null),[error,setError]=useState<string|null>(null),[logs,setLogs]=useState<Log[]>([]),[pages,setPages]=useState([0]),[nextOffset,setNextOffset]=useState<number|null>(null),[logLoading,setLogLoading]=useState(true),[rows,setRows]=useState<Row[]|null>(null);
  const offset=pages[pages.length-1];
  const [observedAt,setObservedAt]=useState(0);
  const eta=rewardEta(status,observedAt);
  function turnPage(older:boolean){setLogLoading(true);setNextOffset(null);setPages(p=>older&&nextOffset!==null?[...p,nextOffset]:p.length>1?p.slice(0,-1):p);}
  const busy=useRef(false),mounted=useRef(true);
  const refresh=useCallback(async()=>{
    if(busy.current)return;busy.current=true;
    try {const r=await fetch(`${API}/status`,{signal:AbortSignal.timeout(15000)});if(!r.ok)throw Error();const data=await r.json();if(!data.progress||!data.budget)throw Error();if(mounted.current){setStatus(data);setObservedAt(Date.now());setError(null);}
      if(data.status==='complete'){const a=await fetch(`${API}/analysis`,{signal:AbortSignal.timeout(15000)});if(a.ok){const body=await a.json();if(mounted.current&&Array.isArray(body.rows))setRows(body.rows);}}
    }catch{if(mounted.current)setError('Connection delayed. The cloud runner does not depend on this page. Retrying automatically.');}finally{busy.current=false;}
  },[]);
  useEffect(()=>{mounted.current=true;void refresh();const timer=setInterval(()=>{if(document.visibilityState==='visible')void refresh();},15000);return()=>{mounted.current=false;clearInterval(timer);};},[refresh]);
  useEffect(()=>{const c=new AbortController();fetch(`${API}/logs?offset=${offset}`,{signal:c.signal}).then(r=>{if(!r.ok)throw Error();return r.json();}).then(d=>{if(Array.isArray(d.calls)&&!c.signal.aborted){setLogs(d.calls);setNextOffset(typeof d.next==='number'?d.next:null);setLogLoading(false);}}).catch(()=>{if(!c.signal.aborted)setLogLoading(false);});return()=>c.abort();},[offset,status?.budget.calls,status?.status]);
  const phase=status?.progress.current?.phase??status?.stage??'gate',running=status?.status==='running';
  const report=status?.recent.find(e=>typeof e.data.text==='string')?.data.text;
  const bounds=(rows??[]).flatMap(r=>r.lossInterval??(r.additionalMonitorLoss===null?[]:[r.additionalMonitorLoss]));
  const lower=Math.min(-.1,...bounds),upper=Math.max(.1,...bounds),span=upper-lower;
  const plotX=(n:number)=>315+((n-lower)/span)*355;
  return <main className="reward-page">
    <nav className="reward-nav" aria-label="Main navigation"><Link href="/">A / AUTOLABS</Link><div><Link href="/experiments">Experiments</Link><a href={REPO}>Source ↗</a></div></nav>
    <header className="reward-heading"><p className="reward-label">EXPERIMENT 002 · AI SAFETY</p><h1>Measuring reward<br/>compatibility.</h1><p>Can a small test predict when optimizing reasoning makes it harder to monitor?</p></header>
    <section className="reward-overview" aria-label="Live experiment">
      <div className={`reward-researcher ${running?'is-running':''}`}>
        <div className="reward-bubble" role="status">{status?status.status==='paused'?`Paused safely: ${status.reason?.replaceAll('_',' ')}`:status.status==='complete'?'Run complete. Evaluation records released.':labels[phase]??'Ready for the feasibility run.':'Connecting to the cloud laboratory…'}</div>
        <AlienForm agent={{id:'luna-002',name:'Luna',color:'#477969'}} index={2} meeting={phase==='report'}/>
        <div className="reward-desk"/><p className="reward-label">LUNA · SOLO RESEARCH COORDINATOR</p>
        <p className="reward-caption">Fresh actor and grader calls; isolated contexts.</p>
      </div>
      <div className="reward-readout">
        <div className="reward-state"><span className={running?'status-live':''}>{status?.status??'connecting'}</span><button onClick={()=>void refresh()} aria-label="Refresh experiment status">Refresh ↻</button></div>
        {error&&<p className="reward-warning" role="alert">{error}</p>}
        <h2>{labels[phase]??'Cloud execution'}</h2>
        <div className="reward-numbers"><div><strong>{status?.progress.done??'—'}<small> / {status?.progress.total??'—'}</small></strong><span>{status?.stage==='main'?'main-study units':'feasibility units'}</span></div><div><strong>${status?.budget.spentUsd.toFixed(2)??'—'}</strong><span>of $40 OpenAI cap</span></div></div>
        <progress value={status?.progress.done??0} max={status?.progress.total??68} aria-label="Current stage progress"/>
        <div className="reward-eta" aria-live="polite"><p className="reward-label">{status?.stage==='gate'?'FEASIBILITY ETA':'ESTIMATED FINISH'}</p><p>{error?'Updating estimate when connection returns…':eta.label}</p>{!error&&eta.earliest&&eta.latest&&<p className="reward-caption">{new Date(eta.earliest).toLocaleString(undefined,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})}–{new Date(eta.latest).toLocaleTimeString(undefined,{hour:'numeric',minute:'2-digit',timeZoneName:'short'})} · Rough recent-pace range; updates every 15 seconds. Later phases may differ.</p>}</div>
        <dl><div><dt>Active calls · up to {status?.execution?.concurrency??1}</dt><dd>{status?.active.length?status.active.map(call=><div key={call.id}>{call.id}</div>):'Between checkpoints'}</dd></div><div><dt>Reserved for pending calls</dt><dd>${status?.budget.reservedUsd.toFixed(4)??'0.0000'}</dd></div>{status?.ledger&&<div><dt>Ledger · safe pause threshold</dt><dd>{(status.ledger.storageBytes/1048576).toFixed(1)} / {(status.ledger.softLimitBytes/1048576).toFixed(0)} MiB</dd></div>}<div><dt>Actor / evaluators</dt><dd>Luna None / Luna High</dd></div><div><dt>Last checkpoint</dt><dd>{status?new Date(status.updatedAt).toLocaleTimeString():'Awaiting connection'}</dd></div></dl>
        {status?.gate&&<details><summary>Feasibility: {status.gate.pass?'passed':'did not pass'}</summary>{Object.entries(status.gate.checks).map(([k,v])=><p key={k}>{v?'✓':'×'} {k}</p>)}</details>}
      </div>
    </section>
    <div className="reward-links"><a href={`${REPO}/blob/main/PROTOCOL.md`}>Registered protocol ↗</a><a href={`${API}/results`}>Scored records ↗</a><a href={`${API}/logs`}>API ledger ↗</a><Link href="/experiments/erdos-885">Pilot 001 archive</Link></div>
    <section className="reward-section"><p className="reward-label">METHOD</p><div className="reward-method">{[['01','Search','Find readable strategies meeting each reward.'],['02','Optimize','Matched outcome-only and combined-reward histories.'],['03','Evaluate','Blind monitoring on held-out tasks.']].map(([n,t,d])=><div key={n}><span>{n}</span><h3>{t}</h3><p>{d}</p></div>)}</div><p className="reward-caption">Eight configurations · three repeats · five fresh transfer tasks per candidate. Coin tracking only; the code-backdoor arm is deferred. No weight updates. No claim of private reasoning access.</p></section>
    <section className="reward-section"><p className="reward-label">RESULT FIGURE</p><h2>Prediction versus observed monitoring loss</h2>{rows?<>
      <svg className="reward-figure" viewBox="0 0 700 440" role="img" aria-label="Each reward's diagnostic witness rate and additional monitoring loss, with full 95 percent repeat-level intervals">
        <text x="10" y="25">Reward</text><text x="150" y="25">Witness rate</text><text x="495" y="25" textAnchor="middle">Additional monitoring loss</text>
        <line x1={plotX(0)} x2={plotX(0)} y1="40" y2="376" stroke="#9ca998" strokeDasharray="3 4"/>
        {rows.map((r,i)=>{const y=65+i*40;return <g key={r.config}><title>{r.config}: {pct(r.additionalMonitorLoss)} extra loss; CI {r.lossInterval?.map(pct).join(' to ')}</title><text x="10" y={y+4}>{r.config}</text><rect x="150" y={y-6} width="100" height="12" fill="#e0e4d7"/><rect x="150" y={y-6} width={(r.witnessRate??0)*100} height="12" fill="#839f80"/><text x="265" y={y+4} textAnchor="middle">{pct(r.witnessRate)}</text>{r.lossInterval&&<line x1={plotX(r.lossInterval[0])} x2={plotX(r.lossInterval[1])} y1={y} y2={y} stroke="#477969" strokeWidth="2"/>}{r.additionalMonitorLoss!==null&&<circle cx={plotX(r.additionalMonitorLoss)} cy={y} r="5" fill="#305b4a"/>}</g>;})}
        <line x1="315" y1="386" x2="670" y2="386" stroke="currentColor"/>
        {[lower,(lower+upper)/2,upper].map(v=><text key={v} x={plotX(v)} y="406" textAnchor="middle">{(v*100).toFixed(0)} pp</text>)}
        <text x="495" y="432" textAnchor="middle">Positive = greater monitoring loss</text>
      </svg><p className="reward-caption">Positive loss means combined-reward optimization reduced monitoring more than the outcome-only control. Bars: full 95% t intervals across three histories, in percentage points (pp). The diagnostic is a model-audited proxy, not a proof of compatibility. Family-held-out prediction comparisons and diagnostic uncertainty are included in the full analysis.</p><a href={`${API}/analysis`}>Download full estimates and intervals ↗</a>
      <div className="reward-table-wrap"><table><thead><tr><th>Reward</th><th>Witness rate</th><th>Extra monitoring loss</th><th>Correctness effect</th><th>Reward attained</th></tr></thead><tbody>{rows.map(r=><tr key={r.config}><td>{r.config}</td><td>{pct(r.witnessRate)}</td><td>{pct(r.additionalMonitorLoss)}</td><td>{pct(r.correctnessEffect)}</td><td>{pct(r.reasoningRewardAttainment)}</td></tr>)}</tbody></table></div>
    </>:<div className="reward-sealed"><span>Evaluation sealed</span><p>The figure appears after the frozen run completes. No interim test scores enter the research loop.</p></div>}</section>
    {typeof report==='string'&&<section className="reward-section"><p className="reward-label">RESEARCHER NOTE</p><p>{report}</p></section>}
    <section className="reward-section"><p className="reward-label">RESEARCH RECORD</p><h2>Visible API output</h2><p className="reward-caption">Completed output, prompts and usage. Private model reasoning is not available. Evaluation calls remain sealed until completion.</p>{logs.length?logs.map(log=><details className="reward-log" key={log.id}><summary><span>{log.id}</span><small>{log.state} · {log.effort} · ${(log.charged/1e6).toFixed(4)}</small></summary><h3>Output</h3><pre>{log.result?.text??log.error??'Request in progress.'}</pre><h3>Prompt</h3><pre>{log.prompt}</pre></details>):<p>No public calls on this ledger page yet.</p>}<div className="reward-pagination"><button disabled={offset===0||logLoading} onClick={()=>turnPage(false)}>Newer</button><button disabled={nextOffset===null||logLoading} onClick={()=>turnPage(true)}>Older</button></div></section>
    <footer><p>Based on <a href="https://arxiv.org/abs/2603.30036">Kaufmann et al., 2026</a>. Exploratory evidence, not a safety certificate. All public controls are read-only.</p></footer>
  </main>;
}
