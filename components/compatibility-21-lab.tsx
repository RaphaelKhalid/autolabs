'use client';
import Link from 'next/link';
import {useCallback, useEffect, useRef, useState} from 'react';
import {AlienForm} from './autolabs-observatory';
import {COMPATIBILITY21_API as API, compatibility21CallTitle, compatibility21Eta, validCompatibility21Status, type Compatibility21Status} from '@/lib/compatibility-21-display';

const REPO = 'https://github.com/RaphaelKhalid/reward-compatibility';
type Log = {id:string; state:string; charged:number; prompt:string; error:string|null; response:{text?:string}|null; split:string; parsed:number|null};
type LogPage = {sealed:boolean; next:number|null; calls:Log[]};
function validLogs(value: unknown): value is LogPage {
  if (!value || typeof value !== 'object') return false;
  const page=value as Partial<LogPage>;
  return typeof page.sealed==='boolean' && (page.next===null || (Number.isSafeInteger(page.next)&&Number(page.next)>=0)) && Array.isArray(page.calls) && page.calls.length<=5 && page.calls.every(log=>log&&typeof log.id==='string'&&typeof log.state==='string'&&typeof log.prompt==='string'&&typeof log.charged==='number'&&(log.response===null||typeof log.response==='object'));
}
function money(value: number|undefined, digits=2) {return value===undefined?'—':`$${value.toFixed(digits)}`;}

export function Compatibility21Lab() {
  const [status,setStatus]=useState<Compatibility21Status|null>(null);
  const [error,setError]=useState<string|null>(null),[refreshing,setRefreshing]=useState(false);
  const [logs,setLogs]=useState<LogPage|null>(null),[logError,setLogError]=useState(false),[logLoading,setLogLoading]=useState(true);
  const [pages,setPages]=useState([0]),[retry,setRetry]=useState(0);
  const offset=pages.at(-1)??0;
  const request=useRef<AbortController|null>(null);
  const refresh=useCallback(async()=>{
    if(request.current)return;
    const controller=new AbortController();request.current=controller;setRefreshing(true);
    const timer=setTimeout(()=>controller.abort('timeout'),15000);
    try {
      const response=await fetch(`${API}/status`,{signal:controller.signal,cache:'no-store'});
      if(!response.ok)throw Error('status');
      const data:unknown=await response.json();
      if(!validCompatibility21Status(data))throw Error('schema');
      if(!controller.signal.aborted){setStatus(data);setError(null);}
    }catch{
      if(controller.signal.reason!=='unmount')setError('Connection delayed. Showing the last available checkpoint; cloud execution is independent of this page.');
    }finally{
      clearTimeout(timer);if(request.current===controller)request.current=null;
      if(controller.signal.reason!=='unmount')setRefreshing(false);
    }
  },[]);
  useEffect(()=>{
    void refresh();
    const timer=setInterval(()=>{if(document.visibilityState==='visible')void refresh();},15000);
    const onVisible=()=>{if(document.visibilityState==='visible')void refresh();};
    document.addEventListener('visibilitychange',onVisible);
    return()=>{clearInterval(timer);document.removeEventListener('visibilitychange',onVisible);request.current?.abort('unmount');request.current=null;};
  },[refresh]);
  useEffect(()=>{
    const controller=new AbortController();const timer=setTimeout(()=>controller.abort('timeout'),15000);
    fetch(`${API}/logs?offset=${offset}`,{signal:controller.signal,cache:'no-store'}).then(async response=>{
      if(!response.ok)throw Error('logs');
      const data:unknown=await response.json();if(!validLogs(data))throw Error('schema');
      if(!controller.signal.aborted){setLogs(data);setLogError(false);setLogLoading(false);}
    }).catch(()=>{if(controller.signal.reason!=='unmount'){setLogError(true);setLogLoading(false);}}).finally(()=>clearTimeout(timer));
    return()=>{controller.abort('unmount');clearTimeout(timer);};
  },[offset,retry,status?.budget.calls,status?.status,status?.updatedAt]);
  const running=status?.status==='running';
  const phase=status?.status==='complete'?'Evaluation released':status?.stage==='eval'?'Held-out evaluation':'Development & safety gate';
  const phaseText=!status?'Connecting to the cloud laboratory…':status.status==='ready'?'Protocol ready. Awaiting verified launch.':status.status==='paused'?`Paused safely: ${status.reason?.replaceAll('_',' ')??'review required'}`:status.status==='complete'?'Fixed run complete. All records are available.':status.stage==='eval'?'Testing the frozen method. Held-out outputs remain sealed.':'Checking proposals against exact, independent answers.';
  const sealed=status?.evaluationSealed!==false;
  const current=status?.progress.cases.find(row=>row.split===status.stage);
  const onPage=(older:boolean)=>{setLogLoading(true);setLogs(null);setPages(previous=>older&&logs?.next!=null?[...previous,logs.next]:previous.length>1?previous.slice(0,-1):previous);};

  return <main className="reward-page compatibility21-page">
    <nav className="reward-nav" aria-label="Main navigation"><Link href="/">A / AUTOLABS</Link><div><Link href="/experiments">Experiments</Link><a href={REPO}>Source ↗</a></div></nav>
    <header className="reward-heading"><p className="reward-label">EXPERIMENT 002.1 · AI SAFETY</p><h1>Testing reward<br/>compatibility.</h1><p>Can a checked example establish that two rewards can be satisfied together?</p></header>
    <section className="reward-overview" aria-label="Live experiment">
      <div className={`reward-researcher ${running?'is-running':''}`}>
        <div className="reward-bubble" role="status">{phaseText}</div>
        <AlienForm agent={{id:'luna-0021',name:'Luna',color:'#477969'}} index={2} meeting={false}/>
        <div className="reward-desk"/><p className="reward-label">LUNA · INDEPENDENT API CALLS</p>
        <div className="compatibility21-slots" aria-label={`${status?.active.length??0} active calls; capacity ${status?.execution.concurrency??8}`}>
          {Array.from({length:Math.min(8,status?.execution.concurrency??8)},(_,index)=><span key={index} className={running&&index<(status?.active.length??0)?'is-active':''} aria-hidden="true"/>)}
        </div>
        <p className="reward-caption">One experiment · up to eight calls in parallel</p>
      </div>
      <div className="reward-readout">
        <div className="reward-state"><span className={running?'status-live':''}>{status?.status??'connecting'}</span><button disabled={refreshing} onClick={()=>{void refresh();setRetry(value=>value+1);}}>{refreshing?'Refreshing…':'Refresh ↻'}</button></div>
        {error&&<p className="reward-warning" role="alert">{error}</p>}
        <h2>{phase}</h2>
        <div className="reward-numbers"><div><strong>{status?.progress.callsDone.toLocaleString()??'—'}<small> / {status?.progress.callsTotal.toLocaleString()??'3,600'}</small></strong><span>planned call steps completed</span></div></div>
        <progress value={status?.progress.callsDone??0} max={status?.progress.callsTotal||3600} aria-label="Overall fixed-study progress"/>
        <p className="compatibility21-eta" aria-live="polite">{error?'ETA unavailable while reconnecting':compatibility21Eta(status)}</p>
        <p className="reward-caption">Rough elapsed-pace estimate, refreshed every 15 seconds. Different phases may take longer.</p>
        <dl>
          <div><dt>Phase jobs completed</dt><dd>{current?`${current.done} / ${current.total}`:'Awaiting first checkpoint'}</dd></div>
          <div><dt>Model / reasoning effort</dt><dd>{status?.model??'gpt-5.6-luna'} / none</dd></div>
          <div><dt>Isolation preflight</dt><dd>{status?.isolation?.passed===true?'Passed':status?.isolation?'Not passed':'Awaiting verification'}</dd></div>
          <div><dt>Last checkpoint</dt><dd>{status?new Date(status.updatedAt).toLocaleString():'Awaiting connection'}</dd></div>
        </dl>
        {status?.gate&&<p className="reward-caption">Development gate: {status.gate.pass?'passed':'not passed'}{typeof status.gate.parseRate==='number'?` · ${(status.gate.parseRate*100).toFixed(1)}% parseable`:''}. This checks validity and budget, not whether results are positive.</p>}
      </div>
    </section>
    <div className="reward-links"><a href={`${API}/protocol`}>Frozen protocol ↗</a><a href={`${API}/results`}>Scored records ↗</a><a href={`${API}/logs`}>API ledger ↗</a><Link href="/experiments/reward-compatibility">Experiment 002 archive</Link><Link href="/experiments/reward-categories-22">Follow-on: Experiment 002.2</Link></div>
    <section className="reward-section" aria-labelledby="budget21"><p className="reward-label">SHARED BUDGET</p><h2 id="budget21">{money(status?.budget.totalCommittedUsd)} committed / $40 cap</h2>
      <div className="compatibility21-budget"><div><span>Experiment 002 + prior reservations</span><strong>{money(status?.budget.priorCommittedUsd,4)}</strong></div><div><span>002.1 conservative accounted spend</span><strong>{money(status?.budget.spentUsd,4)}</strong></div><div><span>Reserved, not yet settled</span><strong>{money(status?.budget.reservedUsd,4)}</strong></div></div>
      {(status?.budget.knownSpendUsd!==undefined||status?.budget.uncertainChargeUpperBoundUsd!==undefined)&&<p className="reward-caption">Within accounted spend: {money(status?.budget.knownSpendUsd,4)} from reported usage; {money(status?.budget.uncertainChargeUpperBoundUsd,4)} maximum allowance for requests with unknown charges. The allowance is not a confirmed provider charge.</p>}
      <p className="reward-caption">The $40 ceiling is shared, not reset for this run. Pending requests retain reservations; failed requests with unknown charges remain conservatively accounted. Usage is application accounting, not a reconciled provider invoice.</p>
      {status?.ledger&&<p className="reward-caption">Ledger: {(status.ledger.storageBytes/1048576).toFixed(1)} / {(status.ledger.softLimitBytes/1048576).toFixed(0)} MiB safe-pause threshold.</p>}
    </section>
    <section className="reward-section"><p className="reward-label">WHAT IS BEING MEASURED</p><h2>Two rewards. Independently checked.</h2><div className="reward-metric-grid">
      <div><h3><i>r</i><sub>out</sub></h3><p>Does the proposed answer or finite program meet the outcome requirement?</p><p className="reward-caption">Exact coin-state or bounded integer checks.</p></div>
      <div><h3><i>r</i><sub>CoT</sub></h3><p>Does the visible trace meet the specified reasoning constraint?</p><p className="reward-caption">A check on public structured output, not hidden chain of thought.</p></div>
      <div><h3>Joint witness</h3><p>A single valid proposal that satisfies both reward requirements.</p><p className="reward-caption">Failure to find one is not a proof of conflict.</p></div>
    </div><p className="reward-caption">This stage validates compatible versus conflicting requirements within a declared finite language. It does not yet establish all three aligned / orthogonal / in-conflict categories in unrestricted tasks.</p></section>
    <section className="reward-section"><p className="reward-label">FIXED COMPARISON</p><div className="reward-method">{[['01','Description only','Judge the reward pair from its specification.'],['02','Unguided search','Propose examples without verifier feedback.'],['03','Verifier-guided search','Use exact feedback to improve the next proposals.']].map(([number,title,description])=><div key={number}><span>{number}</span><h3>{title}</h3><p>{description}</p></div>)}</div><p className="reward-caption">80 development cases · 320 held-out cases · 40 held-out templates · two domains. The backdoor task uses a restricted affine-trigger JSON language, not arbitrary generated code. Fixed sample size; no stopping for significance.</p></section>
    <section className="reward-section"><p className="reward-label">EVALUATION</p><h2>{sealed?'Held-out results are sealed':'Fixed evaluation is available'}</h2><div className="reward-sealed"><span>{sealed?'No interim held-out scores':'Evaluation unsealed'}</span><p>{sealed?'Development logs are public. Held-out calls and outcomes are withheld until the fixed run completes.': 'The run has ended. Inspect the full scored records and analysis before interpreting individual examples as findings.'}</p>{!sealed&&<a href={`${API}/results`}>Open evaluated records ↗</a>}</div></section>
    <section className="reward-section"><p className="reward-label">LIVE QUEUE</p><h2>{status?.active.length??0} active / {status?.execution.concurrency??8} maximum</h2>
      {status?.active.length?<ul className="compatibility21-active">{status.active.map(call=><li key={call.id}><span>{compatibility21CallTitle(call.id)}</span><small>{call.id}</small></li>)}</ul>:<p className="reward-caption">{status?.status==='complete'?'No calls remain.':status?'Between checkpoints.':'Connecting to the queue…'}</p>}
    </section>
    <section className="reward-section"><p className="reward-label">RESEARCH RECORD</p><h2>{sealed?'Development calls':'Released API calls'}</h2><p className="reward-caption">Public explanations and proposals, not private reasoning. Five records per page; no full-ledger download is needed to watch progress.</p>
      {logError&&<p className="reward-warning" role="alert">The ledger could not refresh. <button onClick={()=>{setLogLoading(true);setRetry(value=>value+1);}}>Retry ledger</button></p>}
      {logLoading?<p role="status">Loading records…</p>:logs?.calls.length?logs.calls.map(log=><details className="reward-log" key={log.id}><summary><span><span className="reward-call-title">{compatibility21CallTitle(log.id)}</span><span className="reward-call-id">{log.id}</span></span><small>{log.state} · {money(log.charged/1e6,4)}</small></summary><p className="reward-caption">{log.split==='eval'?'Held-out':'Development'} · {log.parsed===1?'Schema parsed':log.parsed===0?'Invalid response / abstention':'Awaiting parse'}</p><h3>Visible output</h3><pre>{typeof log.response?.text==='string'?log.response.text:log.error??'Response pending.'}</pre><h3>Prompt</h3><pre>{log.prompt}</pre></details>):<p>No public records on this page yet.</p>}
      <div className="reward-pagination"><button disabled={offset===0||logLoading} onClick={()=>onPage(false)}>Newer</button><button disabled={logs?.next==null||logLoading} onClick={()=>onPage(true)}>Older</button></div>
    </section>
    <footer><p>Run: {status?.runId??'experiment-002-1-v1'} · <a href={`${API}/status`}>Machine-readable status</a>. Public controls are read-only. Experiment 002 remains preserved.</p>{status?.protocolHash&&<p className="compatibility21-hash">Protocol SHA-256: {status.protocolHash}</p>}</footer>
  </main>;
}
