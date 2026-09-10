'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { ArrowDownToLine, ArrowUpFromLine, Check, ChevronRight, CircleHelp, Copy, FlaskConical, Grip, History, KeyRound, Pause, Play, Plus, Settings2, Trash2, X } from 'lucide-react';
import { AlienForm } from './autolabs-observatory';
import { starter, validateConfiguration, type Agent, type Configuration } from '@/selfhost/config';
import type { Run } from '@/selfhost/engine';

const forms = [
  {name:'Mira', color:'#ca6844', role:'Map approaches and propose testable hypotheses.'},
  {name:'Pip', color:'#759444', role:'Construct candidates and check constraints.'},
  {name:'Orum', color:'#8771bb', role:'Challenge assumptions and look for counterexamples.'},
  {name:'Solvi', color:'#36979d', role:'Look for transformations and alternative representations.'},
  {name:'Tess', color:'#b35d80', role:'Explore overlooked cases and compare competing approaches.'},
];
type Point = {x:number;y:number};
type Payload = {kind:'form';index:number}|{kind:'model';model:string}|{kind:'move';id:string};
type Summary = Pick<Run,'id'|'status'|'round'|'spentUsd'> & {title:string;provider:string};
const initial = () => ({...structuredClone(starter), agents:starter.agents.map((a,i)=>({...a,name:forms[i].name,color:forms[i].color,appearance:i}))});
const placement = (i:number):Point => ({x:[27,71,49,25,74,48,18,81][i],y:[35,38,66,72,74,25,55,56][i]});

export function ExperimentWorkbench({selfHosted=false}:{selfHosted?:boolean}) {
  const reducedMotion=useReducedMotion();
  const [config,setConfig]=useState<Configuration>(initial);
  const [selected,setSelected]=useState<string|null>(starter.agents[0].id);
  const [positions,setPositions]=useState<Record<string,Point>>({});
  const [tray,setTray]=useState<'forms'|'models'>('forms');
  const [query,setQuery]=useState('');
  const [token,setToken]=useState('');
  const [connected,setConnected]=useState(false);
  const [ready,setReady]=useState(false);
  const [models,setModels]=useState<{id:string;name:string}[]>([
    {id:'openai/gpt-4.1-nano',name:'OpenAI: GPT-4.1 Nano'},
    {id:'openai/gpt-4.1-mini',name:'OpenAI: GPT-4.1 Mini'},
    {id:'anthropic/claude-haiku-4.5',name:'Anthropic: Claude Haiku 4.5'},
  ]);
  const [panel,setPanel]=useState<'connection'|'protocol'|'runs'|'help'|null>(null);
  const [busy,setBusy]=useState(false);
  const [message,setMessage]=useState('');
  const [runs,setRuns]=useState<Summary[]>([]);
  const [run,setRun]=useState<(Run&{eventCount?:number})|null>(null);
  const [over,setOver]=useState(false);
  const canvas=useRef<HTMLDivElement>(null);
  const dialog=useRef<HTMLDialogElement>(null);
  const importFile=useRef<HTMLInputElement>(null);
  const agent=config.agents.find(a=>a.id===selected);
  const active=run?.status==='running';
  const change=<K extends keyof Configuration>(key:K,value:Configuration[K])=>setConfig(c=>({...c,[key]:value}));
  const patchAgent=(id:string,patch:Partial<Agent>)=>setConfig(c=>({...c,agents:c.agents.map(a=>a.id===id?{...a,...patch}:a)}));

  async function api(path:string,body?:unknown) {
    if(!selfHosted)throw new Error('Public creator: export your setup and run it on your own host.');
    const response=await fetch(`/api/selfhost/${path}`,{method:body===undefined?'GET':'POST',headers:{authorization:`Bearer ${token}`,'content-type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
    const value=await response.json();if(!response.ok)throw new Error(value.error??'Request failed.');return value;
  }
  async function action(work:()=>Promise<void>) {setBusy(true);setMessage('');try{await work();}catch(e){setMessage(e instanceof Error?e.message:'Request failed.');}finally{setBusy(false);}}
  function download(name:string,value:unknown) {const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  function addForm(index:number,point?:Point) {
    if(config.agents.length>=8){setMessage('Maximum eight agents per run.');return;}
    const form=forms[index];if(!form)return;
    const id=`agent-${crypto.randomUUID().slice(0,8)}`;
    change('agents',[...config.agents,{id,name:form.name,model:config.agents[0].model,instructions:form.role,color:form.color,appearance:index}]);
    if(point)setPositions(p=>({...p,[id]:point}));setSelected(id);setMessage(`${form.name} added.`);
  }
  function drag(event:React.DragEvent,payload:Payload) {event.dataTransfer.setData('application/x-autolabs',JSON.stringify(payload));event.dataTransfer.effectAllowed=payload.kind==='move'?'move':'copy';}
  function payload(event:React.DragEvent):Payload|null {
    try {
      const p=JSON.parse(event.dataTransfer.getData('application/x-autolabs'));
      if(!p||typeof p!=='object')return null;
      if(p.kind==='form'&&Number.isInteger(p.index)&&p.index>=0&&p.index<forms.length)return {kind:'form',index:p.index};
      if(p.kind==='move'&&typeof p.id==='string')return {kind:'move',id:p.id};
      if(p.kind==='model'&&typeof p.model==='string'&&p.model.length<=120)return {kind:'model',model:p.model};
      return null;
    }catch{return null;}
  }
  function drop(event:React.DragEvent) {
    event.preventDefault();setOver(false);const data=payload(event),rect=canvas.current?.getBoundingClientRect();if(!data||!rect)return;
    const point={x:Math.max(14,Math.min(86,(event.clientX-rect.left)/rect.width*100)),y:Math.max(23,Math.min(77,(event.clientY-rect.top)/rect.height*100))};
    if(data.kind==='form')addForm(data.index,point);
    else if(data.kind==='move'&&config.agents.some(a=>a.id===data.id))setPositions(p=>({...p,[data.id]:point}));
    else if(data.kind==='model')setMessage('Drop a model directly onto an agent.');
  }
  function assign(model:string,id=selected) {if(!id){setMessage('Select an agent first.');return;}patchAgent(id,{model});setMessage('Model assigned.');}
  function copyConfiguration(value:Configuration) {setConfig(structuredClone(value));setSelected(value.agents[0].id);setPositions({});setMessage('Configuration loaded. Existing runs are unchanged.');}

  useEffect(()=>{if(panel)dialog.current?.showModal();else dialog.current?.close();},[panel]);
  useEffect(()=>{
    if(!selfHosted||!connected)return;
    const controller=new AbortController();let pending=false;
    async function refresh(){if(pending)return;pending=true;try{
      const options={headers:{authorization:`Bearer ${token}`},signal:controller.signal};
      const r=await fetch('/api/selfhost/runs',options);if(!r.ok)throw new Error();setRuns((await r.json()).runs);
      if(run?.id){const detail=await fetch(`/api/selfhost/runs/${run.id}`,options);if(detail.ok)setRun(await detail.json());}
    }catch{if(!controller.signal.aborted)setMessage('Runner disconnected. Jobs may still be running.');}finally{pending=false;}}
    void refresh();const timer=setInterval(()=>void refresh(),5000);return()=>{controller.abort();clearInterval(timer);};
  },[selfHosted,connected,token,run?.id]);

  const available=models.filter(m=>`${m.name} ${m.id}`.toLowerCase().includes(query.toLowerCase())).slice(0,35);
  const selectedIndex=agent?.appearance??Math.max(0,config.agents.findIndex(a=>a.id===selected))%5;
  return <main className="workbench">
    <header className="wb-header">
      <a className="wb-brand" href="/">A<span>AUTOLABS</span></a><span className="wb-divider"/>
      <input aria-label="Experiment title" className="wb-title" value={config.title} onChange={e=>change('title',e.target.value)}/>
      <div className="wb-header-actions">{!selfHosted?<><span>Configuration only</span><a href="https://github.com/RaphaelKhalid/autolabs/tree/main/selfhost">Self-host</a></>:<><button onClick={()=>setPanel('runs')}><History size={17}/><span>Runs</span></button><button className={connected?'wb-connected':''} onClick={()=>setPanel('connection')}><KeyRound size={16}/>{connected?'Connected':'Connect runner'}</button></>}</div>
    </header>
    <div className="wb-body">
      <aside className="wb-library" aria-label="Component library">
        <div className="wb-tabs"><button aria-pressed={tray==='forms'} onClick={()=>setTray('forms')}>Agents</button><button aria-pressed={tray==='models'} onClick={()=>setTray('models')}>Models</button></div>
        <p className="wb-hint">{tray==='forms'?'Drag into the lab, or tap to add.':'Drop onto an agent, or tap to assign.'}</p>
        {tray==='forms'?<div className="wb-form-library">{forms.map((form,index)=><button key={form.name} className="wb-form-tile" draggable onDragStart={e=>drag(e,{kind:'form',index})} onClick={()=>addForm(index)} aria-label={`Add ${form.name}`}><AlienForm agent={{id:`library-${index}`,name:form.name,color:form.color}} index={index} meeting={false} compact/><span>{form.name}</span><Plus size={14}/></button>)}</div>:<div className="wb-model-library"><input aria-label="Search models" placeholder="Find a model…" value={query} onChange={e=>setQuery(e.target.value)}/>{selfHosted&&!connected&&<button onClick={()=>setPanel('connection')}>Connect to browse models</button>}{available.map(model=><button key={model.id} draggable onDragStart={e=>drag(e,{kind:'model',model:model.id})} onClick={()=>assign(model.id)} title={model.id}><Grip size={14}/><span>{model.name}<small>{model.id.split('/')[0]}</small></span></button>)}{!available.length&&<p>No matches. Enter a model ID in the inspector.</p>}</div>}
        <div className="wb-library-bottom"><button onClick={()=>importFile.current?.click()}><ArrowUpFromLine size={16}/>Import</button><button onClick={()=>void action(async()=>download('autolabs-experiment.json',validateConfiguration(config)))}><ArrowDownToLine size={16}/>Export</button><input ref={importFile} hidden type="file" accept=".json,application/json" onChange={e=>{const file=e.target.files?.[0];if(file)void action(async()=>{if(file.size>64000)throw new Error('Maximum configuration size: 64 KB.');copyConfiguration(validateConfiguration(JSON.parse(await file.text())));});e.target.value='';}}/></div>
      </aside>

      <section className="wb-workspace" aria-label="Lab workspace">
        <div className="wb-workspace-top"><span>{active?`ROUND ${run.round} / ${run.phase.toUpperCase()}`:'EXPERIMENT DESIGN'}</span><button onClick={()=>setPanel('help')} aria-label="How this lab works"><CircleHelp size={18}/></button></div>
        <div ref={canvas} className={`wb-canvas ${over?'wb-drop-active':''}`} onDragOver={e=>{e.preventDefault();setOver(true);}} onDragLeave={e=>{if(!e.currentTarget.contains(e.relatedTarget as Node))setOver(false);}} onDrop={drop} aria-label="Drop agents here">
          <div className="wb-orbit wb-orbit-one"/><div className="wb-orbit wb-orbit-two"/>
          <div className="wb-center"><span>{String(config.agents.length).padStart(2,'0')}</span><small>AGENTS</small></div>
          {config.agents.map((a,index)=>{const point=positions[a.id]??placement(index);const current=active&&run.pending?.agentId===a.id;return <motion.div key={a.id} className={`wb-agent ${selected===a.id?'is-selected':''} ${current?'is-working':''}`} animate={{left:`${point.x}%`,top:`${point.y}%`}} transition={reducedMotion?{duration:0}:{type:'spring',stiffness:150,damping:24}}>
            <button className="wb-agent-handle" draggable onDragStart={e=>drag(e,{kind:'move',id:a.id})} onClick={()=>setSelected(a.id)} onKeyDown={e=>{const shifts:Record<string,Point>={ArrowLeft:{x:-3,y:0},ArrowRight:{x:3,y:0},ArrowUp:{x:0,y:-3},ArrowDown:{x:0,y:3}};if(shifts[e.key]){e.preventDefault();const s=shifts[e.key];setPositions(p=>({...p,[a.id]:{x:Math.max(14,Math.min(86,point.x+s.x)),y:Math.max(23,Math.min(77,point.y+s.y))}}));}}} onDragOver={e=>e.preventDefault()} onDrop={e=>{const data=payload(e);if(data?.kind==='model'){e.preventDefault();e.stopPropagation();setOver(false);assign(data.model,a.id);}}} aria-label={`Select ${a.name}`} aria-pressed={selected===a.id} title="Drag to arrange. Arrow keys also move this agent."><AlienForm agent={a} index={a.appearance??index%5} meeting={active&&run.phase==='meeting'}/><span className="wb-agent-name">{a.name}</span><span className="wb-agent-model">{config.provider==='mock'?'Mock provider':a.model.split('/').slice(-1)[0]}</span>{current&&<span className="wb-working-label">Working</span>}</button>
          </motion.div>;})}
          <span className="wb-canvas-caption">{over?'Release to place':'Drag to arrange · select to configure'}</span>
        </div>
        <div className="wb-protocol"><span className="wb-protocol-step"><FlaskConical size={17}/>Research</span><ChevronRight size={16}/><button aria-pressed={config.meetings} onClick={()=>change('meetings',!config.meetings)}>{config.meetings?'Discussion on':'Discussion off'}</button><ChevronRight size={16}/><button onClick={()=>setPanel('protocol')}>{config.rounds} rounds <Settings2 size={15}/></button><span className="wb-protocol-end">{config.phaseSeconds}s interval</span></div>
        <div className="wb-status" role="status" aria-live="polite">{busy?'Working…':message||(selfHosted?'Draft · no run started by this editor':'Public creator · configure and export; no jobs run here')}</div>
      </section>

      <aside className="wb-inspector" aria-label="Inspector">
        <div className="wb-inspector-title"><span>INSPECTOR</span>{agent&&<button disabled={config.agents.length===1} aria-label="Remove selected agent" onClick={()=>{const remaining=config.agents.filter(a=>a.id!==agent.id);change('agents',remaining);setSelected(remaining[0].id);}}><Trash2 size={16}/></button>}</div>
        {agent&&<><div className="wb-inspector-art"><AlienForm agent={{...agent,id:`inspector-${agent.id}`}} index={selectedIndex} meeting={false}/></div><label>Name<input value={agent.name} onChange={e=>patchAgent(agent.id,{name:e.target.value})}/></label><label>Model<input list="wb-models" value={agent.model} onChange={e=>patchAgent(agent.id,{model:e.target.value})}/></label><datalist id="wb-models">{models.map(m=><option key={m.id} value={m.id}>{m.name}</option>)}</datalist><div className="wb-appearance">{forms.map((form,index)=><button key={form.name} aria-label={`Use ${form.name} form`} aria-pressed={selectedIndex===index} style={{background:form.color}} onClick={()=>patchAgent(agent.id,{appearance:index,color:form.color})}>{selectedIndex===index&&<Check size={14}/>}</button>)}</div><details><summary>Instructions</summary><textarea aria-label="Agent instructions" rows={4} value={agent.instructions} onChange={e=>patchAgent(agent.id,{instructions:e.target.value})}/></details></>}
        <div className="wb-task"><h2>Experiment</h2><label>Evaluator<select value={config.template} onChange={e=>change('template',e.target.value as Configuration['template'])}><option value="integer-search-v1">Integer search · systems test</option><option value="research-notes-v1">Research notes · human review</option></select></label><details><summary>Objective</summary><textarea aria-label="Objective" rows={4} value={config.objective} onChange={e=>change('objective',e.target.value)}/></details><label>Connection<select value={config.provider} onChange={e=>change('provider',e.target.value as Configuration['provider'])}><option value="mock">Mock · no API spend</option><option value="openrouter">OpenRouter · paid</option></select></label></div>
      </aside>
    </div>

    <footer className="wb-launchbar"><button onClick={()=>setPanel('protocol')} className="wb-budget"><small>RUN BUDGET</small><strong>${config.budgetUsd.toFixed(2)}</strong></button><div className="wb-launch-summary"><strong>{config.provider==='mock'?'Mock run':'OpenRouter run'}</strong><span>{config.agents.length} agents · {config.rounds} rounds{run?` · $${run.spentUsd.toFixed(4)} recorded`:''}</span></div><button className="wb-settings" onClick={()=>setPanel('protocol')} aria-label="Run settings"><Settings2 size={20}/></button><button className="wb-launch" disabled={busy||(selfHosted&&(!connected||(config.provider==='openrouter'&&!ready)||active))} onClick={()=>void action(async()=>{if(!selfHosted){download('autolabs-experiment.json',validateConfiguration(config));setMessage('Setup exported. Run it on your own host.');return;}setRun(await api('runs',validateConfiguration(config)));setRuns((await api('runs')).runs);setMessage(config.provider==='mock'?'Mock run started. No API spend.':'Paid run started. View its record under Runs.');})}><Play size={17} fill="currentColor"/>{!selfHosted?'Export setup':config.provider==='mock'?'Start mock':'Start paid run'}</button></footer>

    <dialog ref={dialog} className="wb-dialog" aria-labelledby="wb-panel-title" onCancel={()=>setPanel(null)} onClose={()=>setPanel(null)}><header><h2 id="wb-panel-title">{panel==='connection'?'Runner connection':panel==='protocol'?'Run settings':panel==='runs'?'Experiment records':'Lab guide'}</h2><button aria-label="Close panel" onClick={()=>setPanel(null)}><X size={20}/></button></header>
      {selfHosted&&panel==='connection'&&<><p>Your provider key stays on the runner.</p><label>Owner token<input autoComplete="off" type="password" value={token} onChange={e=>{setToken(e.target.value);setConnected(false);}}/></label><button className="wb-primary" disabled={busy||!token} onClick={()=>void action(async()=>{const data=await api('connections');setModels(data.models);setReady(data.openrouterConfigured);setRuns((await api('runs')).runs);setConnected(true);setMessage('Runner connected.');setPanel(null);})}>Connect</button>{connected&&<button disabled={busy||!ready} onClick={()=>void action(async()=>{const check=await api('check',{model:agent?.model??config.agents[0].model});setMessage(`Connection check ${check.ok?'passed':'returned no text'} · $${check.costUsd.toFixed(7)} recorded`);})}>Test selected model · ≤ $0.01 reservation</button>}<p className="wb-fine">Use AUTOLABS_LOCAL_TOKEN from your local environment file, not your provider key.</p></>}
      {panel==='protocol'&&<><div className="wb-settings-grid">{([{key:'rounds',label:'Rounds',min:1,max:100},{key:'phaseSeconds',label:'Interval (seconds)',min:0,max:3600},{key:'budgetUsd',label:'Budget (USD)',min:.01,max:50},{key:'maxOutputTokens',label:'Output token limit',min:64,max:4096}] as const).map(f=><label key={f.key}>{f.label}<input type="number" min={f.min} max={f.max} step={f.key==='budgetUsd'?.01:1} value={config[f.key]} onChange={e=>change(f.key,Number(e.target.value))}/></label>)}</div><label className="wb-check"><input type="checkbox" checked={config.meetings} onChange={e=>change('meetings',e.target.checked)}/>Discussion after research</label><label className="wb-check"><input type="checkbox" checked={config.stopOnSuccess} onChange={e=>change('stopOnSuccess',e.target.checked)}/>Stop at verified template success</label><p className="wb-fine">One request per agent per phase, executed sequentially. The interval is a pause between phases. Cost reservations are estimates; set a provider-side limit too.</p></>}
      {selfHosted&&panel==='runs'&&<><div className="wb-record-list">{runs.map(item=><button key={item.id} onClick={()=>void action(async()=>setRun(await api(`runs/${item.id}`)))}><strong>{item.title}</strong><span>{item.status} · R{item.round} · ${item.spentUsd.toFixed(4)}</span></button>)}{!runs.length&&<p>{connected?'No saved runs.':'Connect your runner to load records.'}</p>}</div>{run&&<section className="wb-record"><h3>{run.config.title}</h3><p>{run.status} · round {run.round} · {run.phase}</p><div className="wb-record-actions"><button disabled={busy||!['running','paused'].includes(run.status)} onClick={()=>void action(async()=>setRun(await api(`runs/${run.id}/${run.status==='paused'?'resume':'pause'}`,{})))}><Pause size={15}/>{run.status==='paused'?'Resume':'Pause'}</button><button onClick={()=>void action(async()=>download(`${run.id}.json`,await api(`runs/${run.id}/export`)))}><ArrowDownToLine size={15}/>Export run</button><button onClick={()=>{copyConfiguration(run.config);setPanel(null);}}><Copy size={15}/>Use setup</button></div><p className="wb-fine">Latest {run.events.length} of {run.eventCount??run.events.length} events. Full record in export.</p>{run.events.slice().reverse().map(e=><details key={e.seq}><summary><small>R{e.round} · {e.agentId??'runner'} · {e.kind}</small>{e.summary}</summary>{e.data!=null&&<pre>{JSON.stringify(e.data,null,2)}</pre>}</details>)}</section>}</>}
      {panel==='help'&&<><p>Drag a form into the lab. Select it to set a model and instructions. Drop a model onto an agent to assign it.</p><p>On a phone, tap a form to add it and a model to assign it. Keyboard users can select an agent and arrange it with arrow keys. Positions are visual only and are not exported.</p><p>Integer search is a known-answer test: minimize (x−17)²+(y+9)² for integer coordinates in [−100,100]. Changing the objective text does not change its evaluator.</p><p>Research notes require human review. Neither starter has web access or code execution. Forms identify agents; they do not change model capabilities.</p><p>Edits affect the next launch, never an existing run. Records include assistant output, not hidden reasoning.</p><a href="/experiments/erdos-885">View the Erdős pilot archive</a></>}
      <div role="status" aria-live="polite">{busy?'Working…':message}</div>
    </dialog>
  </main>;
}
