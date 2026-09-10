import { createHash, randomUUID } from 'node:crypto';
import { mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { evaluate, validateConfiguration } from './config.ts';
import type { Configuration, Evaluation } from './config.ts';
import { complete, parseReport, reserveCost } from './provider.ts';
import type { Completion, Message, Model } from './provider.ts';

export type Event = {seq:number;at:string;round:number;phase:string;agentId?:string;kind:string;summary:string;data?:unknown};
export type Run = {id:string;createdAt:string;config:Configuration;configHash:string;status:'running'|'paused'|'complete'|'budget-stop'|'verified-success';round:number;phase:'research'|'meeting';agentIndex:number;nextAt:number;spentUsd:number;pending:null|{agentId:string;reservedUsd:number};events:Event[];reports:Record<string,{summary:string;proposals:unknown[]}>;best:Evaluation|null};
export class Engine {
  directory:string; runs=new Map<string,Run>(); busy=false;
  constructor(directory:string) {
    this.directory=directory;mkdirSync(directory,{recursive:true,mode:0o700});
    for(const name of readdirSync(directory).filter(n=>/^run-[\w-]+\.json$/.test(n))) {
      const run=JSON.parse(readFileSync(join(directory,name),'utf8')) as Run;
      if(run.pending) {run.status='paused';run.pending=null;this.event(run,'recovery','Runner stopped during a request. Its full reservation remains charged. Review before resuming.');}
      this.runs.set(run.id,run);
    }
  }
  save(run:Run) {const file=join(this.directory,`${run.id}.json`);writeFileSync(`${file}.tmp`,JSON.stringify(run),{mode:0o600});renameSync(`${file}.tmp`,file);}
  event(run:Run,kind:string,summary:string,data?:unknown,agentId?:string) {run.events.push({seq:run.events.length+1,at:new Date().toISOString(),round:run.round,phase:run.phase,agentId,kind,summary,data});this.save(run);}
  create(value:unknown):Run {
    if([...this.runs.values()].some(r=>r.status==='running')) throw new Error('Pause the active run before starting another.');
    const config=validateConfiguration(value),id=`run-${randomUUID()}`;
    const run:Run={id,createdAt:new Date().toISOString(),config,configHash:createHash('sha256').update(JSON.stringify(config)).digest('hex'),status:'running',round:1,phase:'research',agentIndex:0,nextAt:0,spentUsd:0,pending:null,events:[],reports:{},best:null};
    this.runs.set(id,run);this.event(run,'start',config.provider==='mock'?'Mock run started. No model calls or research claims.':'Experiment started.',{config,configHash:run.configHash});return run;
  }
  control(id:string,action:'pause'|'resume') {
    const run=this.runs.get(id);if(!run) throw new Error('Run not found.');
    if(action==='pause'&&run.status==='running') {run.status='paused';this.event(run,'control','Owner paused the run. Any in-flight request may finish.');}
    else if(action==='resume'&&run.status==='paused') {
      if([...this.runs.values()].some(r=>r.status==='running')) throw new Error('Another run is active.');
      run.status='running';this.event(run,'control','Owner resumed the run.');
    } else throw new Error('This action is not available for the run status.');
    return run;
  }
  async tick(catalogue:Model[],apiKey?:string,call:typeof complete=complete) {
    if(this.busy)return;this.busy=true;
    try {
      const run=[...this.runs.values()].find(r=>r.status==='running');if(!run||run.nextAt>Date.now())return;
      const agent=run.config.agents[run.agentIndex];
      const task=run.config.template==='integer-search-v1'
        ? 'Evaluator: minimize (x-17)^2+(y+9)^2 for integer x,y in [-100,100]. Proposals are objects with numeric x,y. This known-answer test is not a research discovery.'
        : 'Write evidence-oriented research notes. No web access or computation tools are available in this template. Do not invent sources or claim verification. Proposals are research notes for human review.';
      const context=run.phase==='meeting'?JSON.stringify(run.reports):JSON.stringify({ownPrevious:run.reports[agent.id]??null});
      const messages:Message[]=[{role:'system',content:`${agent.instructions}\nReturn JSON only: {"summary":"public research summary","proposals":[]}. No hidden reasoning. ${task}`},{role:'user',content:`Objective: ${run.config.objective}\nRound ${run.round}, ${run.phase}. ${run.phase==='meeting'?'Discuss the revealed research reports.':'Work independently; peer reports for this round are sealed.'}\n${context.slice(0,24000)}`}];
      const model=catalogue.find(m=>m.id===agent.model);
      if(run.config.provider==='openrouter'&&!model) {run.status='paused';this.event(run,'error','Configured model is unavailable in the provider catalogue. No substitution made.');return;}
      const reserve=run.config.provider==='mock'?0:reserveCost(messages,run.config.maxOutputTokens,model!.price);
      if(run.spentUsd+reserve>run.config.budgetUsd) {run.status='budget-stop';this.event(run,'budget','Stopped before the next request: conservative reservation exceeds remaining budget.');return;}
      run.pending={agentId:agent.id,reservedUsd:reserve};run.spentUsd+=reserve;
      this.event(run,'request',`Request started for ${agent.name}.`,{reservedUsd:reserve,model:agent.model},agent.id);
      let response:Completion;
      try {response=await call(run.config,agent,messages,reserve,apiKey);}
      catch {run.pending=null;run.status='paused';this.event(run,'error','Model request failed or timed out. Reservation retained; no automatic retry. Resuming authorizes a new attempt.',undefined,agent.id);return;}
      run.spentUsd+=response.costUsd-reserve;
      this.event(run,'usage','Provider response received.',{model:response.model,responseId:response.responseId,costUsd:response.costUsd,costBasis:response.costBasis,inputTokens:response.inputTokens,outputTokens:response.outputTokens},agent.id);
      try {
        const report=parseReport(response.text);run.reports[agent.id]=report;
        const checks=run.phase==='research'?report.proposals.map(p=>({proposal:p,evaluation:evaluate(run.config.template,p)})):[];
        for(const check of checks) if(check.evaluation.score!==null&&(!run.best||run.best.score===null||check.evaluation.score<run.best.score))run.best=check.evaluation;
        this.event(run,run.phase,report.summary,{report,checks},agent.id);
        if(run.config.stopOnSuccess&&checks.some(c=>c.evaluation.success)) {run.pending=null;run.status='verified-success';this.event(run,'complete','Template success criterion verified. This does not establish research novelty.');return;}
      } catch {this.event(run,'invalid-output','Response failed the report schema; no score assigned.',{publicOutput:response.text.slice(0,16000)},agent.id);}
      run.agentIndex+=1;
      if(run.spentUsd>=run.config.budgetUsd) {run.pending=null;run.status='budget-stop';this.event(run,'budget','Recorded cost reached the budget. No further requests will be issued.');return;}
      this.advance(run);
    } finally {this.busy=false;}
  }
  advance(run:Run) {
    run.pending=null;
    if(run.agentIndex>=run.config.agents.length) {
      run.agentIndex=0;
      if(run.phase==='research'&&run.config.meetings)run.phase='meeting';
      else {run.phase='research';run.round+=1;}
      if(run.round>run.config.rounds) {run.round=run.config.rounds;run.status='complete';this.event(run,'complete','Configured rounds completed.');return;}
      run.nextAt=Date.now()+run.config.phaseSeconds*1000;
      this.event(run,'phase',`Next phase: round ${run.round}, ${run.phase}.`);
    } else this.save(run);
  }
}
