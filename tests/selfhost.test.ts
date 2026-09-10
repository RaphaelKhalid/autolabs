import { afterEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { starter, validateConfiguration, evaluate } from '../selfhost/config';
import { Engine } from '../selfhost/engine';
import { complete, parseReport, reserveCost } from '../selfhost/provider';
const directories:string[]=[];
function engine(){const dir=mkdtempSync(join(tmpdir(),'autolabs-test-'));directories.push(dir);return new Engine(dir);}
const configuration=()=>({...structuredClone(starter),phaseSeconds:0});
afterEach(()=>{for(const dir of directories.splice(0))rmSync(dir,{recursive:true,force:true});vi.restoreAllMocks();});
describe('portable configuration and evaluator',()=>{
  it('preserves valid agent forms in exports and rejects unsupported forms',()=>{
    const c=configuration();c.agents[0].appearance=4;
    expect(validateConfiguration(c).agents[0].appearance).toBe(4);
    c.agents[0].appearance=5;expect(()=>validateConfiguration(c)).toThrow('Form');
  });
  it('validates and strips unrecognized configuration properties',()=>{expect(validateConfiguration({...configuration(),apiKey:'not exported'})).not.toHaveProperty('apiKey');});
  it('rejects secrets, duplicate identities and unbounded parameters',()=>{
    expect(()=>validateConfiguration({...configuration(),objective:'sk-or-v1-123456789012345678901234'})).toThrow('API keys');
    expect(()=>validateConfiguration({...configuration(),agents:[starter.agents[0],starter.agents[0]]})).toThrow('unique');
    expect(()=>validateConfiguration({...configuration(),budgetUsd:Infinity})).toThrow();
    expect(()=>validateConfiguration({...configuration(),provider:'unknown'})).toThrow();
  });
  it('checks the installed objective without treating research notes as verified',()=>{
    expect(evaluate('integer-search-v1',{x:17,y:-9})).toMatchObject({score:0,success:true});
    expect(evaluate('integer-search-v1',{x:17.5,y:-9}).status).toBe('invalid');
    expect(evaluate('research-notes-v1',{claim:'Solved everything'})).toMatchObject({status:'human-review',success:false,score:null});
  });
});
describe('runner checkpoints and budgets',()=>{
  it('runs a no-cost verified mock and reloads its immutable configuration',async()=>{
    const e=engine(),c=configuration(),r=e.create(c);c.title='Changed later';await e.tick([]);
    expect(r.config.title).not.toBe(c.title);expect(r.status).toBe('verified-success');expect(r.spentUsd).toBe(0);
    expect(new Engine(e.directory).runs.get(r.id)?.configHash).toBe(r.configHash);
  });
  it('completes configured research and discussion phases',async()=>{
    const e=engine(),r=e.create({...configuration(),stopOnSuccess:false});for(let i=0;i<8;i++)await e.tick([]);
    expect(r.status).toBe('complete');expect(r.events.filter(e=>e.kind==='research')).toHaveLength(4);expect(r.events.filter(e=>e.kind==='meeting')).toHaveLength(4);
  });
  it('stops before unaffordable calls',async()=>{
    const e=engine(),r=e.create({...configuration(),provider:'openrouter'}),call=vi.fn();
    await e.tick([{id:r.config.agents[0].model,name:'Expensive',price:{input:1,output:1,request:0}}],'test',call);
    expect(call).not.toHaveBeenCalled();expect(r.status).toBe('budget-stop');
  });
  it('keeps reservations on failures and does not retry automatically',async()=>{
    const e=engine(),r=e.create({...configuration(),provider:'openrouter'}),call=vi.fn().mockRejectedValue(new Error('timeout'));
    const catalogue=[{id:r.config.agents[0].model,name:'Test',price:{input:0.0000001,output:0.0000002,request:0}}];
    await e.tick(catalogue,'test',call);const charged=r.spentUsd;await e.tick(catalogue,'test',call);
    expect(r.status).toBe('paused');expect(charged).toBeGreaterThan(0);expect(call).toHaveBeenCalledTimes(1);
    e.control(r.id,'resume');await e.tick(catalogue,'test',call);expect(call).toHaveBeenCalledTimes(2);
  });
  it('pauses an ambiguous in-flight request on restart without charging twice',()=>{
    const e=engine(),r=e.create(configuration());r.pending={agentId:'agent-1',reservedUsd:.01};r.spentUsd=.01;e.save(r);
    const recovered=new Engine(e.directory).runs.get(r.id)!;expect(recovered.status).toBe('paused');expect(recovered.spentUsd).toBe(.01);
  });
  it('does not leak current peer reports into independent prompts',async()=>{
    const e=engine(),r=e.create({...configuration(),stopOnSuccess:false});const call=vi.fn(complete);
    await e.tick([],undefined,call);await e.tick([],undefined,call);
    expect(call.mock.calls[1][2][1].content).not.toContain('Mock response');expect(r.agentIndex).toBe(0);
  });
  it('rejects a second active run and supports explicit pause/resume',()=>{const e=engine(),r=e.create(configuration());expect(()=>e.create(configuration())).toThrow('Pause');e.control(r.id,'pause');expect(r.status).toBe('paused');e.control(r.id,'resume');expect(r.status).toBe('running');});
});
describe('provider boundary',()=>{
  it('calculates a positive conservative request reservation',()=>{expect(reserveCost([{role:'user',content:'Test'}],100,{input:.000001,output:.000002,request:0})).toBeGreaterThan(.004);});
  it('rejects malformed reports and accepts fenced JSON',()=>{expect(()=>parseReport('done')).toThrow();expect(parseReport('```json\n{"summary":"Test","proposals":[]}\n```').summary).toBe('Test');});
  it('does not retain hidden reasoning and accounts for reported cost',async()=>{
    vi.spyOn(globalThis,'fetch').mockResolvedValue(new Response(JSON.stringify({id:'test',model:'model',choices:[{message:{content:'OK',reasoning:'private'}}],usage:{cost:.001,prompt_tokens:4,completion_tokens:1}})));
    const result=await complete({...configuration(),provider:'openrouter'},starter.agents[0],[{role:'user',content:'OK'}],.01,'test-key');
    expect(result.costUsd).toBe(.001);expect(result.costBasis).toBe('reported');expect(JSON.stringify(result)).not.toContain('private');
  });
});
