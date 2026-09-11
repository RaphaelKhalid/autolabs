import {describe, expect, it} from 'vitest';
import {COMPATIBILITY21_API, compatibility21CallTitle, compatibility21Eta, validCompatibility21Status} from '../lib/compatibility-21-display';
import {experiments, finiteCompatibility, latestExperiment, rewardCompatibility} from '../lib/experiment-catalog';

const status = {runId:'experiment-002-1-v1',version:'finite-compatibility-1',model:'gpt-5.6-luna',status:'running',stage:'dev',reason:null,updatedAt:'2026-09-11T06:00:00Z',protocolHash:'abc',evaluationSealed:true,progress:{callsDone:24,callsTotal:3600,cases:[{split:'dev',done:8,total:240}]},budget:{spentUsd:.02,reservedUsd:.04,priorCommittedUsd:1.291626,totalCommittedUsd:1.351626,capUsd:40,calls:32},execution:{concurrency:8},ledger:{storageBytes:4096,softLimitBytes:268435456},active:[{id:'dev-c01/guided/0',started:123}],etaSeconds:5400,gate:null,isolation:{passed:true},recent:[]};
describe('Experiment 002.1 display',()=>{
  it('accepts public status and refuses incomplete/malformed responses',()=>{
    expect(validCompatibility21Status(status)).toBe(true);
    for(const candidate of [null,{}, {...status,budget:{}},{...status,progress:{callsDone:10}},{...status,etaSeconds:NaN},{...status,execution:{concurrency:-1}},{...status,active:[{}]},{...status,evaluationSealed:undefined}]) expect(validCompatibility21Status(candidate)).toBe(false);
  });
  it('uses runner ETA only while active, never a paused countdown',()=>{
    expect(compatibility21Eta({status:'running',etaSeconds:5400})).toBe('About 1 h 30 min remaining');
    expect(compatibility21Eta({status:'running',etaSeconds:70})).toBe('About 2 min remaining');
    expect(compatibility21Eta({status:'paused',etaSeconds:5400})).toContain('on hold');
    expect(compatibility21Eta({status:'complete',etaSeconds:5400})).toBe('Complete');
    expect(compatibility21Eta(null)).toContain('Waiting');
    expect(compatibility21Eta({status:'running',etaSeconds:null})).toContain('Estimating');
  });
  it('explains method labels without modifying raw IDs',()=>{
    expect(compatibility21CallTitle('dev-c01/guided/0')).toBe('Verifier-guided search · step 1');
    expect(compatibility21CallTitle('eval-c01/description/0')).toBe('Description-only judgment · step 1');
    expect(compatibility21CallTitle('future-record')).toBe('Research call');
  });
  it('keeps the completed predecessor and shared-cap successor distinct',()=>{
    expect(latestExperiment).toBe(finiteCompatibility);
    expect(rewardCompatibility.status).toBe('complete');
    expect(experiments).toContain(rewardCompatibility);
    expect(finiteCompatibility.runId).not.toBe(rewardCompatibility.runId);
    expect(COMPATIBILITY21_API).toContain('autolabs-compatibility-21.');
  });
});
