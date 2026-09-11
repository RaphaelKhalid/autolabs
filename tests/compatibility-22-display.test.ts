import {describe,expect,it} from 'vitest';
import {validCompatibility22Status,validCategoryAnalysis,categoryLabel,observedCategoryLabel,referenceMean} from '../lib/compatibility-22-display';
const status={runId:'experiment-002-2-v1',version:'v2',model:'gpt-5.6-luna',status:'waiting',stage:'dev',reason:null,updatedAt:'2026-09-11T06:00:00Z',protocolHash:null,evaluationSealed:true,progress:{callsDone:0,callsTotal:4116,histories:[{split:'dev',done:0,total:20}]},budget:{spentUsd:0,reservedUsd:0,priorCommittedUsd:null,totalCommittedUsd:null,capUsd:40,calls:0},execution:{concurrency:8},active:[],etaSeconds:null,gate:null,isolation:null,recent:[]};
const row={templateId:'coin-0',domain:'coin',label:'mixed-or-insufficient',observedHistoryLabel:'observed-equivalence-with-witnesses',completePairs:64,eligiblePairs:64,meanGainMin:0,meanGainMax:0,distributionFreeIntervals:{minimum:[-.5,.5],maximum:[-.5,.5]},discoveredPreservingWitnessPairs:64,referenceCeilingPairs:64,pairs:[{qReference:1}]};
describe('Experiment 002.2 display',()=>{
  it('accepts unsettled waiting status without fabricating settled spend',()=>{
    expect(validCompatibility22Status(status)).toBe(true);
    expect(status.budget.priorCommittedUsd).toBeNull();
    expect(validCompatibility22Status({...status,budget:{}})).toBe(false);
    expect(validCompatibility22Status({...status,progress:{callsTotal:4116}})).toBe(false);
  });
  it('keeps unavailable or sealed analysis distinct from released results',()=>{
    expect(validCategoryAnalysis({available:false,sealed:true})).toBe(true);
    expect(validCategoryAnalysis({available:true,sealed:true,analysis:{templates:[row]}})).toBe(false);
    expect(validCategoryAnalysis({available:true,sealed:false,analysis:{templates:[row]}})).toBe(true);
    expect(validCategoryAnalysis({available:true,sealed:false,analysis:{templates:[{...row,meanGainMin:NaN}]}})).toBe(false);
  });
  it('does not turn an observed pattern into population support',()=>{
    expect(observedCategoryLabel(row.observedHistoryLabel)).toBe('Equivalence + witnesses');
    expect(categoryLabel(row.label)).toBe('Mixed / insufficient');
    expect(categoryLabel('population-conflict-direction-supported')).toBe('Conflict direction supported');
    expect(categoryLabel('unknown')).toBe('Unclassified');
  });
  it('does not silently remove missing reference histories from the average',()=>{
    expect(referenceMean([{qReference:.5},{qReference:1}])).toBe(.75);
    expect(referenceMean([{qReference:1},{qReference:null}])).toBeNull();
    expect(referenceMean([])).toBeNull();
  });
});
