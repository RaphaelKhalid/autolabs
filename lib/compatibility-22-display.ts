import {validCompatibility21Status, type Compatibility21Status} from './compatibility-21-display';
export const COMPATIBILITY22_API='https://autolabs-reward-categories-22.raphaelbahadurkhan.workers.dev';
export interface Compatibility22Status extends Omit<Compatibility21Status,'status'|'progress'|'budget'|'ledger'> {
  status:'ready'|'waiting'|'running'|'paused'|'complete';
  progress:{callsDone:number;callsTotal:number;histories:{split:string;done:number;total:number}[]};
  budget:Omit<Compatibility21Status['budget'],'priorCommittedUsd'|'totalCommittedUsd'>&{priorCommittedUsd:number|null;totalCommittedUsd:number|null};
}
export function validCompatibility22Status(value:unknown):value is Compatibility22Status {
  if(!value||typeof value!=='object')return false;
  const candidate=value as Partial<Compatibility22Status>;
  if(!candidate.progress||!candidate.budget)return false;
  return validCompatibility21Status({...candidate,status:candidate.status==='waiting'?'ready':candidate.status,
    progress:{...candidate.progress,cases:candidate.progress.histories},budget:{...candidate.budget,priorCommittedUsd:candidate.budget.priorCommittedUsd===null?0:candidate.budget.priorCommittedUsd,totalCommittedUsd:candidate.budget.totalCommittedUsd===null?0:candidate.budget.totalCommittedUsd},ledger:{storageBytes:0,softLimitBytes:0}});
}
export interface CategoryResult {
  templateId:string;domain:string;label:string;observedHistoryLabel:string;completePairs:number;eligiblePairs:number;
  meanGainMin:number|null;meanGainMax:number|null;
  distributionFreeIntervals:{minimum:[number,number];maximum:[number,number]}|null;
  discoveredPreservingWitnessPairs:number;referenceCeilingPairs:number;
  pairs:{qReference:number|null}[];
}
export interface CategoryAnalysis {available:boolean;sealed:boolean;analysis?:{templates:CategoryResult[];scope?:string};}
export function validCategoryAnalysis(value:unknown):value is CategoryAnalysis {
  if(!value||typeof value!=='object')return false;
  const candidate=value as Partial<CategoryAnalysis>;
  if(typeof candidate.available!=='boolean'||typeof candidate.sealed!=='boolean')return false;
  if(!candidate.available)return true;
  const finiteOrNull=(n:unknown)=>n===null||(typeof n==='number'&&Number.isFinite(n));
  const bound=(v:unknown)=>Array.isArray(v)&&v.length===2&&v.every(x=>typeof x==='number'&&Number.isFinite(x)&&x>=-1&&x<=1)&&v[0]<=v[1];
  return candidate.sealed===false&&Array.isArray(candidate.analysis?.templates)&&candidate.analysis.templates.length<=8&&candidate.analysis.templates.every(row=>row&&typeof row.templateId==='string'&&typeof row.domain==='string'&&typeof row.label==='string'&&typeof row.observedHistoryLabel==='string'&&[row.completePairs,row.eligiblePairs,row.discoveredPreservingWitnessPairs,row.referenceCeilingPairs].every(n=>Number.isSafeInteger(n)&&n>=0&&n<=64)&&finiteOrNull(row.meanGainMin)&&finiteOrNull(row.meanGainMax)&&(row.distributionFreeIntervals===null||(row.distributionFreeIntervals&&bound(row.distributionFreeIntervals.minimum)&&bound(row.distributionFreeIntervals.maximum)))&&Array.isArray(row.pairs)&&row.pairs.length<=64&&row.pairs.every(pair=>pair&&finiteOrNull(pair.qReference)));
}
export function categoryLabel(label:string):string {
  return ({'population-aligned-direction-supported':'Aligned direction supported','population-conflict-direction-supported':'Conflict direction supported','population-outcome-equivalence-with-observed-witnesses':'Outcome equivalence supported','mixed-or-insufficient':'Mixed / insufficient'} as Record<string,string>)[label]??'Unclassified';
}
export function observedCategoryLabel(label:string):string {
  return ({'observed-aligned-direction':'Aligned direction','observed-conflict-direction':'Conflict direction','observed-equivalence-with-witnesses':'Equivalence + witnesses','mixed-or-insufficient':'Mixed / insufficient'} as Record<string,string>)[label]??'Unclassified';
}
export function referenceMean(pairs:{qReference:number|null}[]):number|null {
  if(!pairs.length||pairs.some(pair=>pair.qReference===null))return null;
  return pairs.reduce((sum,pair)=>sum+pair.qReference!,0)/pairs.length;
}
