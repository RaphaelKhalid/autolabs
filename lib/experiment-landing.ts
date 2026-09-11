import {latestExperiment} from './experiment-catalog';
import {COMPATIBILITY22_API} from './compatibility-22-display';

/** Public status only. A waiting, failed, unknown or unavailable successor never replaces the active page. */
export async function currentExperimentPath():Promise<string>{
  const fallback=`/experiments/${latestExperiment.slug}`;
  try{
    const response=await fetch(`${COMPATIBILITY22_API}/status`,{next:{revalidate:15},signal:AbortSignal.timeout(2500)});
    if(!response.ok)return fallback;
    const value:unknown=await response.json();
    if(!value||typeof value!=='object')return fallback;
    const status=value as Record<string,unknown>;
    if(status.runId==='experiment-002-2-v1'&&['running','complete'].includes(String(status.status))&&typeof status.startedAt==='string'&&Number.isFinite(Date.parse(status.startedAt))&&typeof status.protocolHash==='string'&&/^[a-f0-9]{64}$/.test(status.protocolHash))return '/experiments/reward-categories-22';
  }catch{/* Keep the currently published experiment available during cold starts/outages. */}
  return fallback;
}
