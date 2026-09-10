interface TimingEvent {id:number;time:string;type:string;data:Record<string,unknown>}
interface TimingStatus {status:string;stage:string;progress:{done:number;total:number};recent:TimingEvent[]}

/** Display-only recent-pace projection. The range is heuristic, not a confidence interval. */
export function rewardEta(status:TimingStatus|null,now:number){
  if(!status)return {label:'Calculating…'};
  if(status.status==='complete')return {label:'Complete'};
  if(status.status==='paused')return {label:'Paused — ETA on resume'};
  if(status.status!=='running')return {label:'Available once running'};
  const starts=new Map<string,number>(),durations:number[]=[];let lastComplete=0;
  for(const event of [...status.recent].sort((a,b)=>a.id-b.id)){
    if(['paused','execution_revision','main_protocol_frozen'].includes(event.type)){starts.clear();durations.length=0;lastComplete=0;continue;}
    const id=event.data.id,time=Date.parse(event.time);
    if(typeof id!=='string'||!Number.isFinite(time)||(status.stage==='main'&&id.startsWith('gate-')))continue;
    if(event.type==='unit_started')starts.set(id,time);
    if(event.type==='unit_complete'){
      const start=starts.get(id);starts.delete(id);
      // Reject interrupted, out-of-window or malformed timings. Include normal checkpoint overhead.
      if(start!==undefined&&time>start&&time-start<600000&&event.data.phase!=='report'){
        durations.push(time-start+1500);lastComplete=time;
      }
    }
  }
  if(durations.length<3)return {label:'Calibrating recent pace…'};
  if(now-lastComplete>600000)return {label:'Waiting for a fresh checkpoint…'};
  const remaining=Math.max(0,status.progress.total-status.progress.done);
  if(!remaining)return {label:'Finalizing…'};
  const mean=durations.reduce((a,b)=>a+b,0)/durations.length;
  const estimate=Math.max(mean,remaining*mean-Math.max(0,now-lastComplete));
  // Wide bounds acknowledge API latency and different work in subsequent phases.
  const lowMinutes=Math.max(1,Math.floor(estimate*.65/300000)*5);
  const highMinutes=Math.max(lowMinutes+1,Math.ceil(estimate*1.6/300000)*5);
  return {label:`About ${duration(lowMinutes)}–${duration(highMinutes)} remaining`,
    earliest:now+lowMinutes*60000,latest:now+highMinutes*60000,samples:durations.length};
}
function duration(minutes:number){
  if(minutes<60)return `${minutes} min`;
  const h=Math.floor(minutes/60),m=minutes%60;return `${h}h${m?` ${m}m`:''}`;
}
