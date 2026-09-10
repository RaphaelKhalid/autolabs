import {expect,it} from 'vitest';
import {rewardEta} from '../lib/reward-eta';
const start=Date.UTC(2026,8,10,20);
const recent=Array.from({length:4},(_,i)=>[
  {id:i*2,time:new Date(start+i*20000).toISOString(),type:'unit_started',data:{id:`train-${i}`,phase:'train'}},
  {id:i*2+1,time:new Date(start+i*20000+18000).toISOString(),type:'unit_complete',data:{id:`train-${i}`,phase:'train'}},
]).flat();
const status={status:'running',stage:'main',progress:{done:100,total:716},recent};
it('estimates from matched completed units regardless of event ordering',()=>{
  const eta=rewardEta(status,start+80000);
  expect(eta.samples).toBe(4);expect(eta.label).toContain('remaining');expect(eta.earliest).toBeGreaterThan(start+80000);expect(eta.latest).toBeGreaterThan(eta.earliest!);
  expect(rewardEta({...status,recent:[...recent].reverse()},start+80000)).toEqual(eta);
});
it('suppresses estimates when paused, complete, stale or insufficiently sampled',()=>{
  expect(rewardEta({...status,status:'paused'},start).label).toContain('Paused');
  expect(rewardEta({...status,status:'complete'},start).label).toBe('Complete');
  expect(rewardEta(status,start+900000).label).toContain('fresh checkpoint');
  expect(rewardEta({...status,recent:recent.slice(0,4)},start).label).toContain('Calibrating');
});
it('does not count paused time or timings from before a concurrency change',()=>{
  const interrupted=[...recent,{id:8,time:new Date(start+90000).toISOString(),type:'paused',data:{}},{id:9,time:new Date(start+900000).toISOString(),type:'unit_complete',data:{id:'train-4'}}];
  expect(rewardEta({...status,recent:interrupted},start+900000).label).toContain('Calibrating');
  expect(rewardEta({...status,recent:[...recent,{id:8,time:new Date(start+90000).toISOString(),type:'execution_revision',data:{}}]},start+90000).label).toContain('Calibrating');
});
