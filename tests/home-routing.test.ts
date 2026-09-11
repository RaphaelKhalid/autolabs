import {afterEach,describe,expect,it,vi} from 'vitest';
vi.mock('next/navigation',()=>({redirect:(path:string)=>{throw new Error(`redirect:${path}`);}}));
vi.mock('next/server',()=>({connection:async()=>{}}));
import Home from '../app/page';
import {experiments,latestExperiment} from '../lib/experiment-catalog';

afterEach(()=>{vi.unstubAllEnvs();vi.unstubAllGlobals();});
describe('newest experiment landing page',()=>{
  it('orders the register by descending unique experiment sequence',()=>{
    expect(new Set(experiments.map(e=>e.sequence)).size).toBe(experiments.length);
    expect(experiments.map(e=>e.sequence)).toEqual([...experiments].sort((a,b)=>b.sequence-a.sequence).map(e=>e.sequence));
    expect(latestExperiment.slug).toBe('reward-compatibility-21');
  });
  it('opens the published experiment when its successor is not running',async()=>{
    vi.stubEnv('AUTOLABS_SELF_HOSTED','0');
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({status:'waiting'})}));
    await expect(Home()).rejects.toThrow(`redirect:/experiments/${latestExperiment.slug}`);
  });
  it('keeps the creator as the self-hosted entry point without any status fetch',async()=>{
    vi.stubEnv('AUTOLABS_SELF_HOSTED','1');
    const fetcher=vi.fn();vi.stubGlobal('fetch',fetcher);
    await expect(Home()).rejects.toThrow('redirect:/studio');
    expect(fetcher).not.toHaveBeenCalled();
  });
});
