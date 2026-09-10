import {afterEach,describe,expect,it,vi} from 'vitest';
vi.mock('next/navigation',()=>({redirect:(path:string)=>{throw new Error(`redirect:${path}`);}}));
import Home from '../app/page';
import {experiments,latestExperiment} from '../lib/experiment-catalog';

afterEach(()=>vi.unstubAllEnvs());
describe('newest experiment landing page',()=>{
  it('orders the register by descending unique experiment sequence',()=>{
    expect(new Set(experiments.map(e=>e.sequence)).size).toBe(experiments.length);
    expect(experiments.map(e=>e.sequence)).toEqual([...experiments].sort((a,b)=>b.sequence-a.sequence).map(e=>e.sequence));
    expect(latestExperiment.slug).toBe('reward-compatibility');
  });
  it('opens the newest experiment on the public site',()=>{
    vi.stubEnv('AUTOLABS_SELF_HOSTED','0');
    expect(()=>Home()).toThrow(`redirect:/experiments/${latestExperiment.slug}`);
  });
  it('keeps the creator as the self-hosted entry point',()=>{
    vi.stubEnv('AUTOLABS_SELF_HOSTED','1');
    expect(()=>Home()).toThrow('redirect:/studio');
  });
});
