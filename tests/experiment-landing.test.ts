import {afterEach,describe,expect,it,vi} from 'vitest';
import {currentExperimentPath} from '../lib/experiment-landing';
const launched={runId:'experiment-002-2-v1',status:'running',startedAt:'2026-09-11T08:00:00Z',protocolHash:'a'.repeat(64)};
afterEach(()=>vi.unstubAllGlobals());
describe('bounded live experiment promotion',()=>{
  it.each(['running','complete'])('promotes a frozen launched %s successor',async status=>{
    const fetcher=vi.fn().mockResolvedValue({ok:true,json:async()=>({...launched,status})});vi.stubGlobal('fetch',fetcher);
    expect(await currentExperimentPath()).toBe('/experiments/reward-categories-22');
    expect(fetcher).toHaveBeenCalledWith(expect.stringMatching(/\/status$/),expect.objectContaining({next:{revalidate:15},signal:expect.any(AbortSignal)}));
    expect(fetcher.mock.calls[0][1].method).toBeUndefined();
  });
  it.each(['waiting','ready','paused'])('does not promote %s',async status=>{
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>({...launched,status})}));
    expect(await currentExperimentPath()).toBe('/experiments/reward-compatibility-21');
  });
  it.each([null,{}, {...launched,protocolHash:null},{...launched,startedAt:'invalid'},{...launched,runId:'another-run'}])('rejects malformed or different launches',async data=>{
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:true,json:async()=>data}));
    expect(await currentExperimentPath()).toBe('/experiments/reward-compatibility-21');
  });
  it('falls back on network errors and HTTP failures',async()=>{
    vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('timeout')));
    expect(await currentExperimentPath()).toBe('/experiments/reward-compatibility-21');
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:false}));
    expect(await currentExperimentPath()).toBe('/experiments/reward-compatibility-21');
  });
});
