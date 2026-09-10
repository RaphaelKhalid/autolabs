import { afterEach, describe, expect, it, vi } from 'vitest';
import { GET, POST } from '../app/api/selfhost/[...path]/route';
const token='test-owner-token-with-at-least-32-characters';
const context=(path=['runs'])=>({params:Promise.resolve({path})});
afterEach(()=>{vi.unstubAllEnvs();vi.restoreAllMocks();});
describe('self-hosted control boundary',()=>{
  it.each([['runs'],['check'],['runs','run-test','resume'],['runs','run-test','pause']])('blocks public writes to %s even with an owner credential',async(...path)=>{
    vi.stubEnv('AUTOLABS_SELF_HOSTED','0');vi.stubEnv('AUTOLABS_LOCAL_TOKEN',token);
    const forward=vi.spyOn(globalThis,'fetch');
    const response=await POST(new Request(`https://autolabs-ebon.vercel.app/api/selfhost/${path.join('/')}`,{method:'POST',headers:{authorization:`Bearer ${token}`,origin:'https://autolabs-ebon.vercel.app'},body:'{}'}),context(path));
    expect(response.status).toBe(404);expect(forward).not.toHaveBeenCalled();
  });
  it('accepts the browser host when Next normalizes the internal request URL',async()=>{
    vi.stubEnv('AUTOLABS_SELF_HOSTED','1');vi.stubEnv('AUTOLABS_LOCAL_TOKEN',token);
    vi.spyOn(globalThis,'fetch').mockResolvedValue(Response.json({id:'run-test'},{status:201}));
    const request=new Request('http://localhost:3000/api/selfhost/runs',{method:'POST',headers:{authorization:`Bearer ${token}`,host:'127.0.0.1:3000',origin:'http://127.0.0.1:3000'},body:'{}'});
    expect((await POST(request,context())).status).toBe(201);
  });
  it('disables the entire proxy outside self-hosted mode',async()=>{vi.stubEnv('AUTOLABS_SELF_HOSTED','0');expect((await GET(new Request('http://localhost/api/selfhost/runs'),context())).status).toBe(404);});
  it('requires the owner credential',async()=>{vi.stubEnv('AUTOLABS_SELF_HOSTED','1');vi.stubEnv('AUTOLABS_LOCAL_TOKEN',token);expect((await GET(new Request('http://localhost/api/selfhost/runs'),context())).status).toBe(401);});
  it('blocks cross-origin writes even with a valid owner credential',async()=>{vi.stubEnv('AUTOLABS_SELF_HOSTED','1');vi.stubEnv('AUTOLABS_LOCAL_TOKEN',token);expect((await POST(new Request('http://localhost/api/selfhost/runs',{method:'POST',headers:{authorization:`Bearer ${token}`,origin:'https://untrusted.example'},body:'{}'}),context())).status).toBe(403);});
  it('does not turn arbitrary paths into an open proxy',async()=>{vi.stubEnv('AUTOLABS_SELF_HOSTED','1');vi.stubEnv('AUTOLABS_LOCAL_TOKEN',token);expect((await GET(new Request('http://localhost/api/selfhost/nope',{headers:{authorization:`Bearer ${token}`}}),context(['nope']))).status).toBe(404);});
  it('only forwards the owner authorization, not server-side provider keys',async()=>{
    vi.stubEnv('AUTOLABS_SELF_HOSTED','1');vi.stubEnv('AUTOLABS_LOCAL_TOKEN',token);vi.stubEnv('OPENROUTER_API_KEY','private-provider-secret');
    const fetch=vi.spyOn(globalThis,'fetch').mockResolvedValue(Response.json({runs:[]}));
    const response=await GET(new Request('http://localhost/api/selfhost/runs',{headers:{authorization:`Bearer ${token}`}}),context());
    expect(response.status).toBe(200);expect(JSON.stringify(fetch.mock.calls)).not.toContain('private-provider-secret');
  });
});
