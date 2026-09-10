import { createHash, timingSafeEqual } from 'node:crypto';

export const runtime='nodejs';
export const dynamic='force-dynamic';
async function proxy(request:Request,context:{params:Promise<{path:string[]}>}) {
  const token=process.env.AUTOLABS_LOCAL_TOKEN;
  if(process.env.AUTOLABS_SELF_HOSTED!=='1'||!token||token.length<32)return Response.json({error:'Self-hosted controls are disabled.'},{status:404});
  const supplied=request.headers.get('authorization')??'';
  if(!timingSafeEqual(createHash('sha256').update(supplied).digest(),createHash('sha256').update(`Bearer ${token}`).digest()))return Response.json({error:'Owner token required.'},{status:401});
  const origin=request.headers.get('origin');
  // Next may normalize request.url to its bind address; Host retains the browser authority.
  const requestUrl=new URL(request.url);
  const expectedOrigin=`${requestUrl.protocol}//${request.headers.get('host')??requestUrl.host}`;
  if(origin&&origin!==expectedOrigin)return Response.json({error:'Cross-origin requests are not permitted.'},{status:403});
  const {path}=await context.params;
  const route=path.join('/');
  if(!/^(connections|check|runs(?:\/run-[a-z0-9-]+(?:\/(?:pause|resume|export))?)?)$/.test(route))return Response.json({error:'Unknown operation.'},{status:404});
  try {
    const body=request.method==='POST'?await request.text():undefined;
    if(body&&Buffer.byteLength(body)>64000)return Response.json({error:'Request too large.'},{status:413});
    const result=await fetch(`${process.env.AUTOLABS_RUNNER_URL??'http://127.0.0.1:8788'}/${route}`,{method:request.method,headers:{authorization:supplied,'content-type':'application/json'},body,cache:'no-store',signal:AbortSignal.timeout(route==='check'?95000:20000)});
    return new Response(result.body,{status:result.status,headers:{'content-type':'application/json','cache-control':'no-store',...(result.headers.has('content-disposition')?{'content-disposition':result.headers.get('content-disposition')!}:{})}});
  } catch{return Response.json({error:'Local runner unavailable. Start the runner and retry.'},{status:503});}
}
export const GET=proxy;
export const POST=proxy;
