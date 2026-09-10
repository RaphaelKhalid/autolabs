import { createServer } from 'node:http';
import { createHash, timingSafeEqual } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, openSync, readFileSync, unlinkSync, writeFileSync, closeSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { Engine } from './engine.ts';
import { complete, models, reserveCost } from './provider.ts';
import type { Model, Message } from './provider.ts';
import { starter } from './config.ts';

const token=process.env.AUTOLABS_LOCAL_TOKEN;
if(!token||token.length<32)throw new Error('Set AUTOLABS_LOCAL_TOKEN to at least 32 random characters.');
const directory=resolve(process.env.AUTOLABS_DATA_DIR??'.autolabs');
mkdirSync(directory,{recursive:true,mode:0o700});
const lock=join(directory,'runner.lock');
if(existsSync(lock)) {
  const previous=Number(readFileSync(lock,'utf8'));
  if(!Number.isSafeInteger(previous)||previous<1)throw new Error('Invalid runner lock; inspect the data directory before starting.');
  try {process.kill(previous,0);throw new Error('A runner process already owns this data directory.');}
  catch(error) {if((error as NodeJS.ErrnoException).code==='ESRCH')unlinkSync(lock);else throw error;}
}
const lockFd=openSync(lock,'wx',0o600);writeFileSync(lockFd,String(process.pid));closeSync(lockFd);
process.on('exit',()=>{if(existsSync(lock)&&readFileSync(lock,'utf8')===String(process.pid))unlinkSync(lock);});
const engine=new Engine(directory);
let catalogue:Model[]=[],catalogueAt=0,checkBusy=false,lastCheck=0;
async function getModels(){if(Date.now()-catalogueAt>15*60*1000){catalogue=await models();catalogueAt=Date.now();}return catalogue;}
function authorized(value:string|undefined){return timingSafeEqual(createHash('sha256').update(value??'').digest(),createHash('sha256').update(`Bearer ${token}`).digest());}
const server=createServer(async(req,res)=>{
  res.setHeader('content-type','application/json');res.setHeader('cache-control','no-store');res.setHeader('x-content-type-options','nosniff');
  const send=(status:number,value:unknown)=>{res.statusCode=status;res.end(JSON.stringify(value));};
  if(req.url==='/health'&&req.method==='GET')return send(200,{ok:true});
  if(!authorized(req.headers.authorization))return send(401,{error:'Owner authorization required.'});
  const url=new URL(req.url??'/','http://runner');
  try {
    if(req.method==='GET'&&url.pathname==='/connections'){
      let available:Model[]=[];
      try {available=await getModels();} catch { /* Mock runs remain usable offline. */ }
      return send(200,{openrouterConfigured:Boolean(process.env.OPENROUTER_API_KEY),models:available});
    }
    if(req.method==='GET'&&url.pathname==='/runs')return send(200,{runs:[...engine.runs.values()].reverse().map(r=>({id:r.id,title:r.config.title,provider:r.config.provider,status:r.status,round:r.round,rounds:r.config.rounds,spentUsd:r.spentUsd,createdAt:r.createdAt}))});
    const match=url.pathname.match(/^\/runs\/(run-[a-z0-9-]+)(?:\/(pause|resume|export))?$/);
    if(match&&req.method==='GET'){
      const run=engine.runs.get(match[1]);if(!run)return send(404,{error:'Run not found.'});
      if(match[2]==='export'){res.setHeader('content-disposition',`attachment; filename="${run.id}.json"`);return send(200,run);}
      return send(200,{...run,eventCount:run.events.length,events:run.events.slice(-100)});
    }
    if(req.method!=='POST')return send(404,{error:'Not found.'});
    const chunks:Buffer[]=[];let size=0;
    for await(const chunk of req){size+=chunk.length;if(size>64000)return send(413,{error:'Request is too large.'});chunks.push(Buffer.from(chunk));}
    const body=JSON.parse(Buffer.concat(chunks).toString()||'{}');
    if(url.pathname==='/runs'){
      if(body.provider==='openrouter'){
        if(!process.env.OPENROUTER_API_KEY)return send(400,{error:'Configure OPENROUTER_API_KEY in the server environment first.'});
        await getModels();
      }
      return send(201,engine.create(body));
    }
    if(match&&(match[2]==='pause'||match[2]==='resume'))return send(200,engine.control(match[1],match[2]));
    if(url.pathname==='/check'){
      if(checkBusy||Date.now()-lastCheck<15000)return send(429,{error:'Wait 15 seconds between connection checks.'});
      if(!process.env.OPENROUTER_API_KEY)return send(400,{error:'OpenRouter key is not configured.'});
      const model=(await getModels()).find(m=>m.id===body.model);
      if(!model)return send(400,{error:'Select an available model.'});
      const messages:Message[]=[{role:'user',content:'Connection check. Reply with OK only.'}];
      const reserve=reserveCost(messages,16,model.price);
      if(reserve>0.01)return send(400,{error:'This model exceeds the $0.01 connection-check reservation limit.'});
      checkBusy=true;lastCheck=Date.now();
      const checkId=crypto.randomUUID();
      appendFileSync(join(directory,'connection-checks.jsonl'),JSON.stringify({id:checkId,at:new Date().toISOString(),model:model.id,status:'reserved',reservedUsd:reserve})+'\n',{mode:0o600});
      try {
        const result=await complete({...starter,provider:'openrouter',maxOutputTokens:16},{...starter.agents[0],model:model.id},messages,reserve,process.env.OPENROUTER_API_KEY);
        appendFileSync(join(directory,'connection-checks.jsonl'),JSON.stringify({id:checkId,status:'returned',model:result.model,costUsd:result.costUsd,costBasis:result.costBasis})+'\n');
        return send(200,{ok:Boolean(result.text.trim()),model:result.model,costUsd:result.costUsd,costBasis:result.costBasis});
      } finally {checkBusy=false;}
    }
    return send(404,{error:'Not found.'});
  } catch(error){return send(400,{error:error instanceof Error?error.message:'Request failed.'});}
});
const port=Number(process.env.AUTOLABS_RUNNER_PORT??8788);
server.listen(port,process.env.AUTOLABS_RUNNER_HOST??'127.0.0.1',()=>console.log(`AutoLabs runner listening on port ${port}. Owner authorization required.`));
const timer=setInterval(()=>void (async()=>{
  const paid=[...engine.runs.values()].find(r=>r.status==='running'&&r.config.provider==='openrouter');
  if(paid) {
    try {if(!process.env.OPENROUTER_API_KEY)throw new Error('Missing key');await getModels();}
    catch {paid.status='paused';engine.event(paid,'error','Provider credentials or pricing catalogue unavailable. No new request issued; review before resuming.');return;}
  }
  await engine.tick(catalogue,process.env.OPENROUTER_API_KEY);
})().catch(()=>console.error('Runner tick failed; inspect the local ledger.')),1000);
const cleanup=()=>{clearInterval(timer);server.close(()=>process.exit(0));};
process.on('SIGTERM',cleanup);process.on('SIGINT',cleanup);
