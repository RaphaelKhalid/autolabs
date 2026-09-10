import type { Agent, Configuration } from './config.ts';
export type Message = {role:'system'|'user';content:string};
export type Price = {input:number;output:number;request:number};
export type Model = {id:string;name:string;price:Price};
export type Completion = {text:string;model:string;responseId:string;costUsd:number;costBasis:'reported'|'reserved'|'mock';inputTokens:number;outputTokens:number};
export async function models(): Promise<Model[]> {
  const response = await fetch('https://openrouter.ai/api/v1/models',{signal:AbortSignal.timeout(15000)});
  if (!response.ok) throw new Error(`Model catalogue unavailable (${response.status}).`);
  const body = await response.json() as {data:{id:string;name:string;pricing:Record<string,string>;architecture?:{input_modalities?:string[];output_modalities?:string[]}}[]};
  return body.data.filter(m=>m.architecture?.output_modalities?.includes('text')).flatMap(m=>{
    const price = {input:Number(m.pricing.prompt),output:Number(m.pricing.completion),request:Number(m.pricing.request??0)};
    return Object.values(price).every(n=>Number.isFinite(n)&&n>=0) ? [{id:m.id,name:m.name,price}] : [];
  });
}
export function reserveCost(messages:Message[],maxOutputTokens:number,price:Price):number {
  // Conservative text-token estimate with overhead; not a substitute for a provider-side key limit.
  return ((Buffer.byteLength(JSON.stringify(messages),'utf8')+4096)*price.input+maxOutputTokens*price.output+price.request)*1.25;
}
export async function complete(config:Configuration,agent:Agent,messages:Message[],reserve:number,apiKey:string|undefined):Promise<Completion> {
  if (config.provider==='mock') return {text:JSON.stringify({summary:'Mock response: no model was called.',proposals:[{x:17,y:-9}]}),model:'mock',responseId:crypto.randomUUID(),costUsd:0,costBasis:'mock',inputTokens:0,outputTokens:0};
  if (!apiKey) throw new Error('OpenRouter is not configured on the server.');
  const response = await fetch('https://openrouter.ai/api/v1/chat/completions',{
    method:'POST',signal:AbortSignal.timeout(90000),headers:{authorization:`Bearer ${apiKey}`,'content-type':'application/json'},
    body:JSON.stringify({model:agent.model,messages,max_tokens:config.maxOutputTokens,stream:false,provider:{require_parameters:true},plugins:[]}),
  });
  if (!response.ok) throw new Error(`OpenRouter request failed (${response.status}). No automatic retry; reservation retained.`);
  const result=await response.json() as {id?:string;model?:string;error?:unknown;choices?:{message?:{content?:string}}[];usage?:{cost?:number;prompt_tokens?:number;completion_tokens?:number}};
  if (result.error) throw new Error('OpenRouter returned an error. Reservation retained.');
  const rawCost=result.usage?.cost;
  const reported=typeof rawCost==='number'&&Number.isFinite(rawCost)&&rawCost>=0;
  // Only public assistant text is retained. Reasoning fields are deliberately ignored.
  return {text:result.choices?.[0]?.message?.content??'',model:result.model??agent.model,responseId:result.id??'unavailable',costUsd:reported?rawCost:reserve,costBasis:reported?'reported':'reserved',inputTokens:result.usage?.prompt_tokens??0,outputTokens:result.usage?.completion_tokens??0};
}
export function parseReport(text:string):{summary:string;proposals:unknown[]} {
  const stripped=text.trim().replace(/^```(?:json)?\s*/,'').replace(/\s*```$/,'');
  const body=JSON.parse(stripped) as {summary?:unknown;proposals?:unknown};
  if(typeof body.summary!=='string'||!Array.isArray(body.proposals)) throw new Error('Response must contain a summary and a proposals array.');
  return {summary:body.summary.slice(0,8000),proposals:body.proposals.slice(0,20)};
}
