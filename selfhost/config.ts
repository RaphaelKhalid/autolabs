export type Agent = { id: string; name: string; model: string; instructions: string; color: string };
export type Configuration = {
  schemaVersion: 1; title: string; objective: string;
  template: 'integer-search-v1' | 'research-notes-v1';
  provider: 'openrouter' | 'mock'; agents: Agent[];
  rounds: number; phaseSeconds: number; budgetUsd: number; maxOutputTokens: number;
  meetings: boolean; stopOnSuccess: boolean;
};
export const starter: Configuration = {
  schemaVersion: 1, title: 'Integer-search systems test',
  objective: 'Find integer x,y in [-100,100] minimizing (x-17)^2 + (y+9)^2. This is a known-answer infrastructure test, not open research.',
  template: 'integer-search-v1', provider: 'mock', rounds: 2, phaseSeconds: 10,
  budgetUsd: 0.10, maxOutputTokens: 1024, meetings: true, stopOnSuccess: true,
  agents: [
    {id:'agent-1',name:'Researcher 1',model:'openai/gpt-4.1-nano',instructions:'Derive candidate solutions algebraically. Report assumptions.',color:'#9a593c'},
    {id:'agent-2',name:'Researcher 2',model:'openai/gpt-4.1-nano',instructions:'Check constraints and challenge unsupported conclusions.',color:'#366f73'},
  ],
};
function text(value: unknown, label: string, max: number): string {
  if (typeof value !== 'string' || !value.trim() || value.length > max) throw new Error(`${label} must contain 1–${max} characters.`);
  if (/sk-[\w-]{16,}/.test(value)) throw new Error('Do not put API keys in experiment configurations. Use the server environment.');
  return value.trim();
}
function numeric(value: unknown, label: string, min: number, max: number, integer = true): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) throw new Error(`${label} must be between ${min} and ${max}.`);
  return value;
}
export function validateConfiguration(value: unknown): Configuration {
  if (!value || typeof value !== 'object') throw new Error('Configuration must be an object.');
  const v = value as Record<string, unknown>;
  if (v.schemaVersion !== 1) throw new Error('Unsupported configuration version.');
  if (v.template !== 'integer-search-v1' && v.template !== 'research-notes-v1') throw new Error('Unknown evaluator template.');
  if (v.provider !== 'openrouter' && v.provider !== 'mock') throw new Error('Choose OpenRouter or the no-cost mock provider.');
  if (!Array.isArray(v.agents) || v.agents.length < 1 || v.agents.length > 8) throw new Error('Configure 1–8 agents.');
  const agents = v.agents.map((value): Agent => {
    if (!value || typeof value !== 'object') throw new Error('Invalid agent.');
    const a = value as Record<string, unknown>;
    const id = text(a.id,'Agent ID',40);
    if (!/^[a-z0-9-]+$/.test(id)) throw new Error('Agent IDs must use lowercase letters, digits and hyphens.');
    const color = text(a.color,'Color',7);
    if (!/^#[0-9a-f]{6}$/i.test(color)) throw new Error('Use a six-digit hex color.');
    const model = text(a.model,'Model',120);
    if (!/^[a-zA-Z0-9_./:-]+$/.test(model) || model.includes('..')) throw new Error('Invalid model ID.');
    return {id,name:text(a.name,'Agent name',60),model,instructions:text(a.instructions,'Agent instructions',4000),color};
  });
  if (new Set(agents.map(a=>a.id)).size !== agents.length) throw new Error('Agent IDs must be unique.');
  if (typeof v.meetings !== 'boolean' || typeof v.stopOnSuccess !== 'boolean') throw new Error('Protocol switches must be boolean.');
  return {schemaVersion:1,title:text(v.title,'Title',160),objective:text(v.objective,'Objective',8000),template:v.template,provider:v.provider,agents,
    rounds:numeric(v.rounds,'Rounds',1,100),phaseSeconds:numeric(v.phaseSeconds,'Phase interval',0,3600),
    budgetUsd:numeric(v.budgetUsd,'Budget',0.01,50,false),maxOutputTokens:numeric(v.maxOutputTokens,'Output tokens',64,4096),meetings:v.meetings,stopOnSuccess:v.stopOnSuccess};
}
export type Evaluation = {status:'verified'|'invalid'|'human-review'; score:number|null; success:boolean; detail:string};
export function evaluate(template: Configuration['template'], proposal: unknown): Evaluation {
  if (template === 'research-notes-v1') return {status:'human-review',score:null,success:false,detail:'Research notes require human review. No automated novelty or correctness claim.'};
  const p = proposal as {x?:unknown;y?:unknown} | null;
  if (!p || typeof p.x !== 'number' || typeof p.y !== 'number' || !Number.isInteger(p.x) || !Number.isInteger(p.y) || Math.abs(p.x)>100 || Math.abs(p.y)>100) return {status:'invalid',score:null,success:false,detail:'x and y must be integers between -100 and 100.'};
  const score = (p.x-17)**2+(p.y+9)**2;
  return {status:'verified',score,success:score===0,detail:`Exact objective value ${score}; lower is better. Known-answer systems test.`};
}
