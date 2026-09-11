import {describe,it,expect} from 'vitest';
import {rewardCallDisplay as display} from '../lib/reward-call-display';
describe('public reward call labels',()=>{
  it('maps the semantic rating to the actual negative reward',()=>{expect(display('train-semantic-combined-2-10/score-4-reward','{"rating":4}')).toMatchObject({metric:'r_CoT',value:-.8,scope:'Training sample'});});
  it('keeps preference and process signs correct',()=>{expect(display('train-preference-outcome-1-0/score-1-reward','{"rating":1}').value).toBe(1);expect(display('diagnostic-process-0/score-1-reward','{"rating":1}').value).toBe(-1);});
  it('does not invent values for pending, malformed or invalid output',()=>{for(const text of [undefined,'invalid','{"rating":"1"}','{"rating":9}','{"rating":0.5}'])expect(display('train-semantic-combined-0-0/score-1-reward',text).value).toBeUndefined();});
  it('distinguishes monitoring from its transformed reward',()=>{expect(display('evaluation-semantic-combined-0/eval-1-monitor','{"rating":4}')).toMatchObject({metric:'m',value:4,scope:'Held-out sample'});});
  it('does not invent correctness scores from actor text',()=>{expect(display('train-length-16-combined-0-0/transfer-1','<answer>Heads</answer>').value).toBeUndefined();});
  it('recognizes gate calls and does not equate audits with reward',()=>{expect(display('gate-trial-process/score-1-reward','{"rating":0}').value).toBeCloseTo(0);expect(display('diagnostic-string-0/score-1-audit').metric).toBeUndefined();});
});
