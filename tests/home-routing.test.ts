import {describe,expect,it} from 'vitest';
import Home from '../app/page';
import {experiments,latestExperiment} from '../lib/experiment-catalog';
import {Persona3BLab} from '../components/persona-3b-lab';

describe('newest experiment landing page',()=>{
  it('orders the register by descending unique experiment sequence',()=>{
    expect(new Set(experiments.map(e=>e.sequence)).size).toBe(experiments.length);
    expect(experiments.map(e=>e.sequence)).toEqual([...experiments].sort((a,b)=>b.sequence-a.sequence).map(e=>e.sequence));
    expect(latestExperiment.slug).toBe('persona-discovery');
  });
  it('renders the Experiment 3B lab as the current homepage',()=>{
    const element=Home();
    expect(element.type).toBe(Persona3BLab);
  });
});
