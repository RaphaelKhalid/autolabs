import type {Metadata} from 'next';
import Image from 'next/image';
import {Compatibility21Lab} from '@/components/compatibility-21-lab';
import '../reward-compatibility/reward-lab.css';
import '../reward-compatibility/photograph.css';
import '../reward-compatibility/metrics.css';
import './compatibility21.css';

export const metadata:Metadata={title:'Finite reward compatibility · Experiment 002.1 · AutoLabs',description:'A fixed, independently checked reward-compatibility study with live cloud progress, shared budget accounting and sealed held-out evaluation.'};
export default function Compatibility21Page(){return <div className="reward-experiment">
  <div className="reward-photograph" aria-hidden="true"><Image src="/photography/sunset-flight.jpg" alt="" fill sizes="100vw" loading="eager"/></div>
  <Compatibility21Lab/>
</div>;}
