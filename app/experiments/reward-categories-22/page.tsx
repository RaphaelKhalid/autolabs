import type {Metadata} from 'next';
import Image from 'next/image';
import {Compatibility22Lab} from '@/components/compatibility-22-lab';
import '../reward-compatibility/reward-lab.css';
import '../reward-compatibility/photograph.css';
import '../reward-compatibility/metrics.css';
import '../reward-compatibility-21/compatibility21.css';
import './categories22.css';
export const metadata:Metadata={title:'Reward pair categories · Experiment 002.2 · AutoLabs',description:'Reference-relative reward categories from independent paired API searches over finite executable policies. Live cloud progress, shared budget and sealed evaluation.'};
export default function Categories22Page(){return <div className="reward-experiment"><div className="reward-photograph" aria-hidden="true"><Image src="/photography/sunset-flight.jpg" alt="" fill sizes="100vw" loading="eager"/></div><Compatibility22Lab/></div>;}
