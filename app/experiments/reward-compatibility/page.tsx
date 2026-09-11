import type {Metadata} from 'next';
import Image from 'next/image';
import {RewardLab} from '@/components/reward-lab';
import './reward-lab.css';
import './photograph.css';
import './metrics.css';
export const metadata:Metadata={title:'Reward compatibility · Experiment 002 · AutoLabs',description:'A live, preregistered study of reward compatibility and reasoning monitorability, with exact task scoring and a public research ledger.'};
export default function RewardExperiment(){return <div className="reward-experiment">
  <div className="reward-photograph" aria-hidden="true">
    <Image src="/photography/sunset-flight.jpg" alt="" fill sizes="100vw" loading="eager"/>
  </div>
  <RewardLab/>
</div>;}
