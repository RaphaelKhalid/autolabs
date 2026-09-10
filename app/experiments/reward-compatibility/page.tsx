import type {Metadata} from 'next';
import {RewardLab} from '@/components/reward-lab';
import './reward-lab.css';
export const metadata:Metadata={title:'Reward compatibility · Experiment 002 · AutoLabs',description:'A live, preregistered study of reward compatibility and reasoning monitorability, with exact task scoring and a public research ledger.'};
export default function RewardExperiment(){return <RewardLab/>;}
