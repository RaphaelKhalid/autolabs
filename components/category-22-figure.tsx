import {categoryLabel,observedCategoryLabel,type CategoryResult} from '@/lib/compatibility-22-display';
const coordinate=(value:number)=>5+90*(Math.max(-1,Math.min(1,value))+1)/2;
const percent=(value:number)=>`${value>=0?'+':''}${(100*value).toFixed(1)} pp`;
export function Category22Figure({rows}:{rows:CategoryResult[]}){
  return <figure className="category22-figure" aria-label="Outcome change and simultaneous uncertainty by reward template">
    <div className="category22-axis"><span>−100 pp</span><span>0</span><span>+100 pp</span></div>
    {rows.map(row=>{
      const bounds=row.distributionFreeIntervals;
      const color=row.label==='population-aligned-direction-supported'?'#3d755a':row.label==='population-conflict-direction-supported'?'#a74e39':row.label==='population-outcome-equivalence-with-observed-witnesses'?'#546d91':'#74776e';
      return <div className="category22-figure-row" key={row.templateId}>
        <div className="category22-figure-name"><strong>{row.templateId}</strong><span>{categoryLabel(row.label)}</span></div>
        <svg viewBox="0 0 100 12" preserveAspectRatio="none" role="img" aria-label={bounds?`Simultaneous uncertainty envelope ${percent(bounds.minimum[0])} to ${percent(bounds.maximum[1])}`:'Estimate unavailable'}>
          <rect x={coordinate(-.05)} y="0" width={coordinate(.05)-coordinate(-.05)} height="12" fill="#d6d9cb"/>
          <line x1="50" x2="50" y1="0" y2="12" stroke="#83877c" strokeWidth=".3"/>
          <line x1="5" x2="95" y1="6" y2="6" stroke="#d3d4ca" strokeWidth=".3"/>
          {bounds&&<g stroke={color} strokeWidth=".65"><line x1={coordinate(bounds.minimum[0])} x2={coordinate(bounds.maximum[1])} y1="6" y2="6"/><line x1={coordinate(bounds.minimum[0])} x2={coordinate(bounds.minimum[0])} y1="4" y2="8"/><line x1={coordinate(bounds.maximum[1])} x2={coordinate(bounds.maximum[1])} y1="4" y2="8"/></g>}
          {row.meanGainMin!==null&&row.meanGainMax!==null&&<g fill={color}><circle cx={coordinate(row.meanGainMin)} cy="6" r="1.25"/><circle cx={coordinate(row.meanGainMax)} cy="6" r="1.25"/></g>}
        </svg>
        <p className="category22-figure-detail">Witnesses {row.discoveredPreservingWitnessPairs}/64 · reference at ceiling {row.referenceCeilingPairs}/64<br/><span>Observed histories: {observedCategoryLabel(row.observedHistoryLabel)}</span></p>
      </div>;
    })}
    <figcaption>Dots show mean outcome changes across tied optima; lines span the simultaneous distribution-free bounds. Shading marks ±5 percentage points. Color reflects population support only. Finite canonical policies; 64 pairs per template and four calls per arm. Not hidden chain-of-thought measurement or a universal category proof.</figcaption>
  </figure>;
}
