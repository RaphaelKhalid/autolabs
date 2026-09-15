'use client';

export function TraceMark({ size = 36, label = true }: { size?: number; label?: boolean }) {
  return <span className="trace-brand" style={{ '--trace-size': `${size}px` } as React.CSSProperties} aria-label={label ? 'AutoLabs' : 'Trace mark'}>
    <svg className="trace-brand__svg" viewBox="0 0 120 90" role="img" aria-hidden="true">
      <path className="trace-brand__a" d="M13 76 C24 68 30 43 47 15 C51 8 57 8 60 16 L82 76 M40 51 C49 48 59 48 73 50" />
      <path className="trace-brand__orbit" d="M8 45 C25 12 77 9 106 30 C122 43 106 66 80 72 C50 80 17 68 8 45 Z" />
      <circle className="trace-brand__dot" cx="99" cy="27" r="5" />
    </svg>
    {label && <span className="trace-brand__word">AUTOLABS</span>}
    <style jsx>{` 
      .trace-brand{display:inline-flex;align-items:center;gap:10px;color:inherit;white-space:nowrap}
      .trace-brand__svg{width:var(--trace-size);height:calc(var(--trace-size) * .75);overflow:visible}
      .trace-brand__svg path{fill:none;stroke:currentColor;stroke-width:5;stroke-linecap:round;stroke-linejoin:round}
      .trace-brand__orbit{stroke:#a85238!important;stroke-width:3!important;stroke-dasharray:300;stroke-dashoffset:300;animation:trace-draw 2.2s .45s cubic-bezier(.65,0,.2,1) forwards}
      .trace-brand__a{stroke-dasharray:230;stroke-dashoffset:230;animation:trace-draw 1.6s cubic-bezier(.65,0,.2,1) forwards}
      .trace-brand__dot{fill:#a85238;transform-origin:99px 27px;transform:scale(0);animation:trace-dot .45s 2.1s cubic-bezier(.2,.9,.25,1.4) forwards}
      .trace-brand__word{font:650 9px var(--font-mono),monospace;letter-spacing:.16em}
      @keyframes trace-draw{to{stroke-dashoffset:0}}
      @keyframes trace-dot{to{transform:scale(1)}}
      @media(prefers-reduced-motion:reduce){.trace-brand__a,.trace-brand__orbit{animation:none;stroke-dashoffset:0}.trace-brand__dot{animation:none;transform:scale(1)}}
    `}</style>
  </span>;
}
