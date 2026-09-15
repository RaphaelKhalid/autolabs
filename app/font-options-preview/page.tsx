import type { CSSProperties } from 'react';
import type { Metadata } from 'next';
import { Chakra_Petch, Kalam, Recursive } from 'next/font/google';
import { TraceMark } from '@/components/trace-mark';
import './font-options.css';

const recursive = Recursive({
  subsets: ['latin'],
  variable: '--font-recursive',
  axes: ['CASL', 'CRSV', 'MONO', 'slnt'],
  display: 'swap',
});

const kalam = Kalam({
  subsets: ['latin'],
  weight: ['400', '700'],
  variable: '--font-kalam',
  display: 'swap',
});

const chakra = Chakra_Petch({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-chakra',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Font options · AutoLabs',
  description: 'Three readable cyberpunk labnote typography directions for AutoLabs.',
};

const variations = [
  { rotate: -0.8, y: 1, scale: 1.01 },
  { rotate: 0.45, y: -1, scale: .99 },
  { rotate: -0.25, y: .5, scale: 1.005 },
  { rotate: 1.05, y: -0.5, scale: .995 },
  { rotate: -0.55, y: 1.2, scale: 1.01 },
  { rotate: 0.2, y: -0.8, scale: .99 },
];

function VariedText({ children, className = '' }: { children: string; className?: string }) {
  return <span className={className} aria-label={children}>
    {Array.from(children).map((character, index) => {
      const variation = variations[index % variations.length];
      const style = {
        '--glyph-rotate': `${variation.rotate}deg`,
        '--glyph-y': `${variation.y}px`,
        '--glyph-scale': variation.scale,
      } as CSSProperties;
      return <span className="glyph" style={style} aria-hidden="true" key={`${character}-${index}`}>{character === ' ' ? '\u00a0' : character}</span>;
    })}
  </span>;
}

const options = [
  {
    id: 'A',
    name: 'Casual signal',
    fontClass: 'font-recursive',
    className: 'option-casual',
    description: 'Recursive with its casual axis turned up. The most human and lab-notebook-like.',
  },
  {
    id: 'B',
    name: 'Ink terminal',
    fontClass: 'font-kalam',
    className: 'option-ink',
    description: 'Kalam with restrained per-letter drift. The clearest handwritten direction.',
  },
  {
    id: 'C',
    name: 'Angular note',
    fontClass: 'font-chakra',
    className: 'option-angular',
    description: 'Chakra Petch with hand-drawn irregularity. The most futuristic and least cursive.',
  },
];

export default function FontOptionsPreviewPage() {
  return <main className={`font-options ${recursive.variable} ${kalam.variable} ${chakra.variable}`}>
    <header className="font-options__header">
      <TraceMark size={38} />
      <div>
        <p className="font-options__eyebrow">Typography specimen / AutoLabs</p>
        <h1>Three directions for a readable labnote voice</h1>
        <p>Each card uses a different font plus deterministic per-letter variation, so repeated characters do not look mechanically identical.</p>
      </div>
    </header>

    <section className="font-options__grid" aria-label="Three font options">
      {options.map((option) => <article className={`font-option ${option.className}`} key={option.id}>
        <header className="font-option__head"><span>{option.id}</span><h2>{option.name}</h2></header>
        <div className={`font-option__sample ${option.fontClass}`}>
          <VariedText className="font-option__title">AutoLabs</VariedText>
          <VariedText className="font-option__experiment">Experiment 3A</VariedText>
          <VariedText className="font-option__line">signal / feature / context</VariedText>
        </div>
        <p>{option.description}</p>
        <div className="font-option__test"><VariedText>aa 33 // SAE</VariedText></div>
      </article>)}
    </section>

    <footer className="font-options__footer">
      <p><strong>What is real here:</strong> the fonts are real; the per-letter drift is a prototype of the variation effect.</p>
      <p><strong>Next step if one wins:</strong> build or license a font with real alternate glyphs, then use OpenType contextual and stylistic alternates where the browser supports them.</p>
    </footer>
  </main>;
}
