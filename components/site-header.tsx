'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';
import { TraceMark } from '@/components/trace-mark';

type SiteHeaderProps = {
  tone?: 'paper' | 'dark';
  contextLabel?: string;
  trailing?: ReactNode;
};

type Destination = { href: string; label: string; description: string; keywords: string; external?: boolean };

const researchDestinations: Destination[] = [
  { href: '/experiments/erdos-885', label: 'Erdős 885', description: 'The combinatorics benchmark and live lab', keywords: 'erdos 885 math benchmark lab' },
  { href: '/experiments/reward-compatibility', label: 'Reward Compatibility', description: 'Compatibility reward experiments and results', keywords: 'reward compatibility categories results' },
  { href: '/experiments/persona-discovery', label: 'Persona Discovery', description: 'Persona direction studies and evidence', keywords: 'persona discovery steering evidence' },
  { href: '/experiments/persona-discovery-3c', label: 'Experiment 3C', description: 'SAE pipeline walkthrough, visual run replay and the planned full run', keywords: 'experiment 3c sae sparse autoencoder walkthrough replay runpod' },
];

const destinations: Destination[] = [
  { href: '/', label: 'Lab', description: 'The live AutoLabs room', keywords: 'home lab field' },
  { href: '/events', label: 'Events', description: 'Public protocols and sponsored runs', keywords: 'protocols grants sponsored run' },
  { href: '/studio', label: 'Studio', description: 'Configure an experiment', keywords: 'create configure builder workbench' },
  { href: 'https://github.com/RaphaelKhalid/autolabs', label: 'Source', description: 'AutoLabs on GitHub', keywords: 'github code repository', external: true },
];
const directorDestinations = [...destinations.slice(0, 2), ...researchDestinations, ...destinations.slice(2)];

function isCurrent(pathname: string, href: string) {
  return href === '/' ? pathname === '/' : pathname === href || pathname.startsWith(`${href}/`);
}

export function SiteHeader({ tone = 'paper', contextLabel, trailing }: SiteHeaderProps) {
  const pathname = usePathname() ?? '/';
  const [open, setOpen] = useState(false);
  const [researchOpen, setResearchOpen] = useState(false);
  const [query, setQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const researchButtonRef = useRef<HTMLButtonElement>(null);
  const researchMenuRef = useRef<HTMLDivElement>(null);
  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return directorDestinations;
    return directorDestinations.filter((destination) => `${destination.label} ${destination.description} ${destination.keywords}`.toLowerCase().includes(needle));
  }, [query]);
  const researchCurrent = researchDestinations.some((destination) => isCurrent(pathname, destination.href));

  useEffect(() => {
    function onPointerDown(event: PointerEvent) {
      if (researchOpen && researchMenuRef.current && !researchMenuRef.current.contains(event.target as Node)) setResearchOpen(false);
    }
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, [researchOpen]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        if (researchOpen) { setResearchOpen(false); researchButtonRef.current?.focus(); }
        if (open) setOpen(false);
      }
      if (event.key === '/' && !['INPUT', 'TEXTAREA'].includes((event.target as HTMLElement)?.tagName)) {
        event.preventDefault();
        setOpen(true);
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, researchOpen]);

  useEffect(() => {
    if (open) window.setTimeout(() => inputRef.current?.focus(), 0);
    else setQuery('');
  }, [open]);

  return (
    <header className={`site-header site-header--${tone}`}>
      <Link className="site-header__brand" href="/" aria-label="AutoLabs home">
        <TraceMark size={34} label={false} />
        <span>AUTOLABS</span>
        {contextLabel && <small>{contextLabel}</small>}
      </Link>

      <nav className="site-header__nav" aria-label="Primary">
        <Link className={isCurrent(pathname, '/') ? 'site-header__current' : undefined} aria-current={isCurrent(pathname, '/') ? 'page' : undefined} href="/">Lab</Link>
        <Link className={isCurrent(pathname, '/events') ? 'site-header__current' : undefined} aria-current={isCurrent(pathname, '/events') ? 'page' : undefined} href="/events">Events</Link>
        <div className="site-header__dropdown" ref={researchMenuRef}>
          <button ref={researchButtonRef} className={researchCurrent ? 'site-header__current site-header__dropdown-trigger' : 'site-header__dropdown-trigger'} type="button" aria-haspopup="menu" aria-expanded={researchOpen} aria-controls="research-menu" onClick={() => setResearchOpen((value) => !value)}>
            Research <ChevronDown size={12} aria-hidden="true" />
          </button>
          {researchOpen && <div className="site-header__dropdown-menu" id="research-menu" role="menu" aria-label="Research families">
            {researchDestinations.map((destination) => <Link role="menuitem" href={destination.href} key={destination.href} onClick={() => setResearchOpen(false)}>{destination.label}</Link>)}
          </div>}
        </div>
        <Link className={isCurrent(pathname, '/studio') ? 'site-header__current' : undefined} aria-current={isCurrent(pathname, '/studio') ? 'page' : undefined} href="/studio">Studio</Link>
        <a href="https://github.com/RaphaelKhalid/autolabs" target="_blank" rel="noreferrer">Source ↗</a>
      </nav>

      <div className="site-header__actions">
        {trailing}
        <button className="site-header__director" type="button" aria-expanded={open} aria-controls="site-director" onClick={() => setOpen((value) => !value)}>
          <span className="site-header__director-dot" aria-hidden="true" />
          Director
          <kbd>/</kbd>
        </button>
      </div>

      {open && <div className="site-director" id="site-director" role="dialog" aria-label="AutoLabs director">
        <div className="site-director__top"><span>DIRECTOR / ROUTE INDEX</span><button type="button" onClick={() => setOpen(false)} aria-label="Close director">Close <kbd>Esc</kbd></button></div>
        <label className="site-director__search"><span>Search the lab</span><input ref={inputRef} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Try “protocol”, “results”, or “studio”" /></label>
        <div className="site-director__results">
          {matches.length ? matches.map((destination) => {
            const current = isCurrent(pathname, destination.href);
            const content = <><span className="site-director__index">{String(directorDestinations.indexOf(destination) + 1).padStart(2, '0')}</span><span><b>{destination.label}{current ? ' · current' : ''}</b><small>{destination.description}</small></span><span className="site-director__arrow">{destination.external ? '↗' : '→'}</span></>;
            return destination.external
              ? <a className="site-director__result" href={destination.href} target="_blank" rel="noreferrer" key={destination.href}>{content}</a>
              : <Link className="site-director__result" href={destination.href} onClick={() => setOpen(false)} key={destination.href}>{content}</Link>;
          }) : <p className="site-director__empty">No destinations match “{query}”.</p>}
        </div>
        <p className="site-director__hint">Press <kbd>/</kbd> anywhere to open the director. Research groups the three registered families.</p>
      </div>}
    </header>
  );
}
