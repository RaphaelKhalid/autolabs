'use client';

import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { AFTERLIGHT_ORIGIN, AFTERLIGHT_URL, normalizeChildPath, parseAfterlightMessage, type AfterlightQuestion, type AutolabsNavigateMessage } from '@/lib/afterlight-contract';

const DEFAULT_PATH = '/atlas';
function researchUrl(path: string) { return `/research?${new URLSearchParams({ view: path }).toString()}`; }
function studioUrl(question: AfterlightQuestion) { return `/studio?${new URLSearchParams({ question: question.id, title: question.title, source: question.sourceUrl }).toString()}`; }

export function ResearchSurface({ initialPath }: { initialPath: string }) {
  const frame = useRef<HTMLIFrameElement>(null);
  const frameSrc = useRef(`${AFTERLIGHT_URL}#${normalizeChildPath(initialPath) ?? DEFAULT_PATH}`).current;
  const [path, setPath] = useState(() => normalizeChildPath(initialPath) ?? DEFAULT_PATH);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [retry, setRetry] = useState(0);
  const [childReady, setChildReady] = useState(false);

  useEffect(() => {
    const timer = window.setTimeout(() => setStatus((current) => current === 'ready' ? current : 'error'), 12000);
    return () => window.clearTimeout(timer);
  }, [retry]);

  useEffect(() => {
    const onMessage = (event: MessageEvent<unknown>) => {
      const currentFrame = frame.current;
      if (event.origin !== AFTERLIGHT_ORIGIN || !currentFrame || event.source !== currentFrame.contentWindow) return;
      const message = parseAfterlightMessage(event.data);
      if (!message) return;
      if (message.type === 'afterlight:ready') { setChildReady(true); setStatus('ready'); return; }
      if (message.type === 'afterlight:navigation') {
        setPath(message.path);
        const nextUrl = researchUrl(message.path);
        if (window.location.pathname + window.location.search !== nextUrl) window.history.pushState(window.history.state, '', nextUrl);
        return;
      }
      window.location.assign(studioUrl(message.question));
    };
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, []);

  useEffect(() => {
    const onPopState = () => {
      const view = new URL(window.location.href).searchParams.get('view');
      setPath(normalizeChildPath(view) ?? DEFAULT_PATH);
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  useEffect(() => {
    if (!childReady) return;
    const navigate: AutolabsNavigateMessage = { type: 'autolabs:navigate', version: 1, path };
    frame.current?.contentWindow?.postMessage(navigate, AFTERLIGHT_ORIGIN);
  }, [childReady, path]);

  function retryFrame() { setChildReady(false); setStatus('loading'); setRetry((value) => value + 1); }

  return <main className="research-page">
    <header className="research-shell-nav">
      <Link className="research-shell-brand" href="/" aria-label="Auto Labs home"><span>A</span><strong>AUTOLABS</strong></Link>
      <nav aria-label="Primary navigation"><Link href="/studio">Studio</Link><Link href="/experiments">Experiments</Link><Link className="is-active" href="/research" aria-current="page">Research</Link><Link href="/journal">Journal</Link></nav>
      <span className="research-shell-state"><i /> Research library</span>
    </header>

    <section className="research-compact-head"><div><p className="research-kicker">AFTERLIGHT</p><h1>Research</h1></div><div className="research-head-action"><a href={`${AFTERLIGHT_ORIGIN}#${path}`} target="_blank" rel="noreferrer">Open standalone ↗</a></div></section>
    <section className="research-frame-section" aria-label="Afterlight research application"><div className={`research-frame ${status === 'error' ? 'has-error' : ''}`}><iframe key={retry} ref={frame} title="Afterlight research atlas" src={frameSrc} onLoad={() => setStatus((current) => current === 'ready' || current === 'error' ? current : 'loading')} onError={() => setStatus('error')} />{status === 'loading' && <div className="research-frame-overlay" role="status"><span className="research-loader" /><strong>Loading research library</strong><p>Opening questions and their supporting evidence.</p></div>}{status === 'error' && <div className="research-frame-overlay research-frame-error" role="alert"><span className="research-error-mark">!</span><strong>Afterlight is unavailable right now</strong><p>The standalone research atlas may still be reachable.</p><div><button type="button" onClick={retryFrame}>Try again</button><a href={`${AFTERLIGHT_ORIGIN}#${path}`} target="_blank" rel="noreferrer">Open standalone ↗</a></div></div>}</div></section>
  </main>;
}