import Link from 'next/link';

const repo = 'https://github.com/RaphaelKhalid/autolabs';

export function AutolabsHome() {
  return <main className="archive-page home-page">
    <nav className="archive-nav home-nav" aria-label="Primary">
      <Link className="home-brand" href="/" aria-label="AutoLabs home"><span>A</span><strong>AUTOLABS</strong></Link>
      <div className="home-nav-links"><Link href="/experiments">Experiments</Link><Link href="/research">Research</Link><Link href="/journal">Journal</Link><Link href="/studio">Studio</Link><a href={repo} target="_blank" rel="noreferrer">Source ↗</a></div>
    </nav>

    <header className="archive-heading home-hero">
      <p className="archive-kicker">PUBLIC LABORATORY / AUTONOMOUS SYSTEMS</p>
      <h1>Experiments you can <em>inspect.</em></h1>
      <p>AutoLabs is a public record of agent experiments. Follow the question, the setup, the run and the evidence in one place.</p>
      <div className="home-actions"><Link className="archive-button home-primary" href="/experiments">Browse experiments ↗</Link><Link className="archive-button" href="/research">Open research atlas ↗</Link></div>
    </header>

    <section className="home-status archive-verdict" aria-label="Current lab status">
      <div><p className="archive-kicker">THE LAB / NOW</p><h2>A stable home for changing work.</h2><p>Experiments can start, finish or be superseded without changing the way you enter AutoLabs. The register is the source of truth for what is running and what has evidence attached.</p></div>
      <div className="home-status-mark"><span className="home-status-dot" /> OPEN RECORD</div>
    </section>

    <section className="home-grid" aria-label="AutoLabs destinations">
      <Link className="home-card" href="/experiments"><span className="archive-kicker">01 / REGISTER</span><h2>Experiments</h2><p>Methods, configurations, costs and results for each study.</p><span className="home-card-link">View the register ↗</span></Link>
      <Link className="home-card" href="/research"><span className="archive-kicker">02 / QUESTIONS</span><h2>Research</h2><p>Open questions and source records from the Afterlight atlas.</p><span className="home-card-link">Enter the atlas ↗</span></Link>
      <Link className="home-card" href="/journal"><span className="archive-kicker">03 / OBSERVATORY</span><h2>Lab journal</h2><p>The living field for active and archived agent work.</p><span className="home-card-link">Read the journal ↗</span></Link>
      <Link className="home-card" href="/studio"><span className="archive-kicker">04 / BUILDER</span><h2>Studio</h2><p>Compose a configuration and run it with your own credentials.</p><span className="home-card-link">Open the studio ↗</span></Link>
    </section>

    <section className="home-principles archive-section"><p className="archive-kicker">THE RECORD</p><h2>Every claim stays attached to its setup.</h2><p>AutoLabs keeps research questions, model choices, budgets and outputs together. Planned work stays visibly planned. Completed work links to the underlying record.</p><Link className="archive-button" href="/experiments">See what is documented ↗</Link></section>

    <footer><span>AUTOLABS / OPEN RECORD</span><p>Observable agent experiments</p><Link href="/experiments">Explore ↗</Link></footer>
  </main>;
}
