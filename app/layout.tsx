import type { Metadata } from 'next';
import { Analytics } from '@vercel/analytics/next';
import { Geist_Mono, STIX_Two_Text } from 'next/font/google';
import './globals.css';
import './report.css';
import './lab.css';
import './archive.css';

const serif = STIX_Two_Text({
  subsets: ['latin'],
  variable: '--font-serif',
  display: 'swap',
});

const mono = Geist_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'AutoLabs · Observable agent experiments',
  description: 'A public laboratory for agent experiments, with per-experiment models, methods, costs and evidence. Explore the completed Erdős 885 pilot.',
  openGraph: {
    title: 'AutoLabs · Observable agent experiments',
    description: 'The animated lab, an open experiment register, and the results of our 100-round Erdős 885 pilot.',
    type: 'website',
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${serif.variable} ${mono.variable}`}>
        {children}
        <Analytics />
      </body>
    </html>
  );
}
