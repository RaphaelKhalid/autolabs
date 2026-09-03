import type { Metadata } from 'next';
import { Geist_Mono, STIX_Two_Text } from 'next/font/google';
import './globals.css';

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
  title: 'Autolabs · Luna High Erdős 885 Competition',
  description: 'A public, exact and replayable autonomous mathematics experiment.',
  openGraph: {
    title: 'Autolabs · Experiment 885',
    description: 'Five mathematical intelligences. One open Erdős problem. Every claim verified.',
    type: 'website',
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${serif.variable} ${mono.variable}`}>{children}</body>
    </html>
  );
}
