import { loadEnvFile } from 'node:process';
import { createRequire } from 'node:module';
// Load configuration without forwarding --env-file into Next.js worker arguments.
loadEnvFile('.env.selfhost.local');
process.argv=[process.argv[0],'next',...process.argv.slice(2)];
createRequire(import.meta.url)('next/dist/bin/next');
