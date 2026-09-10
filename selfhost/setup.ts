import { randomBytes } from 'node:crypto';
import { existsSync, writeFileSync } from 'node:fs';
const file='.env.selfhost.local';
if(existsSync(file)){console.log('Self-hosted environment already exists; no changes made.');}
else{
  writeFileSync(file,[
    '# Private self-hosted configuration. Never commit this file.',
    'AUTOLABS_SELF_HOSTED=1',
    `AUTOLABS_LOCAL_TOKEN=${randomBytes(32).toString('hex')}`,
    'AUTOLABS_RUNNER_URL=http://127.0.0.1:8788',
    'AUTOLABS_RUNNER_PORT=8788',
    'AUTOLABS_DATA_DIR=.autolabs',
    `OPENROUTER_API_KEY=${process.env.OPENROUTER_API_KEY??''}`,
    '',
  ].join('\n'),{mode:0o600,flag:'wx'});
  console.log('Created .env.selfhost.local. The owner token and optional provider key are not printed.');
}
