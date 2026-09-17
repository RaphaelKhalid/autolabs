import { defineConfig } from 'vitest/config';

// Local config so vitest does not walk up to the repository root's
// vitest.config.ts, which imports from the root node_modules; in CI the
// worker is verified before the root install exists.
export default defineConfig({
  test: {
    include: ['tests/**/*.test.{ts,mjs,js}'],
    exclude: ['**/node_modules/**'],
  },
});
