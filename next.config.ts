import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  ...(process.env.AUTOLABS_STANDALONE === '1' ? { output: 'standalone' as const } : {}),
  experimental: { optimizePackageImports: ['lucide-react'] },
};

export default nextConfig;
