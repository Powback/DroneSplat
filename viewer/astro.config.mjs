import { defineConfig } from 'astro/config';

// Static build — served by nginx behind Traefik. The viewer loads PlayCanvas +
// the Streamed SOG at runtime (CDN + /splat), so nothing heavy to bundle.
export default defineConfig({ output: 'static' });
