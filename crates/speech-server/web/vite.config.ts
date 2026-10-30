import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  root: __dirname,
  build: {
    outDir: path.resolve(__dirname, 'dist'),
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/v1': 'http://localhost:8010',
      '/api': 'http://localhost:8010',
      '/healthz': 'http://localhost:8010',
    },
  },
});
