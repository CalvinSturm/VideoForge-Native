import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  optimizeDeps: {
    exclude: ['electron'], // Electron is external in renderer
  },
  build: {
    rollupOptions: {
      external: ['electron'], // keep electron as external to avoid bundling it
    },
  },
});
