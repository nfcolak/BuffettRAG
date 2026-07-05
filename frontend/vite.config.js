import { defineConfig } from 'vite'

export default defineConfig({
  base: './',
  server: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: [
      'f9b3d2d3f5a618893deb6fa98db6738.proxy-eu1.nuvolos.cloud',
      '.proxy-eu1.nuvolos.cloud'
    ],
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  },
  preview: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: [
      'f9b3d2d3f5a618893deb6fa98db6738.proxy-eu1.nuvolos.cloud',
      '.proxy-eu1.nuvolos.cloud'
    ],
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
