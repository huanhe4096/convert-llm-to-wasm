import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

function normalizeBasePath(path) {
  if (!path || path === '/') return '/'
  return `/${path.replace(/^\/|\/$/g, '')}/`
}

// https://vite.dev/config/
export default defineConfig(() => {
  const base = normalizeBasePath(process.env.BASE_PATH)

  return {
    base,
    plugins: [vue()],
  }
})
