const BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

async function req(path, options = {}) {
  const r = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try {
      const j = await r.json()
      msg = j.detail || msg
    } catch (_) {}
    throw new Error(msg)
  }
  return r.json()
}

export const api = {
  health: () => req('/health'),
  extract: (payload) => req('/api/extract', { method: 'POST', body: JSON.stringify(payload) }),
  history: (limit = 50) => req(`/api/history?limit=${limit}`),
  clearHistory: () => req('/api/history', { method: 'DELETE' }),
  deleteItem: (id) => req(`/api/history/${id}`, { method: 'DELETE' }),
  config: () => req('/api/config'),
  updateConfig: (payload) => req('/api/config', { method: 'PATCH', body: JSON.stringify(payload) }),
  demoQueries: () => req('/api/demo-queries'),
}
