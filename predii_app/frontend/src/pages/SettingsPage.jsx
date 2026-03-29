import { useState, useEffect } from "react"
import { Settings, Save, RefreshCw, CheckCircle, AlertTriangle } from "lucide-react"
import { api } from "../lib/api"

function Field({ label, desc, children }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-sm font-medium text-ink-0">{label}</label>
      {desc && <p className="text-xs text-ink-2">{desc}</p>}
      {children}
    </div>
  )
}

function Input({ value, onChange, placeholder, type = "text" }) {
  return (
    <input
      type={type}
      value={value}
      onChange={e => onChange(e.target.value)}
      placeholder={placeholder}
      className="bg-surface-2 border border-surface-border rounded-xl px-4 py-3
                 text-sm text-ink-0 placeholder:text-ink-2 outline-none font-mono
                 focus:border-brand/60 focus:ring-1 focus:ring-brand/20 transition-all"
    />
  )
}

export default function SettingsPage() {
  const [config, setConfig]   = useState(null)
  const [saved, setSaved]     = useState(false)
  const [error, setError]     = useState(null)
  const [testing, setTesting] = useState(false)
  const [apiOk, setApiOk]     = useState(null)

  useEffect(() => {
    api.config().then(setConfig).catch(e => setError(e.message))
  }, [])

  function update(key) {
    return val => setConfig(prev => ({ ...prev, [key]: val }))
  }

  async function save() {
    setError(null)
    try {
      await api.updateConfig(config)
      setSaved(true)
      setTimeout(() => setSaved(false), 2500)
    } catch (e) {
      setError(e.message)
    }
  }

  async function testApi() {
    setTesting(true)
    setApiOk(null)
    try {
      await api.health()
      setApiOk(true)
    } catch {
      setApiOk(false)
    } finally {
      setTesting(false)
    }
  }

  if (!config) return (
    <div className="p-8 flex items-center justify-center py-20 text-ink-2">
      <div className="w-5 h-5 border-2 border-surface-border border-t-brand rounded-full animate-spin" />
    </div>
  )

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-ink-0 tracking-tight">Settings</h1>
        <p className="text-sm text-ink-2 mt-1">Configure the extraction pipeline</p>
      </div>

      <div className="space-y-6">
        {/* LLM section */}
        <div className="border border-surface-border rounded-xl p-6 bg-surface-1 space-y-5">
          <div className="flex items-center gap-2 pb-3 border-b border-surface-border">
            <Settings size={14} className="text-ink-2" />
            <span className="text-sm font-semibold text-ink-0">LLM Configuration</span>
          </div>

          <Field label="Model" desc="Ollama model name. Must be pulled locally with ollama pull.">
            <select value={config.model} onChange={e => update("model")(e.target.value)}
              className="bg-surface-2 border border-surface-border rounded-xl px-4 py-3
                         text-sm text-ink-0 outline-none font-mono focus:border-brand/60 transition-all cursor-pointer">
              <option value="llama3.1:8b">llama3.1:8b (recommended)</option>
              <option value="llama3.1:70b">llama3.1:70b (high quality)</option>
              <option value="mistral:7b">mistral:7b</option>
              <option value="mistral-nemo">mistral-nemo</option>
              <option value="qwen2.5:14b">qwen2.5:14b (best JSON)</option>
            </select>
          </Field>

          <Field label="Ollama Host" desc="URL where Ollama server is running.">
            <Input value={config.ollama_host} onChange={update("ollama_host")}
              placeholder="http://localhost:11434" />
          </Field>

          <div className="flex items-center gap-3 pt-1">
            <button onClick={testApi} disabled={testing}
              className="flex items-center gap-2 text-xs border border-surface-border px-3 py-2 rounded-lg
                         text-ink-2 hover:text-ink-0 hover:bg-surface-2 transition-all disabled:opacity-50">
              <RefreshCw size={12} className={testing ? "animate-spin" : ""} />
              Test connection
            </button>
            {apiOk === true  && <span className="flex items-center gap-1.5 text-xs text-green-400"><CheckCircle size={12} /> Connected</span>}
            {apiOk === false && <span className="flex items-center gap-1.5 text-xs text-red-400"><AlertTriangle size={12} /> Cannot reach API</span>}
          </div>
        </div>

        {/* Retrieval section */}
        <div className="border border-surface-border rounded-xl p-6 bg-surface-1 space-y-5">
          <div className="flex items-center gap-2 pb-3 border-b border-surface-border">
            <Settings size={14} className="text-ink-2" />
            <span className="text-sm font-semibold text-ink-0">Retrieval</span>
          </div>

          <Field label="Index directory" desc="Path to the phase2_index folder built by chunker_embedder.py.">
            <Input value={config.index_dir} onChange={update("index_dir")}
              placeholder="./phase2_index" />
          </Field>

          <Field label="Top-k chunks" desc="Number of chunks to retrieve per query (1–20). Higher = more context, slower.">
            <div className="flex items-center gap-4">
              <input type="range" min={1} max={20} value={config.k} onChange={e => update("k")(parseInt(e.target.value))}
                className="flex-1 accent-brand" />
              <span className="text-sm font-mono text-brand w-6 text-right">{config.k}</span>
            </div>
            <p className="text-xs text-ink-2 mt-1">
              Recommended: 5 for precision, 10 for recall, 3 for speed
            </p>
          </Field>
        </div>

        {/* Pipeline info */}
        <div className="border border-surface-border rounded-xl p-6 bg-surface-1">
          <div className="flex items-center gap-2 pb-3 mb-4 border-b border-surface-border">
            <span className="text-sm font-semibold text-ink-0">Pipeline info</span>
          </div>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-xs">
            {[
              ["Embedding model",  "BAAI/bge-base-en-v1.5 (768-dim)"],
              ["Vector store",     "FAISS IndexFlatIP (cosine)"],
              ["Keyword search",   "BM25Okapi (rank_bm25)"],
              ["Fusion",           "Reciprocal Rank Fusion (k=60)"],
              ["Fast path",        "Structured spec_value → no LLM"],
              ["LLM path",         "Ollama JSON mode, temp=0"],
            ].map(([k, v]) => (
              <>
                <dt key={k} className="text-ink-2">{k}</dt>
                <dd key={v} className="text-ink-0 font-mono">{v}</dd>
              </>
            ))}
          </dl>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-sm">
            <AlertTriangle size={14} className="text-red-400 mt-0.5 shrink-0" />
            <span className="text-red-400">{error}</span>
          </div>
        )}

        {/* Save */}
        <button onClick={save}
          className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-medium transition-all
            ${saved ? "bg-green-500/20 text-green-400 border border-green-500/30" : "bg-brand hover:bg-brand-dim text-white"}`}>
          {saved ? <><CheckCircle size={15} /> Saved</> : <><Save size={15} /> Save settings</>}
        </button>
      </div>
    </div>
  )
}
