import { useState, useEffect, useRef } from "react"
import { Search, Zap, ChevronDown, Download, AlertTriangle, Info, Loader2, Sparkles } from "lucide-react"
import { api } from "../lib/api"

const SPEC_COLORS = {
  Torque:    { bg: "bg-orange-500/10",  text: "text-orange-400",  dot: "bg-orange-400" },
  Dimension: { bg: "bg-blue-500/10",    text: "text-blue-400",    dot: "bg-blue-400" },
  Alignment: { bg: "bg-purple-500/10",  text: "text-purple-400",  dot: "bg-purple-400" },
  Pressure:  { bg: "bg-green-500/10",   text: "text-green-400",   dot: "bg-green-400" },
  Other:     { bg: "bg-zinc-500/10",    text: "text-zinc-400",    dot: "bg-zinc-400" },
}

function SpecBadge({ type }) {
  const c = SPEC_COLORS[type] || SPEC_COLORS.Other
  return (
    <span className={`inline-flex items-center gap-1.5 text-[11px] font-mono px-2 py-0.5 rounded-full ${c.bg} ${c.text}`}>
      <span className={`w-1 h-1 rounded-full ${c.dot}`} />
      {type}
    </span>
  )
}

function SourceBadge({ source }) {
  return (
    <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
      source === "table" ? "bg-teal-500/10 text-teal-400" :
      source === "llm"   ? "bg-violet-500/10 text-violet-400" :
                           "bg-zinc-500/10 text-zinc-400"
    }`}>
      {source}
    </span>
  )
}

function ResultRow({ r, i }) {
  return (
    <tr className="border-b border-surface-border hover:bg-surface-2/50 transition-colors animate-fade-in group"
        style={{ animationDelay: `${i * 30}ms` }}>
      <td className="px-4 py-3">
        <div className="flex items-start gap-2">
          {r.is_conflict && <AlertTriangle size={12} className="text-yellow-400 mt-0.5 shrink-0" />}
          {r.is_safety_critical && !r.is_conflict && <AlertTriangle size={12} className="text-red-400 mt-0.5 shrink-0" />}
          <span className="text-sm text-ink-0">{r.component}</span>
        </div>
        {r.vehicle_variant?.length > 0 && (
          <div className="flex gap-1 mt-1">
            {r.vehicle_variant.map(v => (
              <span key={v} className="text-[10px] font-mono bg-surface-3 text-ink-2 px-1.5 py-0.5 rounded">{v}</span>
            ))}
          </div>
        )}
      </td>
      <td className="px-4 py-3"><SpecBadge type={r.spec_type} /></td>
      <td className="px-4 py-3 font-mono text-sm text-ink-0 tabular-nums">
        <span className="text-brand font-medium">{r.value}</span>
        <span className="text-ink-2 ml-1">{r.unit}</span>
      </td>
      <td className="px-4 py-3 text-xs text-ink-2 font-mono">{r.section_id}</td>
      <td className="px-4 py-3"><SourceBadge source={r.source} /></td>
      <td className="px-4 py-3 text-xs text-ink-2 font-mono">
        {(r.confidence * 100).toFixed(0)}%
      </td>
    </tr>
  )
}

function exportJSON(results, query) {
  const clean = results.map(r => ({ component: r.component, spec_type: r.spec_type, value: r.value, unit: r.unit }))
  const blob = new Blob([JSON.stringify(clean, null, 2)], { type: "application/json" })
  const a = document.createElement("a")
  a.href = URL.createObjectURL(blob)
  a.download = `specs_${query.slice(0, 20).replace(/\s+/g, "_")}.json`
  a.click()
}

function exportCSV(results) {
  const header = "component,spec_type,value,unit,section_id,vehicle_variant\n"
  const rows = results.map(r =>
    [r.component, r.spec_type, r.value, r.unit, r.section_id, (r.vehicle_variant||[]).join("|")].join(",")
  ).join("\n")
  const blob = new Blob([header + rows], { type: "text/csv" })
  const a = document.createElement("a")
  a.href = URL.createObjectURL(blob)
  a.download = "specs.csv"
  a.click()
}

export default function QueryPage() {
  const [query, setQuery]       = useState("")
  const [variant, setVariant]   = useState("")
  const [specOnly, setSpecOnly] = useState(true)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [result, setResult]     = useState(null)
  const [demos, setDemos]       = useState([])
  const inputRef = useRef()

  useEffect(() => {
    api.demoQueries().then(setDemos).catch(() => {})
    inputRef.current?.focus()
  }, [])

  async function handleSubmit(e) {
    e?.preventDefault()
    const q = query.trim()
    if (!q) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.extract({ query: q, variant: variant || null, spec_only: specOnly })
      setResult(res)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function useDemo(q) {
    setQuery(q)
    setTimeout(() => handleSubmit(), 50)
  }

  const specs = result?.results || []
  const meta  = result?.metadata || {}

  return (
    <div className="p-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-ink-0 tracking-tight">Extract Specifications</h1>
        <p className="text-sm text-ink-2 mt-1">Query the 2014 F-150 service manual using natural language</p>
      </div>

      {/* Query form */}
      <form onSubmit={handleSubmit} className="space-y-3 mb-8">
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-ink-2" />
            <input
              ref={inputRef}
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="e.g. torque for brake caliper bolts"
              className="w-full bg-surface-2 border border-surface-border rounded-xl pl-10 pr-4 py-3
                         text-sm text-ink-0 placeholder:text-ink-2 outline-none
                         focus:border-brand/60 focus:ring-1 focus:ring-brand/20 transition-all"
            />
          </div>

          {/* Variant dropdown */}
          <div className="relative">
            <select
              value={variant}
              onChange={e => setVariant(e.target.value)}
              className="appearance-none bg-surface-2 border border-surface-border rounded-xl
                         px-4 pr-8 py-3 text-sm text-ink-1 outline-none cursor-pointer
                         focus:border-brand/60 transition-all"
            >
              <option value="">All variants</option>
              <option value="RWD">RWD</option>
              <option value="4WD">4WD</option>
            </select>
            <ChevronDown size={13} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-ink-2 pointer-events-none" />
          </div>

          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="flex items-center gap-2 bg-brand hover:bg-brand-dim disabled:opacity-40
                       text-white px-5 py-3 rounded-xl text-sm font-medium transition-all
                       disabled:cursor-not-allowed"
          >
            {loading ? <Loader2 size={15} className="animate-spin" /> : <Zap size={15} />}
            {loading ? "Extracting…" : "Extract"}
          </button>
        </div>

        {/* Options row */}
        <div className="flex items-center gap-4 text-xs text-ink-2">
          <label className="flex items-center gap-1.5 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={specOnly}
              onChange={e => setSpecOnly(e.target.checked)}
              className="accent-brand w-3 h-3"
            />
            Spec index only
          </label>
          <span className="text-surface-border">·</span>
          <span>Llama 3.1 8B via Ollama · FAISS + BM25 · RRF fusion</span>
        </div>
      </form>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 bg-red-500/10 border border-red-500/20 rounded-xl p-4 mb-6 animate-fade-in">
          <AlertTriangle size={16} className="text-red-400 mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium text-red-400">Extraction failed</p>
            <p className="text-xs text-ink-2 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="animate-slide-up">
          {/* Stats bar */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4 text-xs text-ink-2">
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-brand" />
                <span className="text-ink-0 font-medium">{specs.length}</span> specs found
              </span>
              <span>{meta.chunks_retrieved} chunks retrieved</span>
              <span>{meta.elapsed_sec}s</span>
              {meta.fast_path_count > 0 && <span className="text-teal-400">{meta.fast_path_count} fast-path</span>}
              {meta.llm_path_count > 0  && <span className="text-violet-400">{meta.llm_path_count} LLM</span>}
            </div>

            {specs.length > 0 && (
              <div className="flex items-center gap-2">
                <button onClick={() => exportJSON(specs, query)}
                  className="flex items-center gap-1.5 text-xs text-ink-2 hover:text-ink-0 border border-surface-border
                             px-3 py-1.5 rounded-lg transition-all hover:border-surface-border/80 hover:bg-surface-2">
                  <Download size={12} /> JSON
                </button>
                <button onClick={() => exportCSV(specs)}
                  className="flex items-center gap-1.5 text-xs text-ink-2 hover:text-ink-0 border border-surface-border
                             px-3 py-1.5 rounded-lg transition-all hover:border-surface-border/80 hover:bg-surface-2">
                  <Download size={12} /> CSV
                </button>
              </div>
            )}
          </div>

          {specs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-center border border-surface-border rounded-xl bg-surface-1">
              <Info size={24} className="text-ink-2 mb-3" />
              <p className="text-sm text-ink-1">No specifications found</p>
              <p className="text-xs text-ink-2 mt-1">Try a more specific query or different variant</p>
            </div>
          ) : (
            <div className="border border-surface-border rounded-xl overflow-hidden bg-surface-1">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-surface-border bg-surface-2">
                    {["Component", "Type", "Value", "Section", "Source", "Conf."].map(h => (
                      <th key={h} className="px-4 py-3 text-[11px] font-semibold text-ink-2 uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {specs.map((r, i) => <ResultRow key={i} r={r} i={i} />)}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Demo queries */}
      {!result && !loading && demos.length > 0 && (
        <div className="mt-8">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles size={13} className="text-ink-2" />
            <span className="text-xs text-ink-2 font-medium uppercase tracking-wider">Try a demo query</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {demos.slice(0, 6).map((q, i) => (
              <button key={i} onClick={() => useDemo(q)}
                className="text-xs text-ink-2 hover:text-ink-0 border border-surface-border hover:border-brand/40
                           px-3 py-1.5 rounded-lg transition-all hover:bg-surface-2 bg-surface-1">
                {q}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
