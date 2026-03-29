import { useState, useEffect } from "react"
import { Clock, Trash2, RotateCcw, Search, ChevronDown, ChevronRight, AlertTriangle } from "lucide-react"
import { api } from "../lib/api"
import { useNavigate } from "react-router-dom"

function SpecPill({ r }) {
  return (
    <span className="inline-flex items-center gap-1 text-[11px] font-mono bg-surface-3 text-ink-1 px-2 py-0.5 rounded">
      <span className="text-brand">{r.value}</span>
      <span className="text-ink-2">{r.unit}</span>
    </span>
  )
}

function HistoryCard({ item, onDelete }) {
  const [expanded, setExpanded] = useState(false)
  const navigate = useNavigate()

  const specs   = item.results || []
  const elapsed = item.metadata?.elapsed_sec
  const ts      = new Date(item.timestamp).toLocaleString()

  function rerun() {
    navigate("/", { state: { query: item.query, variant: item.variant } })
  }

  return (
    <div className="border border-surface-border rounded-xl bg-surface-1 overflow-hidden animate-fade-in">
      <div className="px-5 py-4">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-sm font-medium text-ink-0 truncate">{item.query}</span>
              {item.variant && (
                <span className="text-[10px] font-mono bg-brand/10 text-brand px-1.5 py-0.5 rounded shrink-0">
                  {item.variant}
                </span>
              )}
            </div>
            <div className="flex items-center gap-3 text-xs text-ink-2">
              <span className="flex items-center gap-1"><Clock size={10} />{ts}</span>
              <span>{specs.length} specs</span>
              {elapsed && <span>{elapsed}s</span>}
            </div>
          </div>

          <div className="flex items-center gap-1.5 shrink-0">
            <button onClick={rerun}
              className="p-1.5 rounded-lg hover:bg-surface-3 text-ink-2 hover:text-ink-0 transition-all"
              title="Re-run query">
              <RotateCcw size={13} />
            </button>
            <button onClick={() => onDelete(item.id)}
              className="p-1.5 rounded-lg hover:bg-red-500/10 text-ink-2 hover:text-red-400 transition-all"
              title="Delete">
              <Trash2 size={13} />
            </button>
            <button onClick={() => setExpanded(!expanded)}
              className="p-1.5 rounded-lg hover:bg-surface-3 text-ink-2 transition-all">
              {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
            </button>
          </div>
        </div>

        {/* Quick preview pills */}
        {!expanded && specs.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-3">
            {specs.slice(0, 4).map((r, i) => <SpecPill key={i} r={r} />)}
            {specs.length > 4 && (
              <span className="text-[11px] text-ink-2">+{specs.length - 4} more</span>
            )}
          </div>
        )}
      </div>

      {/* Expanded results table */}
      {expanded && specs.length > 0 && (
        <div className="border-t border-surface-border">
          <table className="w-full text-left">
            <thead>
              <tr className="bg-surface-2">
                {["Component", "Type", "Value", "Section", "Source"].map(h => (
                  <th key={h} className="px-4 py-2.5 text-[11px] font-semibold text-ink-2 uppercase tracking-wider">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {specs.map((r, i) => (
                <tr key={i} className="border-t border-surface-border hover:bg-surface-2/40 transition-colors">
                  <td className="px-4 py-2.5 text-xs text-ink-0">
                    <div className="flex items-center gap-1.5">
                      {r.is_conflict && <AlertTriangle size={11} className="text-yellow-400" />}
                      {r.component}
                    </div>
                  </td>
                  <td className="px-4 py-2.5">
                    <span className="text-[11px] font-mono text-ink-1">{r.spec_type}</span>
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs">
                    <span className="text-brand">{r.value}</span>
                    <span className="text-ink-2 ml-1">{r.unit}</span>
                  </td>
                  <td className="px-4 py-2.5 text-xs font-mono text-ink-2">{r.section_id}</td>
                  <td className="px-4 py-2.5">
                    <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                      r.source === "table" ? "bg-teal-500/10 text-teal-400" : "bg-violet-500/10 text-violet-400"
                    }`}>{r.source}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {expanded && specs.length === 0 && (
        <div className="px-5 py-4 border-t border-surface-border text-xs text-ink-2">
          No specifications were extracted for this query.
        </div>
      )}
    </div>
  )
}

export default function HistoryPage() {
  const [items, setItems]   = useState([])
  const [search, setSearch] = useState("")
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.history(100)
      .then(setItems)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  async function handleDelete(id) {
    await api.deleteItem(id).catch(console.error)
    setItems(prev => prev.filter(h => h.id !== id))
  }

  async function handleClear() {
    if (!confirm("Clear all history?")) return
    await api.clearHistory().catch(console.error)
    setItems([])
  }

  const filtered = items.filter(h =>
    !search || h.query.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-ink-0 tracking-tight">History</h1>
          <p className="text-sm text-ink-2 mt-1">{items.length} queries this session</p>
        </div>
        {items.length > 0 && (
          <button onClick={handleClear}
            className="flex items-center gap-1.5 text-xs text-red-400 hover:text-red-300 border border-red-500/20
                       hover:border-red-500/40 px-3 py-2 rounded-lg transition-all hover:bg-red-500/5">
            <Trash2 size={12} /> Clear all
          </button>
        )}
      </div>

      {items.length > 0 && (
        <div className="relative mb-5">
          <Search size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-ink-2" />
          <input value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Filter queries…"
            className="w-full bg-surface-2 border border-surface-border rounded-xl pl-9 pr-4 py-2.5
                       text-sm text-ink-0 placeholder:text-ink-2 outline-none focus:border-brand/60 transition-all"
          />
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center py-20 text-ink-2">
          <div className="w-5 h-5 border-2 border-surface-border border-t-brand rounded-full animate-spin" />
        </div>
      )}

      {!loading && filtered.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <Clock size={28} className="text-ink-2 mb-3" />
          <p className="text-sm text-ink-1">{search ? "No matching queries" : "No history yet"}</p>
          <p className="text-xs text-ink-2 mt-1">
            {search ? "Try a different search term" : "Run a query on the Extract page to see it here"}
          </p>
        </div>
      )}

      <div className="space-y-3">
        {filtered.map(item => (
          <HistoryCard key={item.id} item={item} onDelete={handleDelete} />
        ))}
      </div>
    </div>
  )
}
