import React, { useState, useRef, useCallback, useEffect } from 'react'
import {
  Upload, Search, Zap, AlertTriangle, CheckCircle2,
  ChevronDown, Download, X, FileText, Cpu, Database,
  RefreshCw, Wrench, Clock, Shield, Activity, Trash2,
  ChevronRight, ArrowRight, Settings
} from 'lucide-react'

const API = '/api'

// ── helpers ───────────────────────────────────────────────────────────────────
const fmt = (v) => {
  if (v == null) return '—'
  return String(v).replace('.0', '')
}

const unitColor = (unit) => {
  if (!unit) return '#9999aa'
  const u = unit.toLowerCase()
  if (u.includes('nm'))   return '#f97316'
  if (u.includes('lb-ft')) return '#f59e0b'
  if (u.includes('lb-in')) return '#fbbf24'
  if (u.includes('mm') || u.includes('in')) return '#14b8a6'
  if (u.includes('deg')) return '#3b82f6'
  return '#9999aa'
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms))

// ── Gear SVG background decoration ───────────────────────────────────────────
function GearBg() {
  return (
    <svg style={{position:'fixed',top:0,left:0,width:'100%',height:'100%',pointerEvents:'none',zIndex:0,opacity:0.025}} viewBox="0 0 800 600" preserveAspectRatio="xMidYMid slice">
      <g transform="translate(680,80)" opacity="0.6">
        <circle r="90" fill="none" stroke="#f97316" strokeWidth="8"/>
        <circle r="68" fill="none" stroke="#f97316" strokeWidth="3"/>
        {[...Array(12)].map((_,i) => {
          const a = (i/12)*Math.PI*2
          const x1=Math.cos(a)*82, y1=Math.sin(a)*82
          const x2=Math.cos(a)*98, y2=Math.sin(a)*98
          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#f97316" strokeWidth="10" strokeLinecap="square"/>
        })}
        <circle r="20" fill="none" stroke="#f97316" strokeWidth="6"/>
      </g>
      <g transform="translate(120,480)" opacity="0.4">
        <circle r="60" fill="none" stroke="#f97316" strokeWidth="6"/>
        {[...Array(8)].map((_,i) => {
          const a = (i/8)*Math.PI*2
          const x1=Math.cos(a)*54, y1=Math.sin(a)*54
          const x2=Math.cos(a)*66, y2=Math.sin(a)*66
          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#f97316" strokeWidth="8" strokeLinecap="square"/>
        })}
        <circle r="14" fill="none" stroke="#f97316" strokeWidth="5"/>
      </g>
      <g transform="translate(400,300)" opacity="0.15">
        <circle r="140" fill="none" stroke="#f97316" strokeWidth="1" strokeDasharray="4 8"/>
      </g>
    </svg>
  )
}

// ── Topbar ─────────────────────────────────────────────────────────────────────
function Topbar({ apiOk, model }) {
  return (
    <header style={{
      display:'flex', alignItems:'center', justifyContent:'space-between',
      padding:'0 32px', height:56,
      background:'rgba(13,13,18,0.92)',
      backdropFilter:'blur(12px)',
      borderBottom:'1px solid rgba(249,115,22,0.15)',
      position:'sticky', top:0, zIndex:100,
    }}>
      <div style={{display:'flex',alignItems:'center',gap:12}}>
        <div style={{
          width:32, height:32,
          background:'linear-gradient(135deg,#f97316,#ea580c)',
          borderRadius:6, display:'flex', alignItems:'center', justifyContent:'center',
        }}>
          <Wrench size={16} color="#fff"/>
        </div>
        <span style={{fontFamily:'var(--display)',fontSize:18,fontWeight:700,letterSpacing:'0.04em',color:'#fff'}}>
          PREDII<span style={{color:'var(--orange)',marginLeft:4}}>SPEC</span>
        </span>
        <span style={{
          marginLeft:8, padding:'2px 8px',
          background:'rgba(249,115,22,0.1)',
          border:'1px solid rgba(249,115,22,0.25)',
          borderRadius:4, fontSize:10,
          fontFamily:'var(--mono)', color:'var(--orange)',
          letterSpacing:'0.08em'
        }}>v1.0</span>
      </div>

      <div style={{display:'flex',alignItems:'center',gap:20}}>
        <div style={{display:'flex',alignItems:'center',gap:6,fontSize:12,color:'var(--text2)',fontFamily:'var(--mono)'}}>
          <div style={{
            width:6, height:6, borderRadius:'50%',
            background: apiOk ? '#22c55e' : '#ef4444',
            animation: apiOk ? 'pulse-ring 2s infinite' : 'none',
          }}/>
          {apiOk ? 'API connected' : 'API offline'}
        </div>
        {model && (
          <div style={{display:'flex',alignItems:'center',gap:6,fontSize:12,color:'var(--text2)',fontFamily:'var(--mono)'}}>
            <Cpu size={12} color="var(--orange)"/>
            {model}
          </div>
        )}
        <div style={{display:'flex',alignItems:'center',gap:6,fontSize:12,color:'var(--text2)',fontFamily:'var(--mono)'}}>
          <Database size={12} color="#14b8a6"/>
          FAISS · BM25 · RRF
        </div>
      </div>
    </header>
  )
}

// ── Upload zone ───────────────────────────────────────────────────────────────
function UploadZone({ onUpload, processing }) {
  const [drag, setDrag] = useState(false)
  const input = useRef()

  const handle = useCallback(async (file) => {
    if (!file) return
    if (!file.name.match(/\.(pdf|txt)$/i)) {
      alert('Upload a PDF or TXT file')
      return
    }
    onUpload(file)
  }, [onUpload])

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files[0]) }}
      onClick={() => !processing && input.current?.click()}
      style={{
        position:'relative',
        border: drag ? '2px solid var(--orange)' : '2px dashed rgba(249,115,22,0.25)',
        borderRadius:12,
        padding:'48px 32px',
        textAlign:'center',
        cursor: processing ? 'not-allowed' : 'pointer',
        transition:'all 0.2s',
        background: drag ? 'rgba(249,115,22,0.05)' : 'rgba(249,115,22,0.02)',
        overflow:'hidden',
      }}
    >
      {/* Scan line when processing */}
      {processing && (
        <div style={{
          position:'absolute', left:0, right:0, height:2,
          background:'linear-gradient(90deg,transparent,var(--orange),transparent)',
          animation:'scan 1.5s linear infinite',
        }}/>
      )}

      <input ref={input} type="file" accept=".pdf,.txt" style={{display:'none'}}
             onChange={e => handle(e.target.files[0])} />

      <div style={{
        width:56, height:56, margin:'0 auto 16px',
        background:'rgba(249,115,22,0.1)',
        border:'1px solid rgba(249,115,22,0.25)',
        borderRadius:12,
        display:'flex', alignItems:'center', justifyContent:'center',
        transition:'all 0.2s',
        transform: drag ? 'scale(1.1)' : 'scale(1)',
      }}>
        {processing
          ? <RefreshCw size={24} color="var(--orange)" style={{animation:'spin 1s linear infinite'}}/>
          : <Upload size={24} color="var(--orange)"/>
        }
      </div>

      <p style={{fontFamily:'var(--display)',fontSize:18,fontWeight:700,letterSpacing:'0.04em',color:drag?'var(--orange)':'var(--text)',marginBottom:6}}>
        {processing ? 'PROCESSING DOCUMENT' : drag ? 'DROP TO UPLOAD' : 'UPLOAD SERVICE MANUAL'}
      </p>
      <p style={{fontSize:13,color:'var(--text2)'}}>
        {processing ? 'Extracting text, chunking, building vector index…' : 'PDF or TXT — any vehicle service manual'}
      </p>
    </div>
  )
}

// ── Pipeline progress ─────────────────────────────────────────────────────────
function PipelineProgress({ stage }) {
  const stages = [
    { key:'extract',  label:'PDF Extraction',    icon:<FileText size={14}/> },
    { key:'chunk',    label:'Chunking + Embed',   icon:<Cpu size={14}/> },
    { key:'index',    label:'FAISS + BM25 Index', icon:<Database size={14}/> },
    { key:'ready',    label:'Ready to Query',     icon:<CheckCircle2 size={14}/> },
  ]
  const idx = { extract:0, chunk:1, index:2, ready:3 }[stage] ?? -1

  return (
    <div style={{display:'flex',alignItems:'center',gap:0,marginTop:20}}>
      {stages.map((s,i) => (
        <React.Fragment key={s.key}>
          <div style={{
            display:'flex',alignItems:'center',gap:6,
            padding:'6px 14px', borderRadius:6, fontSize:12,
            fontFamily:'var(--mono)',
            background: i===idx ? 'rgba(249,115,22,0.15)' : i<idx ? 'rgba(34,197,94,0.1)' : 'rgba(255,255,255,0.03)',
            border: i===idx ? '1px solid rgba(249,115,22,0.4)' : i<idx ? '1px solid rgba(34,197,94,0.25)' : '1px solid var(--border)',
            color: i===idx ? 'var(--orange)' : i<idx ? '#22c55e' : 'var(--text3)',
            transition:'all 0.3s',
          }}>
            {i<idx ? <CheckCircle2 size={12}/> : i===idx ? <RefreshCw size={12} style={{animation:'spin 1s linear infinite'}}/> : s.icon}
            {s.label}
          </div>
          {i<stages.length-1 && (
            <div style={{width:20, height:1, background:i<idx ? 'rgba(34,197,94,0.4)' : 'var(--border)', flexShrink:0}}/>
          )}
        </React.Fragment>
      ))}
    </div>
  )
}

// ── Session badge ─────────────────────────────────────────────────────────────
function SessionBadge({ session, onClear }) {
  return (
    <div style={{
      display:'flex', alignItems:'center', justifyContent:'space-between',
      padding:'10px 16px',
      background:'rgba(34,197,94,0.07)',
      border:'1px solid rgba(34,197,94,0.2)',
      borderRadius:8, marginTop:12,
    }}>
      <div style={{display:'flex',alignItems:'center',gap:10}}>
        <CheckCircle2 size={16} color="#22c55e"/>
        <div>
          <p style={{fontSize:13,fontWeight:500,color:'#f1f1f3'}}>{session.filename}</p>
          <p style={{fontSize:11,color:'var(--text2)',fontFamily:'var(--mono)'}}>
            {session.stats.spec_chunks} spec chunks · {session.stats.proc_chunks} proc chunks · {session.stats.unique_specs} unique specs
          </p>
        </div>
      </div>
      <div style={{display:'flex',alignItems:'center',gap:8}}>
        <span style={{
          fontSize:10, padding:'2px 8px',
          background:'rgba(34,197,94,0.12)', border:'1px solid rgba(34,197,94,0.2)',
          borderRadius:4, color:'#22c55e', fontFamily:'var(--mono)',
        }}>READY</span>
        <button onClick={onClear} style={{background:'none',border:'none',cursor:'pointer',color:'var(--text3)',padding:4}}>
          <Trash2 size={14}/>
        </button>
      </div>
    </div>
  )
}

// ── Query bar ─────────────────────────────────────────────────────────────────
function QueryBar({ onQuery, loading, disabled }) {
  const [q, setQ] = useState('')
  const [variant, setVariant] = useState('')
  const [specOnly, setSpecOnly] = useState(true)
  const [showOpts, setShowOpts] = useState(false)

  const submit = () => {
    if (!q.trim() || loading || disabled) return
    onQuery({ query: q.trim(), variant: variant || null, spec_only: specOnly })
  }

  const suggestions = [
    'torque for brake caliper bolts',
    'shock absorber lower nuts 4WD',
    'U-bolt tightening sequence stages',
    'halfshaft assembled length mm',
    'upper ball joint nut torque RWD',
    'stabilizer bar link nuts SVT Raptor',
  ]

  return (
    <div style={{marginTop:24}}>
      <div style={{display:'flex',gap:8}}>
        <div style={{flex:1,position:'relative'}}>
          <Search size={16} color="var(--text3)" style={{position:'absolute',left:14,top:'50%',transform:'translateY(-50%)'}}/>
          <input
            value={q}
            onChange={e => setQ(e.target.value)}
            onKeyDown={e => e.key==='Enter' && submit()}
            placeholder={disabled ? 'Upload a document first…' : 'e.g. "torque for brake caliper bolts"'}
            disabled={disabled}
            style={{
              width:'100%', padding:'12px 14px 12px 40px',
              background:'var(--surface)', border:'1px solid var(--border2)',
              borderRadius:8, color:disabled?'var(--text3)':'var(--text)',
              fontFamily:'var(--body)', fontSize:14,
              outline:'none', transition:'border-color 0.2s',
            }}
            onFocus={e => e.target.style.borderColor='rgba(249,115,22,0.5)'}
            onBlur={e => e.target.style.borderColor='var(--border2)'}
          />
        </div>

        <select
          value={variant}
          onChange={e => setVariant(e.target.value)}
          style={{
            padding:'12px 14px', background:'var(--surface)',
            border:'1px solid var(--border2)', borderRadius:8,
            color:'var(--text)', fontSize:13, cursor:'pointer',
            fontFamily:'var(--body)', minWidth:130,
          }}
        >
          <option value="">All variants</option>
          <option value="RWD">RWD</option>
          <option value="4WD">4WD</option>
          <option value="SVT Raptor">SVT Raptor</option>
        </select>

        <button
          onClick={submit}
          disabled={loading || disabled || !q.trim()}
          style={{
            padding:'12px 24px', borderRadius:8, border:'none',
            background: loading||disabled||!q.trim() ? 'rgba(249,115,22,0.2)' : 'var(--orange)',
            color: loading||disabled||!q.trim() ? 'var(--text3)' : '#fff',
            fontFamily:'var(--display)', fontSize:15, fontWeight:700,
            letterSpacing:'0.06em', cursor: loading||disabled||!q.trim() ? 'not-allowed' : 'pointer',
            display:'flex', alignItems:'center', gap:8, transition:'all 0.15s',
          }}
          onMouseEnter={e => { if(!loading&&!disabled&&q.trim()) e.target.style.background='var(--orange2)' }}
          onMouseLeave={e => { if(!loading&&!disabled&&q.trim()) e.target.style.background='var(--orange)' }}
        >
          {loading
            ? <RefreshCw size={15} style={{animation:'spin 0.8s linear infinite'}}/>
            : <Zap size={15}/>
          }
          {loading ? 'EXTRACTING' : 'EXTRACT'}
        </button>
      </div>

      {/* Suggestion chips */}
      {!disabled && (
        <div style={{display:'flex',flexWrap:'wrap',gap:6,marginTop:10}}>
          {suggestions.map(s => (
            <button
              key={s}
              onClick={() => setQ(s)}
              style={{
                padding:'4px 10px', borderRadius:20, fontSize:11,
                background:'transparent',
                border:'1px solid rgba(255,255,255,0.08)',
                color:'var(--text2)', cursor:'pointer', transition:'all 0.15s',
                fontFamily:'var(--mono)',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor='rgba(249,115,22,0.4)'; e.currentTarget.style.color='var(--orange)' }}
              onMouseLeave={e => { e.currentTarget.style.borderColor='rgba(255,255,255,0.08)'; e.currentTarget.style.color='var(--text2)' }}
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Results table ─────────────────────────────────────────────────────────────
function ResultsTable({ results, fullResults, query, elapsed }) {
  const [view, setView] = useState('table') // table | json
  const [expanded, setExpanded] = useState(null)

  const exportJSON = () => {
    const blob = new Blob([JSON.stringify(results, null, 2)], {type:'application/json'})
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
    a.download = `specs_${Date.now()}.json`; a.click()
  }

  const exportCSV = () => {
    const header = 'component,spec_type,value,unit,section_id,vehicle_variant,source,confidence'
    const rows = fullResults.map(r =>
      [r.component,r.spec_type,r.value,r.unit,r.section_id,
       (r.vehicle_variant||[]).join(';'),r.source,r.confidence].map(v=>`"${v}"`).join(',')
    )
    const blob = new Blob([[header,...rows].join('\n')], {type:'text/csv'})
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
    a.download = `specs_${Date.now()}.csv`; a.click()
  }

  if (!results?.length) return (
    <div style={{
      marginTop:24, padding:'40px 32px', textAlign:'center',
      border:'1px solid var(--border)', borderRadius:10,
      background:'var(--surface)',
    }}>
      <AlertTriangle size={28} color="var(--text3)" style={{marginBottom:12}}/>
      <p style={{fontFamily:'var(--display)',fontSize:16,fontWeight:700,letterSpacing:'0.04em',color:'var(--text2)'}}>
        NO SPECS FOUND
      </p>
      <p style={{fontSize:13,color:'var(--text3)',marginTop:4}}>Try a more specific query or a different variant</p>
    </div>
  )

  return (
    <div className="fade-up" style={{marginTop:24}}>
      {/* Header bar */}
      <div style={{
        display:'flex',alignItems:'center',justifyContent:'space-between',
        marginBottom:12,
      }}>
        <div style={{display:'flex',alignItems:'center',gap:12}}>
          <div style={{
            padding:'4px 12px', background:'rgba(249,115,22,0.12)',
            border:'1px solid rgba(249,115,22,0.3)', borderRadius:4,
            fontFamily:'var(--mono)', fontSize:12, color:'var(--orange)',
          }}>
            {results.length} spec{results.length!==1?'s':''} found
          </div>
          <span style={{fontSize:12,color:'var(--text3)',fontFamily:'var(--mono)'}}>
            {elapsed?.toFixed(2)}s
          </span>
        </div>

        <div style={{display:'flex',gap:8}}>
          {[['table','Table'],['json','JSON']].map(([v,l]) => (
            <button key={v} onClick={() => setView(v)} style={{
              padding:'5px 12px', borderRadius:5, fontSize:12, cursor:'pointer',
              fontFamily:'var(--mono)', fontWeight:500,
              background: view===v ? 'rgba(249,115,22,0.15)' : 'transparent',
              border: view===v ? '1px solid rgba(249,115,22,0.4)' : '1px solid var(--border)',
              color: view===v ? 'var(--orange)' : 'var(--text2)',
            }}>{l}</button>
          ))}
          <button onClick={exportJSON} style={{
            padding:'5px 12px', borderRadius:5, fontSize:12, cursor:'pointer',
            fontFamily:'var(--mono)', background:'transparent',
            border:'1px solid var(--border)', color:'var(--text2)',
            display:'flex',alignItems:'center',gap:5, transition:'all 0.15s',
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor='rgba(249,115,22,0.4)'; e.currentTarget.style.color='var(--orange)' }}
          onMouseLeave={e => { e.currentTarget.style.borderColor='var(--border)'; e.currentTarget.style.color='var(--text2)' }}
          >
            <Download size={12}/> JSON
          </button>
          <button onClick={exportCSV} style={{
            padding:'5px 12px', borderRadius:5, fontSize:12, cursor:'pointer',
            fontFamily:'var(--mono)', background:'transparent',
            border:'1px solid var(--border)', color:'var(--text2)',
            display:'flex',alignItems:'center',gap:5, transition:'all 0.15s',
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor='rgba(20,184,166,0.4)'; e.currentTarget.style.color='var(--teal)' }}
          onMouseLeave={e => { e.currentTarget.style.borderColor='var(--border)'; e.currentTarget.style.color='var(--text2)' }}
          >
            <Download size={12}/> CSV
          </button>
        </div>
      </div>

      {/* Table view */}
      {view === 'table' && (
        <div style={{
          border:'1px solid var(--border)',
          borderRadius:10, overflow:'hidden',
          background:'var(--surface)',
        }}>
          <table style={{width:'100%',borderCollapse:'collapse'}}>
            <thead>
              <tr style={{borderBottom:'1px solid var(--border2)'}}>
                {['Component','Type','Value','Section','Variant','Src','Conf.'].map(h => (
                  <th key={h} style={{
                    padding:'10px 14px', textAlign:'left',
                    fontFamily:'var(--mono)', fontSize:10,
                    color:'var(--text3)', fontWeight:500,
                    letterSpacing:'0.1em', background:'var(--surface2)',
                    textTransform:'uppercase',
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {fullResults.map((r, i) => (
                <React.Fragment key={i}>
                  <tr
                    onClick={() => setExpanded(expanded===i ? null : i)}
                    style={{
                      borderBottom:'1px solid var(--border)',
                      cursor:'pointer', transition:'background 0.15s',
                    }}
                    onMouseEnter={e => e.currentTarget.style.background='rgba(249,115,22,0.04)'}
                    onMouseLeave={e => e.currentTarget.style.background='transparent'}
                  >
                    <td style={{padding:'10px 14px', fontSize:13, fontWeight:500}}>
                      <div style={{display:'flex',alignItems:'center',gap:6}}>
                        {r.is_safety_critical && (
                          <Shield size={12} color="var(--red)" title="Safety critical"/>
                        )}
                        {r.is_conflict && (
                          <AlertTriangle size={12} color="var(--amber)" title="Conflicting values found"/>
                        )}
                        {r.component}
                      </div>
                    </td>
                    <td style={{padding:'10px 14px',fontSize:12,color:'var(--text2)',fontFamily:'var(--mono)'}}>
                      {r.spec_type}
                    </td>
                    <td style={{padding:'10px 14px'}}>
                      <span style={{
                        fontFamily:'var(--mono)', fontSize:14, fontWeight:600,
                        color: unitColor(r.unit),
                      }}>
                        {r.value}<span style={{fontSize:11,marginLeft:4,opacity:0.7}}>{r.unit}</span>
                      </span>
                    </td>
                    <td style={{padding:'10px 14px',fontSize:11,color:'var(--text3)',fontFamily:'var(--mono)'}}>
                      {r.section_id}
                    </td>
                    <td style={{padding:'10px 14px'}}>
                      {(r.vehicle_variant||[]).map(v => (
                        <span key={v} style={{
                          padding:'2px 6px', marginRight:4, borderRadius:3,
                          fontSize:10, fontFamily:'var(--mono)',
                          background:'rgba(249,115,22,0.1)',
                          border:'1px solid rgba(249,115,22,0.2)',
                          color:'var(--orange)',
                        }}>{v}</span>
                      ))}
                    </td>
                    <td style={{padding:'10px 14px'}}>
                      <span style={{
                        padding:'2px 6px', borderRadius:3, fontSize:10, fontFamily:'var(--mono)',
                        background: r.source==='llm' ? 'rgba(59,130,246,0.1)' : 'rgba(20,184,166,0.1)',
                        border: r.source==='llm' ? '1px solid rgba(59,130,246,0.2)' : '1px solid rgba(20,184,166,0.2)',
                        color: r.source==='llm' ? '#60a5fa' : '#2dd4bf',
                      }}>{r.source}</span>
                    </td>
                    <td style={{padding:'10px 14px',fontSize:12,fontFamily:'var(--mono)',
                      color: r.confidence>0.95 ? '#22c55e' : r.confidence>0.8 ? 'var(--amber)' : 'var(--text3)'}}>
                      {Math.round(r.confidence*100)}%
                    </td>
                  </tr>
                  {expanded===i && (
                    <tr style={{background:'rgba(249,115,22,0.03)'}}>
                      <td colSpan={7} style={{padding:'10px 14px 14px'}}>
                        <p style={{fontSize:11,color:'var(--text3)',fontFamily:'var(--mono)',marginBottom:4}}>
                          RAW SOURCE TEXT
                        </p>
                        <p style={{
                          fontSize:12, color:'var(--text2)', fontFamily:'var(--mono)',
                          padding:'8px 12px',
                          background:'var(--bg2)', borderRadius:6,
                          borderLeft:'2px solid var(--orange)',
                        }}>
                          {r.raw_text || '—'}
                        </p>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* JSON view */}
      {view === 'json' && (
        <pre style={{
          background:'var(--bg2)', border:'1px solid var(--border)',
          borderRadius:10, padding:20, overflow:'auto',
          fontSize:12, fontFamily:'var(--mono)', color:'var(--text2)',
          maxHeight:400, lineHeight:1.6,
        }}>
          {JSON.stringify(results, null, 2)}
        </pre>
      )}

      {/* Legend */}
      <div style={{display:'flex',gap:16,marginTop:12,flexWrap:'wrap'}}>
        {[
          { color:'#f97316', label:'Nm' },
          { color:'#f59e0b', label:'lb-ft' },
          { color:'#fbbf24', label:'lb-in' },
          { color:'#14b8a6', label:'mm / in' },
          { color:'#3b82f6', label:'degrees' },
        ].map(({color,label}) => (
          <div key={label} style={{display:'flex',alignItems:'center',gap:5,fontSize:11,color:'var(--text3)'}}>
            <div style={{width:8,height:8,borderRadius:'50%',background:color}}/>
            {label}
          </div>
        ))}
        <div style={{marginLeft:'auto',display:'flex',gap:12}}>
          <span style={{fontSize:11,color:'var(--text3)',display:'flex',alignItems:'center',gap:4}}>
            <Shield size={10} color="var(--red)"/> safety-critical
          </span>
          <span style={{fontSize:11,color:'var(--text3)',display:'flex',alignItems:'center',gap:4}}>
            <AlertTriangle size={10} color="var(--amber)"/> conflict
          </span>
        </div>
      </div>
    </div>
  )
}

// ── History panel ─────────────────────────────────────────────────────────────
function HistoryPanel({ history }) {
  if (!history.length) return (
    <div style={{padding:'24px',textAlign:'center'}}>
      <p style={{fontSize:13,color:'var(--text3)'}}>No queries yet</p>
    </div>
  )
  return (
    <div style={{padding:'0 0 8px'}}>
      {history.map((h,i) => (
        <div key={i} style={{
          padding:'10px 16px', borderBottom:'1px solid var(--border)',
          fontSize:13,
        }}>
          <p style={{color:'var(--text)',marginBottom:2}}>{h.query}</p>
          <p style={{fontSize:11,color:'var(--text3)',fontFamily:'var(--mono)'}}>
            {h.count} spec{h.count!==1?'s':''} · {h.elapsed?.toFixed(1)}s
            {h.variant ? ` · ${h.variant}` : ''}
          </p>
        </div>
      ))}
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [apiOk, setApiOk] = useState(false)
  const [model, setModel] = useState('')
  const [session, setSession] = useState(null)
  const [pipeStage, setPipeStage] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [results, setResults] = useState(null)
  const [fullResults, setFullResults] = useState([])
  const [elapsed, setElapsed] = useState(null)
  const [history, setHistory] = useState([])
  const [tab, setTab] = useState('extract') // extract | history

  // Check API health on mount
  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(d => {
        setApiOk(d.status === 'ok')
        if (d.ollama) setModel('llama3.1:8b')
      })
      .catch(() => setApiOk(false))
  }, [])

  const handleUpload = async (file) => {
    setError('')
    setProcessing(true)
    setSession(null)
    setResults(null)
    setPipeStage('extract')

    try {
      const form = new FormData()
      form.append('file', file)

      // Simulate stage progression
      const stageTimer = setTimeout(() => setPipeStage('chunk'), 3000)
      const stageTimer2 = setTimeout(() => setPipeStage('index'), 8000)

      const res = await fetch('/api/upload', { method:'POST', body:form })
      clearTimeout(stageTimer); clearTimeout(stageTimer2)

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Upload failed')
      }

      const data = await res.json()
      setPipeStage('ready')
      setSession(data)

      setTimeout(() => setPipeStage(null), 1500)
    } catch (e) {
      setError(e.message)
      setPipeStage(null)
    } finally {
      setProcessing(false)
    }
  }

  const handleQuery = async ({ query, variant, spec_only }) => {
    if (!session) return
    setError('')
    setLoading(true)
    setResults(null)
    const t0 = Date.now()

    try {
      const res = await fetch('/api/query', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          session_id: session.session_id,
          query, variant, spec_only, k:5,
        }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Query failed')
      }
      const data = await res.json()
      const t = (Date.now() - t0) / 1000
      setElapsed(t)
      setResults(data.results)
      setFullResults(data.full_results || [])
      setHistory(prev => [{query, count:data.results.length, elapsed:t, variant}, ...prev.slice(0,19)])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{minHeight:'100vh'}}>
      <GearBg/>
      <Topbar apiOk={apiOk} model={model}/>

      <main style={{maxWidth:900, margin:'0 auto', padding:'40px 24px'}}>

        {/* Hero headline */}
        <div className="fade-up" style={{marginBottom:40}}>
          <div style={{
            display:'inline-flex',alignItems:'center',gap:8,
            padding:'4px 12px', borderRadius:20, marginBottom:16,
            background:'rgba(249,115,22,0.08)',
            border:'1px solid rgba(249,115,22,0.2)',
            fontSize:11, fontFamily:'var(--mono)',
            color:'var(--orange)', letterSpacing:'0.1em',
          }}>
            <Activity size={11}/> VEHICLE SPECIFICATION EXTRACTION · RAG PIPELINE
          </div>
          <h1 style={{
            fontFamily:'var(--display)',
            fontSize:'clamp(36px,5vw,56px)',
            fontWeight:800,
            letterSpacing:'0.02em',
            lineHeight:1.1,
            color:'#fff',
          }}>
            EXTRACT SPECS<br/>
            <span style={{color:'var(--orange)'}}>FROM ANY</span> MANUAL
          </h1>
          <p style={{
            marginTop:12, fontSize:15, color:'var(--text2)', maxWidth:520,
          }}>
            Upload any vehicle service manual PDF. Natural language queries.
            Powered by Llama 3.1 8B · FAISS · BM25 · RRF fusion.
          </p>
        </div>

        {/* Tabs */}
        <div style={{
          display:'flex',gap:0,
          borderBottom:'1px solid var(--border)',
          marginBottom:28,
        }}>
          {[['extract','Extract'],['history','History']].map(([k,l]) => (
            <button key={k} onClick={() => setTab(k)} style={{
              padding:'8px 20px', background:'none', border:'none',
              borderBottom: tab===k ? '2px solid var(--orange)' : '2px solid transparent',
              color: tab===k ? 'var(--orange)' : 'var(--text2)',
              fontFamily:'var(--display)', fontSize:14, fontWeight:700,
              letterSpacing:'0.06em', cursor:'pointer', transition:'all 0.15s',
              marginBottom:-1,
            }}>{l.toUpperCase()}</button>
          ))}
        </div>

        {tab === 'extract' && (
          <div className="fade-in">
            {/* Error */}
            {error && (
              <div style={{
                padding:'12px 16px', marginBottom:20,
                background:'rgba(239,68,68,0.1)',
                border:'1px solid rgba(239,68,68,0.25)',
                borderRadius:8, display:'flex', alignItems:'center', gap:10,
                fontSize:13, color:'#fca5a5',
              }}>
                <AlertTriangle size={14}/>
                {error}
                <button onClick={()=>setError('')} style={{marginLeft:'auto',background:'none',border:'none',cursor:'pointer',color:'#fca5a5'}}>
                  <X size={14}/>
                </button>
              </div>
            )}

            {/* Upload */}
            {!session && (
              <>
                <UploadZone onUpload={handleUpload} processing={processing}/>
                {pipeStage && <PipelineProgress stage={pipeStage}/>}
              </>
            )}

            {/* Session badge */}
            {session && (
              <SessionBadge session={session} onClear={() => {
                setSession(null); setResults(null); setPipeStage(null)
              }}/>
            )}

            {/* Upload different doc button */}
            {session && (
              <button
                onClick={() => { setSession(null); setResults(null) }}
                style={{
                  marginTop:10, padding:'6px 14px', background:'transparent',
                  border:'1px solid var(--border)', borderRadius:6,
                  color:'var(--text2)', fontSize:12, cursor:'pointer',
                  fontFamily:'var(--mono)', display:'flex',alignItems:'center',gap:6,
                  transition:'all 0.15s',
                }}
                onMouseEnter={e => { e.currentTarget.style.borderColor='rgba(249,115,22,0.4)'; e.currentTarget.style.color='var(--orange)' }}
                onMouseLeave={e => { e.currentTarget.style.borderColor='var(--border)'; e.currentTarget.style.color='var(--text2)' }}
              >
                <Upload size={12}/> Upload different document
              </button>
            )}

            {/* Query bar */}
            <QueryBar onQuery={handleQuery} loading={loading} disabled={!session}/>

            {/* Results */}
            {results !== null && (
              <ResultsTable
                results={results}
                fullResults={fullResults}
                elapsed={elapsed}
              />
            )}
          </div>
        )}

        {tab === 'history' && (
          <div className="fade-in" style={{
            background:'var(--surface)', border:'1px solid var(--border)',
            borderRadius:10, overflow:'hidden',
          }}>
            <HistoryPanel history={history}/>
          </div>
        )}

        {/* Footer */}
        <div style={{
          marginTop:60, paddingTop:24,
          borderTop:'1px solid var(--border)',
          display:'flex',alignItems:'center',justifyContent:'space-between',
        }}>
          <p style={{fontSize:11,color:'var(--text3)',fontFamily:'var(--mono)'}}>
            Phase 1: PyMuPDF/pdftotext · Phase 2: BAAI/bge-base-en-v1.5 · Phase 3: Llama 3.1 8B
          </p>
          <p style={{fontSize:11,color:'var(--text3)',fontFamily:'var(--mono)'}}>
            PREDII SPEC EXTRACTOR
          </p>
        </div>
      </main>
    </div>
  )
}