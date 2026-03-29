"""
Predii Vehicle Spec Extractor — FastAPI Backend
Connects Phase 2 (retrieval) + Phase 3 (LLM extraction) to the React frontend.
"""
from __future__ import annotations
import json, subprocess, sys, uuid, logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from urllib.error import URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup logging
log = logging.getLogger(__name__)

# Add phase dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase2_chunking"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase3_extraction"))

app = FastAPI(title="Predii Spec Extractor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory history (replace with SQLite for persistence) ──────────────────
history_store: List[dict] = []
session_store: Dict[str, dict] = {}
session_pipeline_cache: Dict[str, dict] = {}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_index_dir(index_dir: str) -> Path:
    p = Path(index_dir)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p

# ── Config defaults ──────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "index_dir": str(PROJECT_ROOT / "phase2_index"),
    "model":     "llama3.1:8b",
    "ollama_host": "http://localhost:11434",
    "k": 8,
}
current_config = dict(DEFAULT_CONFIG)
_pipeline = None
_cached_index_stats: Optional[dict] = None

UPLOAD_DIR = PROJECT_ROOT / ".uploaded_docs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PHASE1_SCRIPT = PROJECT_ROOT / "phase1_extraction" / "pdf_extractor.py"
PHASE2_SCRIPT = PROJECT_ROOT / "phase2_chunking" / "chunker_embedder.py"


def _safe_ollama_check(host: str) -> bool:
    url = host.rstrip("/") + "/api/tags"
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=0.5) as resp:  # nosec B310 (trusted local host)
            return 200 <= int(resp.status) < 300
    except (URLError, TimeoutError, ValueError):
        return False


def _run_cmd(cmd: List[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    return p.returncode, p.stdout, p.stderr


def _safe_err(text: str, limit: int = 600) -> str:
    clean = "".join(ch for ch in (text or "") if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    return clean[:limit].strip()


def _extract_text_with_pdftotext(pdf_path: Path, txt_path: Path) -> bool:
    rc, _out, err = _run_cmd([
        "pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), str(txt_path)
    ])
    if rc != 0:
        return False
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        return False
    return True


def _compute_unique_specs(spec_json: Path) -> int:
    try:
        specs = json.loads(spec_json.read_text(encoding="utf-8"))
        unique_components = {
            (s.get("component") or "").strip().lower()
            for s in specs
            if isinstance(s, dict) and (s.get("component") or "").strip()
        }
        return len(unique_components)
    except Exception:
        return 0


def _build_session_index(input_path: Path, session_dir: Path) -> dict:
    out_dir = session_dir / "phase1_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1_input = input_path
    if input_path.suffix.lower() == ".pdf":
        extracted_txt = session_dir / "uploaded_extracted.txt"
        if _extract_text_with_pdftotext(input_path, extracted_txt):
            phase1_input = extracted_txt

    rc1, _out1, err1 = _run_cmd([
        sys.executable,
        str(PHASE1_SCRIPT),
        str(phase1_input),
        "--out-dir",
        str(out_dir),
    ])

    spec_json = next(out_dir.glob("*_spec_segments.json"), None)
    all_json = next(out_dir.glob("*_all_segments.json"), None)
    if not spec_json or not all_json:
        if rc1 != 0:
            raise RuntimeError(f"Phase 1 extraction failed: {_safe_err(err1)}")
        raise RuntimeError("Phase 1 output missing spec/all segments")

    try:
        spec_list = json.loads(spec_json.read_text(encoding="utf-8"))
    except Exception:
        spec_list = []

    # Allow Phase 2 to run even with 0 specs - it will chunk whatever text was extracted
    log.info(f"Phase 1 output: {len(spec_list)} spec segments")

    rc2, _out2, err2 = _run_cmd([
        sys.executable,
        str(PHASE2_SCRIPT),
        "build",
        "--spec",
        str(spec_json),
        "--all",
        str(all_json),
        "--out",
        str(session_dir / "phase2_index"),
    ])
    if rc2 != 0:
        raise RuntimeError(f"Phase 2 indexing failed: {_safe_err(err2)}")

    stats_file = session_dir / "phase2_index" / "build_stats.json"
    stats = {"spec_chunks": 0, "proc_chunks": 0, "unique_specs": 0}
    if stats_file.exists():
        try:
            loaded = json.loads(stats_file.read_text(encoding="utf-8"))
            stats["spec_chunks"] = int(loaded.get("spec_chunks", 0))
            stats["proc_chunks"] = int(loaded.get("proc_chunks", 0))
        except Exception:
            pass
    stats["unique_specs"] = _compute_unique_specs(spec_json)
    return stats


def _load_index_stats() -> dict:
    global _cached_index_stats
    if _cached_index_stats is not None:
        return _cached_index_stats

    index_dir = _resolve_index_dir(current_config["index_dir"])
    stats_file = index_dir / "build_stats.json"
    spec_file = index_dir / "spec_chunks.json"

    stats = {"spec_chunks": 0, "proc_chunks": 0, "unique_specs": 0}

    if stats_file.exists():
        try:
            loaded = json.loads(stats_file.read_text(encoding="utf-8"))
            stats["spec_chunks"] = int(loaded.get("spec_chunks", 0))
            stats["proc_chunks"] = int(loaded.get("proc_chunks", 0))
        except Exception:
            pass

    if spec_file.exists():
        try:
            spec_chunks = json.loads(spec_file.read_text(encoding="utf-8"))
            unique_components = {
                (chunk.get("component") or "").strip().lower()
                for chunk in spec_chunks
                if isinstance(chunk, dict) and (chunk.get("component") or "").strip()
            }
            stats["unique_specs"] = len(unique_components)
            if stats["spec_chunks"] == 0:
                stats["spec_chunks"] = len(spec_chunks)
        except Exception:
            pass

    _cached_index_stats = stats
    return stats

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from extractor import ExtractionPipeline
        _pipeline = ExtractionPipeline(
            index_dir=str(_resolve_index_dir(current_config["index_dir"])),
            model=current_config["model"],
            ollama_host=current_config["ollama_host"],
            k=current_config["k"],
        )
    return _pipeline

# ── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    variant: Optional[str] = None
    spec_only: bool = True
    force_llm: bool = False

class SpecResult(BaseModel):
    component: str
    spec_type: str
    value: str
    unit: str
    section_id: str = ""
    section_name: str = ""
    vehicle_variant: List[str] = []
    source: str = ""
    confidence: float = 1.0
    is_safety_critical: bool = False
    is_conflict: bool = False

class QueryResponse(BaseModel):
    id: str
    query: str
    variant: Optional[str]
    results: List[SpecResult]
    metadata: dict
    timestamp: str

class ConfigUpdate(BaseModel):
    index_dir: Optional[str] = None
    model: Optional[str] = None
    ollama_host: Optional[str] = None
    k: Optional[int] = None


class UploadQueryRequest(BaseModel):
    session_id: str
    query: str
    variant: Optional[str] = None
    spec_only: bool = True
    force_llm: bool = False
    k: Optional[int] = None

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/health")
def api_health():
    return {
        "status": "ok",
        "version": app.version,
        "ollama": _safe_ollama_check(current_config["ollama_host"]),
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename or "uploaded-document"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Upload a PDF or TXT file")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    target = session_dir / f"input{suffix}"
    target.write_bytes(content)

    try:
        stats = _build_session_index(target, session_dir)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    session_store[session_id] = {
        "session_id": session_id,
        "filename": filename,
        "path": str(target),
        "index_dir": str(session_dir / "phase2_index"),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    # Ensure a fresh pipeline is built for this new session on first query.
    session_pipeline_cache.pop(session_id, None)

    return {
        "session_id": session_id,
        "filename": filename,
        "stats": stats,
    }

@app.post("/api/extract", response_model=QueryResponse)
def extract(req: QueryRequest):
    try:
        pipe = get_pipeline()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pipeline unavailable: {e}")

    try:
        result = pipe.run(
            req.query,
            variant=req.variant,
            spec_only=req.spec_only,
            force_llm=req.force_llm,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    entry = {
        "id":        str(uuid.uuid4()),
        "query":     req.query,
        "variant":   req.variant,
        "results":   result.get("full_results", []),
        "metadata":  result.get("metadata", {}),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    history_store.insert(0, entry)
    if len(history_store) > 100:
        history_store.pop()

    return QueryResponse(**entry)


@app.post("/api/query")
def query(req: UploadQueryRequest):
    if req.session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found. Upload a document first.")

    desired_k = int(req.k) if isinstance(req.k, int) and req.k > 0 else int(current_config.get("k", 8))
    sess = session_store[req.session_id]
    index_dir = sess.get("index_dir")
    if not index_dir:
        raise HTTPException(status_code=500, detail="Session index not available")

    try:
        cached = session_pipeline_cache.get(req.session_id)
        if (
            not cached
            or int(cached.get("k", 0)) != desired_k
            or cached.get("model") != current_config["model"]
            or cached.get("ollama_host") != current_config["ollama_host"]
        ):
            from extractor import ExtractionPipeline
            pipe = ExtractionPipeline(
                index_dir=str(_resolve_index_dir(index_dir)),
                model=current_config["model"],
                ollama_host=current_config["ollama_host"],
                k=desired_k,
            )
            session_pipeline_cache[req.session_id] = {
                "k": desired_k,
                "model": current_config["model"],
                "ollama_host": current_config["ollama_host"],
                "pipe": pipe,
            }
        else:
            pipe = cached["pipe"]

        result = pipe.run(
            req.query,
            variant=req.variant,
            spec_only=req.spec_only,
            force_llm=req.force_llm,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return {
        "session_id": req.session_id,
        "query": req.query,
        "variant": req.variant,
        "results": result.get("results", []),
        "full_results": result.get("full_results", []),
        "metadata": result.get("metadata", {}),
    }

@app.get("/api/history")
def get_history(limit: int = 50):
    return history_store[:limit]

@app.delete("/api/history")
def clear_history():
    history_store.clear()
    return {"cleared": True}

@app.delete("/api/history/{item_id}")
def delete_history_item(item_id: str):
    global history_store
    history_store = [h for h in history_store if h["id"] != item_id]
    return {"deleted": item_id}

@app.get("/api/config")
def get_config():
    return current_config

@app.patch("/api/config")
def update_config(cfg: ConfigUpdate):
    global _pipeline, _cached_index_stats
    updates = cfg.model_dump(exclude_none=True)
    if "index_dir" in updates and updates["index_dir"]:
        updates["index_dir"] = str(_resolve_index_dir(updates["index_dir"]))
    current_config.update(updates)
    _pipeline = None  # force reload on next request
    _cached_index_stats = None
    session_pipeline_cache.clear()
    return current_config

@app.get("/api/demo-queries")
def demo_queries():
    return [
        "Torque for brake caliper bolts",
        "Shock absorber lower nuts 4WD",
        "Wheel bearing and hub bolt torque",
        "Stage tightening sequence U-bolt",
        "Upper ball joint nut torque RWD",
        "Halfshaft assembled length",
        "Stabilizer bar link nuts SVT Raptor",
        "Tie-rod end nut torque",
        "Lower arm forward rearward nuts",
        "Wheel speed sensor bolt torque",
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
