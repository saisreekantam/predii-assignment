# Predii Vehicle Spec Extractor

> A RAG (Retrieval-Augmented Generation) pipeline that extracts structured vehicle specifications — torque values, dimensions, pressures — from PDF service manuals. Includes a React frontend and a FastAPI backend.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/LLM-Llama%203.1%208B-black?logo=meta&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS%20%2B%20BM25-orange)

---

## UI Screenshots

📸 **[View all UI screenshots on Google Drive](https://drive.google.com/drive/folders/181nk4R4s3E6twERLDNkMkw0IcfdPHzlp?usp=drive_link)**

---

## Architecture

```
PDF / TXT  →  Phase 1 (extraction)  →  Phase 2 (chunking + FAISS index)  →  Phase 3 (LLM query)
                                                                                    ↕
                                                                          FastAPI  ↔  React UI
```

| Layer | Location | What it does |
|---|---|---|
| Phase 1 | `phase1_extraction/pdf_extractor.py` | Parses PDFs, detects torque tables and inline specs, outputs structured JSON |
| Phase 2 | `phase2_chunking/chunker_embedder.py` | Embeds chunks with `sentence-transformers`, builds dual FAISS + BM25 indexes |
| Phase 3 | `phase3_extraction/extractor.py` | Hybrid retrieval + Llama 3.1 8B via Ollama for structured spec output |
| API | `predii_app/api/main.py` | FastAPI server — upload, query, history, config endpoints |
| UI | `predii_app/frontend/` | React + Vite SPA — upload docs, run queries, view history |

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) installed and running locally

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/saisreekantam/predii-assignment.git
cd predii-assignment
```

### 2. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Pull the LLM model

```bash
ollama pull llama3.1:8b
```

> Make sure `ollama serve` is running (it starts automatically on most systems after install).

### 4. Build the index from a PDF

Run the full pipeline once to build the FAISS index from your service manual:

```bash
# Phase 1 — extract specs from PDF
python phase1_extraction/pdf_extractor.py "your-service-manual.pdf" --out-dir ./phase1_output

# Phase 2 — build chunked FAISS index
python phase2_chunking/chunker_embedder.py build \
  --spec ./phase1_output/<n>_spec_segments.json \
  --all  ./phase1_output/<n>_all_segments.json \
  --out  ./phase2_index
```

### 5. Start the backend

```bash
cd predii_app/api
python main.py
# Server runs at http://localhost:8000
```

### 6. Start the frontend

```bash
cd predii_app/frontend
npm install
npm run dev
# UI available at http://localhost:5173
```

---

## Usage

### Via the UI

1. Open `http://localhost:5173`
2. **Upload** a PDF or TXT service manual — the backend runs Phase 1 + Phase 2 automatically
3. **Query** — type a natural-language spec question, e.g.:
   - `"Torque for brake caliper bolts"`
   - `"Shock absorber lower nuts 4WD"`
   - `"Halfshaft assembled length"`
4. Results show component, spec type, value, unit, vehicle variant, and confidence
5. **History** tab lists all past queries
6. **Settings** tab lets you change the Ollama model, index directory, and retrieval depth (k)

### Via the API directly

```bash
# Health check
curl http://localhost:8000/health

# Upload a document and get a session ID
curl -X POST http://localhost:8000/api/upload \
  -F "file=@your-manual.pdf"

# Query against the uploaded document
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<id>", "query": "torque for wheel bearing bolt"}'

# Query the pre-built index (if phase2_index exists)
curl -X POST http://localhost:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{"query": "brake caliper torque", "spec_only": true}'
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health check |
| `GET` | `/api/health` | Health + Ollama connectivity |
| `POST` | `/api/upload` | Upload PDF/TXT, triggers Phase 1+2 indexing |
| `POST` | `/api/query` | Query uploaded document by session ID |
| `POST` | `/api/extract` | Query the pre-built index |
| `GET` | `/api/history` | List query history |
| `DELETE` | `/api/history` | Clear all history |
| `GET` | `/api/config` | Get current config |
| `PATCH` | `/api/config` | Update model, index dir, k |
| `GET` | `/api/demo-queries` | List sample queries |

---

## Configuration

Default config (changeable via `/api/config` PATCH or the Settings page):

| Key | Default | Description |
|---|---|---|
| `model` | `llama3.1:8b` | Ollama model name |
| `ollama_host` | `http://localhost:11434` | Ollama server URL |
| `index_dir` | `./phase2_index` | Pre-built FAISS index path |
| `k` | `8` | Number of chunks retrieved per query |

---

## Troubleshooting

**Ollama not responding**
```bash
ollama serve        # start the service
ollama list         # verify the model is available
```

**FAISS index not found**  
Run Phases 1 and 2 before starting the backend, or use the Upload feature in the UI — it runs them automatically per session.

**PDF extraction returns no specs**  
The pipeline uses a PyMuPDF → pdfplumber → pdftotext fallback waterfall. If all three fail, the PDF is likely a scanned image — run it through an OCR tool like `tesseract` first to add a text layer.

**Frontend can't reach the backend**  
Check that the backend is on port `8000`. The Vite dev server proxies `/api` automatically, so no manual CORS config is needed in development.

---

## Project Structure

```
predii-assignment/
├── phase1_extraction/
│   ├── pdf_extractor.py          # Main Phase 1 pipeline
│   └── pdf_extractor_generic.py  # Generic fallback extractor
├── phase2_chunking/
│   └── chunker_embedder.py       # FAISS + BM25 dual-index builder
├── phase3_extraction/
│   └── extractor.py              # LLM extraction + fast path
├── predii_app/
│   ├── api/
│   │   └── main.py               # FastAPI backend
│   └── frontend/
│       ├── index.html
│       ├── package.json
│       ├── vite.config.js
│       └── src/
│           ├── App.jsx            # Main upload + query UI
│           ├── main.jsx
│           ├── index.css
│           ├── lib/api.js         # API client
│           └── pages/
│               ├── QueryPage.jsx
│               ├── HistoryPage.jsx
│               └── SettingsPage.jsx
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **Embeddings** — `BAAI/bge-base-en-v1.5` via `sentence-transformers`
- **Vector search** — FAISS + BM25 with Reciprocal Rank Fusion
- **LLM** — Llama 3.1 8B via Ollama (local, no API key needed)
- **Backend** — FastAPI + Uvicorn
- **Frontend** — React 18 + Vite
