# DESIGN.md — Predii Vehicle Spec Extraction: Technical Design Rationale

---

## 1. Problem Statement

The core problem is extracting structured mechanical specifications — torque values, fluid capacities, dimensional tolerances, tire pressures, and similar data — from unstructured automotive PDF service manuals, and returning them in a consistent, machine-readable format.

This is harder than it sounds. Automotive service manuals are notoriously inconsistent: one section might list torque specs in a clean table, while the next embeds them inline ("tighten to 47 Nm ± 5%"), and yet another uses a multi-column layout that completely confuses naive PDF parsers. Specs often carry implicit context — a torque value means nothing without knowing which bolt, which engine variant, and which tightening stage it belongs to. PDFs also have no semantic structure; a heading and a body value are just text at different coordinates.

The target output is a JSON object per extracted spec with six fields: `component`, `spec_type`, `value`, `unit`, `vehicle_variant`, and `confidence`. This schema forces the extraction pipeline to resolve ambiguity rather than pass it through — a value like "47" is useless without a unit; a unit like "Nm" is useless without a component. Every field serves a downstream purpose.

---

## 2. Pipeline Overview

I structured the system as four sequential, independently testable phases. The separation is deliberate: each phase has a single responsibility, fails gracefully without breaking subsequent phases, and can be swapped out without touching the others.

**Phase 1 — PDF Text Extraction** runs a 3-tool fallback waterfall: PyMuPDF first, then pdfplumber, then pdftotext. The extracted text is also passed through a fast-path regex engine that tries to capture high-confidence specs (e.g., patterns like `\d+\s?Nm`) before any retrieval or LLM work happens.

**Phase 2 — Chunking and Indexing** splits the extracted text into overlapping chunks and embeds them using BAAI/bge-base-en-v1.5. These embeddings are stored in a FAISS index. A parallel BM25 index is built over the same chunks. Both indexes are isolated per session under `/tmp/predii_sessions/<session_id>/`.

**Phase 3 — Hybrid Retrieval and Extraction** takes a user query, retrieves candidate chunks from both indexes, fuses the rankings with Reciprocal Rank Fusion, and passes the top chunks to Llama 3.1 8B (via Ollama) for structured extraction. Specs already captured in Phase 1 with high regex confidence skip the LLM entirely via the fast-path.

**Phase 4 — API and UI** exposes the pipeline through a FastAPI backend and a React/Vite frontend. The frontend handles session management, PDF upload, and spec display.

The reason for four phases rather than one monolithic script is simple: when something goes wrong — and with PDFs, something always goes wrong — I need to know *where* it went wrong. A pipeline that mixes parsing, retrieval, and LLM calls into one function is impossible to debug and impossible to improve incrementally.

---

## 3. Tool Choices & Why

**PyMuPDF / pdfplumber / pdftotext (fallback waterfall):** I chose a waterfall over a single parser because no single PDF parser handles every real-world automotive manual. PyMuPDF is fast and accurate for digitally-born PDFs. pdfplumber handles table-heavy layouts better. pdftotext is the final fallback — slower and dumber, but extremely robust. Running all three and picking the best result (measured by text coverage) means the pipeline degrades gracefully instead of silently failing on a malformed PDF.

**BAAI/bge-base-en-v1.5:** I chose this embedding model over OpenAI's text-embedding-ada-002 because the pipeline needs to run fully locally with no API key and no cost. BGE models from BAAI are among the strongest open-weight embedding models on MTEB benchmarks, and they perform well on domain-specific technical text. For an automotive context, the ability to embed phrases like "crankshaft bearing clearance" meaningfully is more important than general-purpose semantic range.

**FAISS:** I chose FAISS over Chroma or Pinecone because this is an assignment with a single-user, per-session workload. FAISS runs entirely in-process with no server, no network dependency, and no infrastructure setup. Chroma is fine but adds a server abstraction layer that isn't needed here. Pinecone is a managed cloud service — completely wrong for a local, privacy-sensitive document pipeline.

**BM25 alongside FAISS:** Dense vector retrieval is good at capturing semantic similarity but bad at exact keyword matching. If a query asks for "camshaft bearing oil clearance," a dense model might return semantically adjacent chunks about "engine lubrication" that don't contain the actual spec. BM25 catches exact token overlap and surfaces the spec directly. I use both because they're complementary, not competitive.

**Reciprocal Rank Fusion:** I chose RRF over a weighted score combination for a specific reason: the scores produced by FAISS (L2 distance) and BM25 (TF-IDF score) live in completely different numerical spaces and are not directly comparable. Normalizing them to combine would introduce an arbitrary scaling decision. RRF works on ranks, not scores, which makes it distribution-agnostic. The fusion parameter k=60 is a widely validated default from the original RRF paper.

**Llama 3.1 8B via Ollama:** I chose this over GPT-4 or Mistral API calls because the pipeline must work without an internet connection, without an API key, and at zero marginal cost. For structured extraction — where the output schema is fully specified in the prompt — an 8B model with a well-engineered prompt performs very well. The task is constrained: I'm not asking for open-ended reasoning, just JSON extraction from retrieved context. Llama 3.1 handles this reliably.

**FastAPI over Flask:** I chose FastAPI because it supports async request handling natively, which matters when the LLM call can take several seconds. Flask's synchronous default would block the worker during LLM inference. FastAPI also auto-generates OpenAPI docs from type annotations, which makes the API self-documenting with no extra effort.

**React + Vite over Streamlit:** I explicitly rejected Streamlit because it offers no meaningful control over layout, theming, or component behavior. The UI for this project needed to communicate the automotive domain — dark backgrounds, precision typography, structured data display — and Streamlit's widget model makes that impossible without fighting the framework the entire way. Vite gives me a fast dev server and clean production builds; React gives me full control over state and rendering.

---

## 4. Retrieval Design Decisions

The choice to use hybrid retrieval comes from a fundamental limitation of dense embeddings: they optimize for semantic proximity, not literal match. In automotive spec extraction, the most important queries are hyper-literal — "what is the torque for the cylinder head bolt" — and the ground-truth answer is likely in a chunk that contains those exact tokens. BM25 finds it; dense retrieval might rank a general section about "engine assembly procedures" higher because it's topically related.

I chose RRF over weighted score combination specifically because it avoids the score calibration problem. Fusing ranks is more robust than fusing scores when the underlying retrievers have different output distributions.

The fast-path design is one of the more important latency decisions in the pipeline. If Phase 1 regex already extracted a spec with high confidence (e.g., it found "47 Nm" adjacent to "cylinder head"), there is no reason to spend 3–8 seconds on an LLM call that will very likely return the same value. The fast-path outputs the spec directly with a high confidence score and flags it as `regex_extracted`. This reduces average query latency substantially for common, well-formatted specs.

Per-session isolation — each uploaded PDF gets its own FAISS index and BM25 index under `/tmp/predii_sessions/<session_id>/` — is a design choice driven by correctness, not convenience. Without isolation, uploading a second PDF would contaminate the first document's index. Sessions make the pipeline document-agnostic: the retrieval logic has no notion of "which document is loaded," it simply queries whatever index belongs to the current session.

---

## 5. Output Quality & Accuracy

Structured output is enforced through prompting, not post-processing. The LLM receives a strict instruction to return only a JSON array conforming to the schema, with no surrounding text. If the LLM returns malformed JSON — which happens rarely but does happen — the backend catches the parse error and falls back to returning the raw text chunk with a low confidence score rather than crashing.

Confidence scoring uses a two-tier approach. Regex fast-path extractions receive a high confidence score (≥ 0.90) because the pattern match is explicit and the value format is unambiguous. LLM-path extractions receive a confidence score derived from two signals: the top retrieval score from RRF fusion (how relevant was the source chunk?) and the LLM's own self-assessment, which I ask it to include as a field in its output. These are combined with a weighted average, with the retrieval score weighted more heavily because the LLM tends to be overconfident.

There are known edge cases I acknowledged but did not fully solve. Dash-range values like "1.5–3.0 Nm" are partially missed by Phase 1 regex because the current patterns are written for single numeric values. Multi-stage tightening sequences (e.g., "first pass 20 Nm, second pass 45°") require multi-sentence context that can span chunk boundaries, which means they sometimes retrieve correctly but extract incompletely. Rare or highly specific specs require increasing retrieval k, which improves recall at the cost of introducing more irrelevant context for the LLM — a classic precision/recall tradeoff I left as a tunable parameter.

---

## 6. Code Design Principles

Each of the four phases maps to a standalone script that can be run and tested independently. Phase 1 (PDF extraction) has no dependency on Phase 2. Phase 2 (indexing) only needs the text output from Phase 1. This means I can test the extraction quality of PyMuPDF independently of anything downstream, and I can test retrieval quality by feeding it hand-crafted text chunks without parsing a PDF at all.

Extraction logic, retrieval logic, and LLM logic are kept in separate modules. The FastAPI app imports from these modules but does not contain any of the core pipeline logic itself — it's purely responsible for request handling, session management, and response serialization. This makes the pipeline logic reusable without going through the API layer.

The frontend uses React hooks for state management with no external state library. The application state (session ID, upload status, query results) is simple enough that Redux or Zustand would add complexity without benefit. A thin `api.js` module wraps all `fetch` calls, so if the backend URL or request format changes, there's one file to update rather than hunting through components.

---

## 7. Bonus: UI & UX Decisions

I built a custom React UI instead of a notebook or CLI because the primary audience for this submission is a human reviewer who will evaluate the pipeline end-to-end. A CLI requires knowing the right commands; a notebook exposes implementation details. A UI makes the pipeline feel like a product rather than a script collection, and it demonstrates that the extraction results are presentable and usable.

The visual design choices are all intentional and domain-referenced. I used a dark background (#0f1117 base) because automotive technical tools — from diagnostic software to manufacturer portals — consistently use dark UIs for extended readability under bright shop lighting. The orange accent (#f97316) maps to the warning/highlight color used in most OEM service documentation. Barlow Condensed is a typeface with clear mechanical associations and excellent readability at small sizes — appropriate for dense spec tables. JetBrains Mono for spec values makes numeric data immediately visually distinct from surrounding text. The mechanical gear SVG in the background is subtle enough not to distract but clearly communicates the domain on first load.

The session-based upload flow solves a real usability problem: if a reviewer wants to compare specs from two different service manuals, they can open two sessions in parallel without one document's index contaminating the other's results. Each session is self-contained and independently queryable.

---

## 8. Ideas for Improvement (if given more time)

**OCR integration for scanned PDFs.** The current waterfall assumes the PDF contains selectable text. Many older automotive service manuals are scanned images. Adding a tesseract OCR stage — triggered when text extraction returns below a minimum character threshold — would extend the pipeline to handle scanned documents.

**Extend Phase 1 regex to handle dash-range values.** The current patterns match single numeric values. A straightforward extension would add patterns for formats like `(\d+\.?\d*)\s*[–\-]\s*(\d+\.?\d*)` and emit the full range as the `value` field. This would improve fast-path coverage significantly.

**Component token-overlap post-filter.** At high retrieval k, the top chunks can include sections from unrelated subsystems. A post-retrieval filter that checks for token overlap between the query's component term and the retrieved chunk's content would prune irrelevant results before they reach the LLM, improving extraction precision.

**Fine-tune a small model on automotive spec extraction.** The current pipeline prompts a general-purpose LLM to extract structured specs. A model fine-tuned on (manual excerpt, spec JSON) pairs — even on a small dataset — would likely outperform the prompted approach on edge cases and reduce hallucination risk on ambiguous spec formats.

**User feedback loop.** The UI could expose simple thumbs-up/thumbs-down controls on each extracted spec. Corrections could be logged and used to re-rank retrieval results (treating corrections as negative training signal for BM25 boosting) or to build a fine-tuning dataset for the improvement above.
