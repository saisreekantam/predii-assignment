"""
Phase 2: Chunking, Embedding & Dual-Index Storage
==================================================
Vehicle Specification Extraction — 2014 F-150 Workshop Manual

Takes Phase 1 JSON output and builds two searchable indexes:
  1. Spec index  — enriched spec chunks (torque/dimensional values)
  2. Proc index  — split procedure chunks (installation/diagnostic context)

Both indexed with FAISS (dense cosine) + BM25 (keyword).
At query time, HybridRetriever fuses both using Reciprocal Rank Fusion (RRF).

Design decisions vs Copilot proposal:
  - Spec chunks used AS-IS from Phase 1 (avg 72 chars, perfectly sized)
    Re-chunking destroys the structured metadata Phase 1 produced.
  - Context prefix prepended before embedding (inline torque has empty component).
  - Procedure chunks split at step boundaries, max 400 chars (not 150-300
    token fixed windows which break numbered steps mid-sentence).
  - FAISS local, not Pinecone — free, no API key, fits assignment scope.
  - RRF fusion, not linear score weighting — rank-based, more robust.

Usage:
    # Build index
    python chunker_embedder.py build \\
        --spec extracted_text_spec_segments.json \\
        --all  extracted_text_all_segments.json \\
        --out  ./phase2_index --demo

    # Query
    python chunker_embedder.py query \\
        --index ./phase2_index \\
        --query "torque for shock absorber lower nuts 4WD" \\
        --variant 4WD
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import pickle
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase2")

# ── Module-level constants for context enrichment ─────────────────────────────
_VARIANT_NAMES = [
    "SVT Raptor", "Raptor", "FX4", "Lariat", "Platinum",
    "Limited", "King Ranch", "Tremor", "STX", "XLT",
]
_VARIANT_RE = re.compile(
    r'\b(' + '|'.join(re.escape(v) for v in _VARIANT_NAMES) + r')\b'
)
# Matches "tighten/install/remove the <component noun phrase ending in nut/bolt/etc>"
# Used to extract a component name from a procedure-step sentence that mentions a variant.
_COMP_NOUN_RE = re.compile(
    r'(?:not\s+tighten|tighten|install|remove|torque)\s+(?:the\s+)?'
    r'([a-zA-Z][a-zA-Z\s]{2,30}?'
    r'(?:nut|bolt|screw|rod|arm|bar|bearing|mount|plate|bracket|sensor|pin|shaft|cap|stud|assembly|link)s?)\b',
    re.IGNORECASE
)


# =============================================================================
# Data model
# =============================================================================

@dataclass
class Chunk:
    """
    Final embedding unit.  All fields needed by retrieval and the LLM
    are carried inline — no secondary lookup required.
    """
    chunk_id:           str
    chunk_type:         str        # "spec" | "procedure"
    embed_text:         str        # text fed to the embedding model
    display_text:       str        # original text shown to the LLM
    section_id:         str
    section_name:       str
    subsection_type:    str
    page_number:        int
    component:          str
    vehicle_variant:    List[str]
    content_type:       str        # original Phase 1 content_type
    is_safety_critical: bool
    confidence:         float
    spec_value:         Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Stage 1 — Context enrichment
# =============================================================================

class ContextEnricher:
    """
    Builds the embed_text string for each segment.

    Problem: inline_torque segments from procedure steps often have:
        component = ""   (Phase 1 could not extract the noun phrase)
        text      = "z Tighten the nuts to 55 Nm (41 lb-ft)."

    Embedding that bare line produces a near-useless vector — it could
    match any component.  We prepend section + variant + subsection context
    so the embedding captures WHERE and WHAT this spec applies to.

    Table rows already have rich text but still benefit from variant prefix
    reinforcement in the embedding space.
    """

    SPEC_TYPES = {
        "torque_table_row",
        "inline_torque",
        "multi_stage_torque",
        "dimensional_spec",
    }

    def enrich(self, seg: dict, context_hint: str = "") -> str:
        import re as _re
        ctype   = seg.get("content_type", "")
        section = seg.get("section_name", "")
        sub     = seg.get("subsection_type", "")
        variant = seg.get("vehicle_variant", [])
        comp    = seg.get("component", "").strip()
        text    = seg.get("text", "").strip()

        # For inline specs with empty component, inject context derived from nearby segments.
        # Only affects embed_text; Chunk.component field remains empty (Phase 1 truth).
        if not comp and context_hint:
            comp = context_hint

        if ctype not in self.SPEC_TYPES:
            # Procedure / general text — prepend section, amplify variant if present.
            # Variant-amplified procedure chunks give the LLM richer context when it
            # retrieves them alongside the matching spec chunk.
            prefix = f"[{section}]" if section else ""
            vm = _VARIANT_RE.search(text)
            if vm:
                vname = vm.group(1)
                prefix = f"{vname} | {prefix}" if prefix else vname
            return (f"{prefix} {text}").strip() if prefix else text

        # ── Extract parenthetical vehicle-variant tag from component name ────
        # e.g. "Shock absorber lower nut (SVT Raptor)" → paren_variant="SVT Raptor"
        # This is different from seg["vehicle_variant"] which only tracks RWD/4WD
        # extracted from the section name, not from the component itself.
        paren_variant = ""
        pv_m = _re.search(r"\(([A-Za-z][^)]{2,39})\)", comp)
        if pv_m:
            candidate = pv_m.group(1).strip()
            # Exclude unit parens like "(66 lb-ft)" or "(406 lb-ft)" — those
            # start with a digit.  Vehicle names start with a letter.
            if _re.match(r"^[A-Za-z]", candidate):
                paren_variant = candidate

        parts = []
        if section:
            parts.append(f"Section: {section}")
        if variant:
            parts.append(f"Variant: {', '.join(variant)}")
        if sub:
            parts.append(f"Context: {sub}")
        if comp:
            parts.append(f"Component: {comp}")

        prefix = " | ".join(parts)

        # ── Variant amplification ────────────────────────────────────────────
        # Root cause of the SVT Raptor / standard-spec confusion:
        #   Both chunks share 19 identical prefix tokens (section + variant: 4WD).
        #   With BGE mean-pool those tokens represent ~45% of the vector.
        #   "SVT Raptor" at position 35/42 contributes only ~10% — not enough
        #   to pull the SVT chunk out of the dense 3-chunk standard-spec cluster.
        #
        # Fix: put the vehicle-variant tag FIRST so it contributes ~30% of the
        # vector weight.  Repeat it to also boost BM25 TF score.
        # Removing "NOT standard F-150" eliminates tokens that pull toward the
        # standard cluster ("f", "150", "standard" appear in every section name).
        if paren_variant:
            # Strip the "(variant)" from comp for the clean component form
            comp_clean = _re.sub(r"\s*\([^)]+\)\s*$", "", comp).strip()
            prefix = (
                f"{paren_variant} | "
                f"{paren_variant} {comp_clean} | "
                f"{prefix}"
            )
        elif "SVT Raptor" in comp or "SVT Raptor" in text:
            # Fallback for SVT Raptor not in parens
            prefix = f"SVT Raptor | SVT Raptor | {prefix}"

        # For table rows the display_text already starts with "Component: X — Torque: …"
        # Strip leading "Component: X — " from text to avoid "Component: X | Component: X —"
        clean_text = _re.sub(r"^Component:\s*[^—]+—\s*", "", text) if comp else text
        embed_body = clean_text if comp and text.startswith("Component:") else text
        return f"{prefix} | {embed_body}" if prefix else text


# =============================================================================
# Stage 2 — Procedure chunk splitter
# =============================================================================

class ProcedureSplitter:
    """
    Splits large procedure_step segments into embedding-sized sub-chunks.

    Problem: Phase 1 concatenates consecutive procedure steps until a section
    header resets the buffer.  ABS diagnostic sections produce 93K-char
    monsters that exceed bge-base's 512-token limit and produce diluted vectors.

    Strategy:
        1. Split on numbered step boundaries ("1. ", "2. " ... "99. ")
        2. If a step is still > MAX_CHARS, split on sentence boundaries
        3. Apply OVERLAP chars from the previous sub-chunk for context
    """

    MAX_CHARS = 400
    OVERLAP   = 60

    _STEP_RE = re.compile(r"(?=\b\d{1,2}\.\s+\S)")

    def split(self, seg: dict) -> List[dict]:
        text = seg.get("text", "")
        if len(text) <= self.MAX_CHARS:
            return [seg]

        parts = self._STEP_RE.split(text)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) <= 1 or max(len(p) for p in parts) > self.MAX_CHARS * 2:
            parts = self._sentence_split(text)

        sub_segs = []
        for i, part in enumerate(parts):
            overlap_prefix = parts[i - 1][-self.OVERLAP:] + " " if i > 0 else ""
            combined = (overlap_prefix + part).strip()
            for chunk_text in self._hard_split(combined):
                s = dict(seg)
                s["text"] = chunk_text
                sub_segs.append(s)

        return sub_segs if sub_segs else [seg]

    def _sentence_split(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        merged, buf = [], ""
        for s in sentences:
            if len(buf) + len(s) < self.MAX_CHARS:
                buf = (buf + " " + s).strip()
            else:
                if buf:
                    merged.append(buf)
                buf = s
        if buf:
            merged.append(buf)
        return merged if merged else [text]

    def _hard_split(self, text: str) -> List[str]:
        if len(text) <= self.MAX_CHARS:
            return [text]
        parts = []
        while len(text) > self.MAX_CHARS:
            split_at = text.rfind(" ", 0, self.MAX_CHARS)
            if split_at == -1:
                split_at = self.MAX_CHARS
            parts.append(text[:split_at].strip())
            text = text[split_at:].strip()
        if text:
            parts.append(text)
        return parts


# =============================================================================
# Stage 3 — Chunk builder
# =============================================================================

class ChunkBuilder:
    """Converts Phase 1 segments into final Chunk objects."""

    SPEC_TYPES = ContextEnricher.SPEC_TYPES

    def __init__(self):
        self.enricher = ContextEnricher()
        self.splitter = ProcedureSplitter()

    def build_spec_chunks(self, spec_segments: List[dict],
                          all_segments: List[dict] = None) -> List[Chunk]:
        # Build context lookup for inline specs with empty component fields.
        context_lookup: dict = {}
        if all_segments:
            context_lookup = self._build_context_lookup(spec_segments, all_segments)
            if context_lookup:
                log.info("Context lookup: %d empty-component specs enriched", len(context_lookup))

        chunks = []
        for i, seg in enumerate(spec_segments):
            ln   = seg.get("line_number", 0)
            hint = context_lookup.get(ln, "")
            chunks.append(Chunk(
                chunk_id=f"spec_{i:04d}",
                chunk_type="spec",
                embed_text=self.enricher.enrich(seg, context_hint=hint),
                display_text=seg.get("text", ""),
                section_id=seg.get("section_id", ""),
                section_name=seg.get("section_name", ""),
                subsection_type=seg.get("subsection_type", ""),
                page_number=seg.get("page_number", 0),
                component=seg.get("component", ""),
                vehicle_variant=seg.get("vehicle_variant", []),
                content_type=seg.get("content_type", ""),
                is_safety_critical=seg.get("is_safety_critical", False),
                confidence=seg.get("confidence", 1.0),
                spec_value=seg.get("spec_value"),
            ))
        log.info("Built %d spec chunks", len(chunks))
        return chunks

    @staticmethod
    def _build_context_lookup(spec_segs: List[dict],
                               all_segs: List[dict]) -> dict:
        """
        For spec segments with empty component, derive context from surrounding
        all_segs entries using three-pass strategy:

          Pass 1 — Value match: if a torque_table_row in the SAME section has the
                   same Nm value (within 5 %), borrow its component name.
                   Handles inline_torque duplicates of table rows (e.g. 80 Nm
                   stabilizer bar link nuts SVT Raptor).

          Pass 2 — Backward field scan (≤30 lines): nearest all_segs segment
                   whose component FIELD is non-empty.

          Pass 3 — Backward text scan (≤150 lines): nearest procedure_step text
                   containing a known vehicle-variant name, with a component noun
                   phrase extracted from the same sentence.  Forward scan (≤40
                   lines) used as fallback when variant only appears after.

        "Deferred-tighten" refinement:
          Some procedures say "do not tighten X until ride height is correct" at
          step N, then tighten at step N+k.  The inline_torque that is LAST in
          its section among those that triggered a text-scan gets the full
          "phrase (variant)" context; earlier siblings are downgraded to
          "variant-only" — preventing them from stealing the specific component
          phrase that belongs to the deferred step.

        Returns {line_number: context_hint_string}
        """
        CLOSE_WIN  = 30
        BACK_WIN   = 150
        FWD_WIN    = 40
        VAL_TOL    = 0.05   # 5% tolerance for value matching

        # ── Index table rows by (section_id, value_nm) for Pass 1 ────────────
        table_idx: dict = defaultdict(list)   # section_id → [(value_nm, component, line_no)]
        for s in spec_segs:
            if (s.get("component", "").strip()
                    and s.get("content_type") == "torque_table_row"):
                sid = s.get("section_id", "")
                val = (s.get("spec_value") or {}).get("value_nm")
                if val:
                    table_idx[sid].append((float(val), s["component"], s.get("line_number", 0)))

        # ── Sort all_segs by line_number for bisect lookups ───────────────────
        sorted_all = sorted(all_segs, key=lambda s: s.get("line_number", 0))
        all_lns    = [s.get("line_number", 0) for s in sorted_all]

        lookup:  dict = {}          # ln → hint string
        sources: dict = {}          # ln → 'value_match' | 'field_scan' | 'text_scan'

        for seg in spec_segs:
            if seg.get("component", "").strip():
                continue            # already has component

            ln  = seg.get("line_number", 0)
            sid = seg.get("section_id", "")
            val = (seg.get("spec_value") or {}).get("value_nm")

            if not ln:
                continue

            # ── Pass 1: value-based component match ───────────────────────────
            hint = ""
            if val is not None and sid in table_idx:
                candidates = [
                    (abs(float(val) - tv) / max(float(tv), 1), tc, tln)
                    for tv, tc, tln in table_idx[sid]
                    if abs(float(val) - float(tv)) / max(float(tv), 1) <= VAL_TOL
                ]
                if candidates:
                    # Pick closest by Nm distance, break ties by line proximity
                    candidates.sort(key=lambda x: (x[0], abs(x[2] - ln)))
                    hint = candidates[0][1]
                    sources[ln] = "value_match"

            if hint:
                lookup[ln] = hint
                continue

            # ── Passes 2 & 3: context window scan ────────────────────────────
            lo = bisect.bisect_left(all_lns,  ln - BACK_WIN)
            hi = bisect.bisect_right(all_lns, ln + FWD_WIN)

            before = sorted(
                [s for s in sorted_all[lo:hi] if s.get("line_number", 0) < ln],
                key=lambda s: s.get("line_number", 0), reverse=True   # nearest-first
            )
            after = [s for s in sorted_all[lo:hi] if s.get("line_number", 0) > ln]

            found_comp    = ""
            found_variant = ""
            found_phrase  = ""

            # Pass 2: backward component FIELD scan (tight window)
            for s in before:
                if ln - s.get("line_number", 0) > CLOSE_WIN:
                    break
                cf = s.get("component", "").strip()
                if cf:
                    found_comp = cf
                    sources[ln] = "field_scan"
                    break

            # Pass 3: backward text scan for variant + component phrase.
            # Multiple segments may share the same line_number (ProcedureSplitter
            # chunks).  Scan ALL segments at each distance level; prefer a segment
            # that has BOTH a variant name AND a component noun phrase over one
            # that has only the variant name (e.g. "SVT Raptor similar").
            if not found_comp:
                # Group before-segments by line_number so we can inspect all
                # chunks at the same distance together.
                by_lineno: dict = {}
                for s in before:
                    sln = s.get("line_number", 0)
                    by_lineno.setdefault(sln, []).append(s)

                found_source_ln = 0
                for dist_ln in sorted(by_lineno.keys(), reverse=True):  # nearest first
                    group = by_lineno[dist_ln]
                    # First pass within group: prefer segments with BOTH variant AND phrase
                    for s in group:
                        text_f = s.get("text", "")
                        vm = _VARIANT_RE.search(text_f)
                        if vm:
                            pm = _COMP_NOUN_RE.search(text_f)
                            if pm:
                                found_variant   = vm.group(1)
                                found_phrase    = pm.group(1).strip()
                                found_source_ln = dist_ln
                                break
                    if found_variant and found_phrase:
                        break
                    # Second pass within group: variant-only if no phrase found here
                    if not found_variant:
                        for s in group:
                            text_f = s.get("text", "")
                            vm = _VARIANT_RE.search(text_f)
                            if vm:
                                found_variant   = vm.group(1)
                                found_source_ln = dist_ln
                                break
                        if found_variant:
                            break   # stop at the nearest level that has a variant

                # Forward fallback — variant sometimes appears in next sub-section header
                if not found_variant:
                    for s in after:
                        text_f = s.get("text", "")
                        vm = _VARIANT_RE.search(text_f)
                        if vm:
                            found_variant   = vm.group(1)
                            found_source_ln = s.get("line_number", 0)
                            break

                if found_variant or found_phrase:
                    # Tag with (type, source_line) so the deferred-tighten step
                    # can group by the exact notice that triggered the context.
                    sources[ln] = ("text_scan", found_source_ln)

            # Build preliminary hint
            if found_comp:
                if found_variant and found_variant not in found_comp:
                    hint = f"{found_comp} ({found_variant})"
                else:
                    hint = found_comp
            elif found_phrase:
                hint = f"{found_phrase} ({found_variant})" if found_variant else found_phrase
            elif found_variant:
                hint = found_variant

            if hint:
                lookup[ln] = hint

        # ── Deferred-tighten refinement ───────────────────────────────────────
        # Some procedures say "do not tighten X until ride height is correct" at
        # step N, then tighten X at step N+k (the deferred step).  Multiple
        # inline_torques in between (step N+1 … N+k-1) are for OTHER components
        # but will scan back to the same SVT Raptor NOTICE (same source line).
        #
        # Group text_scan entries by SOURCE LINE (the all_segs segment that
        # provided the variant+phrase).  Within each source group, only the LAST
        # inline_torque (highest line_number) keeps the full "phrase (variant)"
        # context — it is the deferred-tighten step.  Earlier siblings are
        # downgraded to "variant-only" so they don't wrongly rank for queries
        # about the deferred component (e.g. "shock absorber lower nut SVT Raptor").
        text_scan_by_source: dict = defaultdict(list)

        for ln, src in sources.items():
            if isinstance(src, tuple) and src[0] == "text_scan" and ln in lookup:
                source_ln = src[1]
                text_scan_by_source[source_ln].append(ln)

        for source_ln, lns in text_scan_by_source.items():
            if len(lns) <= 1:
                continue
            lns_sorted = sorted(lns)
            for earlier_ln in lns_sorted[:-1]:
                hint = lookup.get(earlier_ln, "")
                vm   = _VARIANT_RE.search(hint)
                if vm:
                    lookup[earlier_ln] = vm.group(1)   # downgrade to variant-only
                    log.debug(
                        "Deferred-tighten: ln=%d downgraded to '%s' (source ln=%d)",
                        earlier_ln, vm.group(1), source_ln,
                    )

        return lookup

    def build_proc_chunks(self, all_segments: List[dict]) -> List[Chunk]:
        proc_segs = [s for s in all_segments if s.get("content_type") == "procedure_step"]
        chunks, idx = [], 0
        for seg in proc_segs:
            for sub in self.splitter.split(seg):
                text = sub.get("text", "").strip()
                if len(text) < 20:
                    continue
                chunks.append(Chunk(
                    chunk_id=f"proc_{idx:05d}",
                    chunk_type="procedure",
                    embed_text=self.enricher.enrich(sub),
                    display_text=text,
                    section_id=sub.get("section_id", ""),
                    section_name=sub.get("section_name", ""),
                    subsection_type=sub.get("subsection_type", ""),
                    page_number=sub.get("page_number", 0),
                    component=sub.get("component", ""),
                    vehicle_variant=sub.get("vehicle_variant", []),
                    content_type="procedure_step",
                    is_safety_critical=sub.get("is_safety_critical", False),
                    confidence=sub.get("confidence", 1.0),
                    spec_value=None,
                ))
                idx += 1
        log.info("Built %d proc chunks from %d procedure segments", len(chunks), len(proc_segs))
        return chunks


# =============================================================================
# Stage 4 — Embedding
# =============================================================================

class EmbeddingEngine:
    """
    BAAI/bge-base-en-v1.5 wrapper.

    Why bge-base-en-v1.5:
      - 768-dim, outperforms all-MiniLM on technical domain retrieval
      - Trained with instruction prefix for queries (not documents)
      - Free, local, no API key, 109M params
    """

    MODEL_NAME  = "BAAI/bge-base-en-v1.5"
    INSTRUCTION = "Represent this automotive specification for retrieval: "

    def __init__(self):
        log.info("Loading %s …", self.MODEL_NAME)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.MODEL_NAME)
        log.info("Model ready")

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return np.array(
            self.model.encode(texts, batch_size=batch_size,
                              normalize_embeddings=True, show_progress_bar=True),
            dtype="float32",
        )

    def encode_query(self, query: str) -> np.ndarray:
        return np.array(
            self.model.encode([self.INSTRUCTION + query], normalize_embeddings=True),
            dtype="float32",
        )


# =============================================================================
# Stage 5 — FAISS index
# =============================================================================

class FAISSIndexer:
    """
    Flat cosine-similarity FAISS index (IndexFlatIP on L2-normalised vectors).
    Exact search — no approximation.  Fine for 200-5000 chunks.
    """

    def build(self, embeddings: np.ndarray):
        import faiss
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        log.info("FAISS: %d vectors, dim=%d", index.ntotal, embeddings.shape[1])
        return index

    def search(self, index, query_emb: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        scores, ids = index.search(query_emb, k)
        return ids[0].tolist(), scores[0].tolist()


# =============================================================================
# Stage 6 — BM25 index
# =============================================================================

class BM25Indexer:
    """
    Keyword retriever using BM25Okapi.

    Why BM25 is strong for this data:
      - Component names are specific noun phrases: "shock absorber lower nuts"
      - User queries often contain exact words from the component name
      - BM25 handles exact token matches perfectly
      - Keeps numeric values ("350", "66", "90 Nm") as match tokens
    """

    def build(self, texts: List[str]):
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([self._tok(t) for t in texts])
        log.info("BM25: %d documents indexed", len(texts))
        return bm25

    def search(self, bm25, query: str, k: int) -> Tuple[List[int], List[float]]:
        scores = bm25.get_scores(self._tok(query))
        top_ids = np.argsort(scores)[::-1][:k].tolist()
        return top_ids, [float(scores[i]) for i in top_ids]

    @staticmethod
    def _tok(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())


# =============================================================================
# Stage 7 — Hybrid retriever (RRF fusion)
# =============================================================================

class HybridRetriever:
    """
    Fuses BM25 + FAISS results using Reciprocal Rank Fusion (RRF).

    RRF score(doc) = sum over retrievers of  1 / (k + rank(doc))
    k = 60 (standard dampening constant).

    Why RRF over linear score combination:
      - BM25 scores are unbounded (corpus-size-dependent)
      - FAISS cosine scores are in [-1, 1]
      - Normalising + weighting them together is fragile
      - RRF only uses rank, not magnitude — robust and simple
    """

    K = 60

    def __init__(self, spec_chunks, proc_chunks,
                 spec_faiss, proc_faiss, spec_bm25, proc_bm25, embedder):
        self.spec_chunks = spec_chunks
        self.proc_chunks = proc_chunks
        self.spec_faiss  = spec_faiss
        self.proc_faiss  = proc_faiss
        self.spec_bm25   = spec_bm25
        self.proc_bm25   = proc_bm25
        self.embedder    = embedder
        self._fi = FAISSIndexer()
        self._bi = BM25Indexer()

    def retrieve(self, query: str, k: int = 5,
                 spec_only: bool = False,
                 vehicle_variant: Optional[str] = None) -> List[dict]:
        qemb = self.embedder.encode_query(query)

        spec_res = self._search_one(query, qemb, self.spec_faiss,
                                    self.spec_bm25, self.spec_chunks, k * 4)
        proc_res = [] if spec_only else self._search_one(
            query, qemb, self.proc_faiss, self.proc_bm25, self.proc_chunks, k * 2)

        fused = self._rrf(spec_res, proc_res)

        if vehicle_variant:
            fused = [r for r in fused
                     if not r["vehicle_variant"]
                     or vehicle_variant in r["vehicle_variant"]]

        return fused[:k]

    def _search_one(self, query, qemb, faiss_idx, bm25_idx, chunks, k) -> List[dict]:
        if not chunks:
            return []
        faiss_ids, _ = self._fi.search(faiss_idx, qemb, k)
        bm25_ids,  _ = self._bi.search(bm25_idx, query, k)

        rrf: Dict[int, float] = {}
        for rank, idx in enumerate(faiss_ids):
            if 0 <= idx < len(chunks):
                rrf[idx] = rrf.get(idx, 0) + 1 / (self.K + rank + 1)
        for rank, idx in enumerate(bm25_ids):
            if 0 <= idx < len(chunks):
                rrf[idx] = rrf.get(idx, 0) + 1 / (self.K + rank + 1)

        results = []
        for idx in sorted(rrf, key=rrf.__getitem__, reverse=True)[:k]:
            d = chunks[idx].to_dict()
            d["rrf_score"] = round(rrf[idx], 6)
            results.append(d)
        return results

    def _rrf(self, spec_res: List[dict], proc_res: List[dict]) -> List[dict]:
        """
        Cross-index RRF fusion: rank spec results against proc results.

        CORRECT approach: use only the RANK POSITION in each result list.
        The inner rrf_score from _search_one is already baked into the
        ordering of spec_res and proc_res — adding it again here causes
        spec chunks to score 3x higher than equally-relevant proc chunks
        (double-counting their within-index score on top of their rank bonus).
        """
        combined: Dict[str, dict] = {}
        for rank, r in enumerate(spec_res):
            cid = r["chunk_id"]
            combined[cid] = r
            combined[cid].setdefault("_final_rrf", 0)
            combined[cid]["_final_rrf"] += 1 / (self.K + rank + 1)
        for rank, r in enumerate(proc_res):
            cid = r["chunk_id"]
            if cid not in combined:
                combined[cid] = r
                combined[cid]["_final_rrf"] = 0
            combined[cid]["_final_rrf"] += 1 / (self.K + rank + 1)
        return sorted(combined.values(), key=lambda x: x["_final_rrf"], reverse=True)


# =============================================================================
# Pipeline orchestrator
# =============================================================================

class Phase2Pipeline:
    """
    Orchestrates all stages and saves/loads the index bundle.

    Output layout:
        phase2_index/
            spec_chunks.json     enriched spec chunks
            proc_chunks.json     split procedure chunks
            spec_faiss.index     FAISS binary
            proc_faiss.index     FAISS binary
            spec_bm25.pkl        BM25 model
            proc_bm25.pkl        BM25 model
            build_stats.json     stats
    """

    def __init__(self, out_dir: str | Path = "./phase2_index"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def build(self, spec_json: str, all_json: str) -> dict:
        import faiss

        log.info("=" * 60)
        log.info("Phase 2 — Build")
        log.info("=" * 60)

        log.info("[1/5] Loading Phase 1 outputs …")
        spec_segs = json.loads(Path(spec_json).read_text())
        all_segs  = json.loads(Path(all_json).read_text())
        log.info("      spec=%d  all=%d", len(spec_segs), len(all_segs))

        log.info("[2/5] Building chunks …")
        builder     = ChunkBuilder()
        spec_chunks = builder.build_spec_chunks(spec_segs, all_segs)
        proc_chunks = builder.build_proc_chunks(all_segs)

        proc_lens = [len(c.embed_text) for c in proc_chunks]
        log.info("      proc: %d chunks | avg %d | max %d chars",
                 len(proc_chunks),
                 sum(proc_lens) // max(1, len(proc_lens)),
                 max(proc_lens) if proc_lens else 0)

        log.info("[3/5] Embedding …")
        emb = EmbeddingEngine()
        spec_embs = emb.encode([c.embed_text for c in spec_chunks])
        proc_embs = emb.encode([c.embed_text for c in proc_chunks])

        log.info("[4/5] Indexing …")
        fi = FAISSIndexer()
        bi = BM25Indexer()
        spec_faiss = fi.build(spec_embs)
        proc_faiss = fi.build(proc_embs)
        spec_bm25  = bi.build([c.embed_text for c in spec_chunks])
        proc_bm25  = bi.build([c.embed_text for c in proc_chunks])

        log.info("[5/5] Saving …")
        self._jdump([c.to_dict() for c in spec_chunks], "spec_chunks.json")
        self._jdump([c.to_dict() for c in proc_chunks], "proc_chunks.json")
        faiss.write_index(spec_faiss, str(self.out_dir / "spec_faiss.index"))
        faiss.write_index(proc_faiss, str(self.out_dir / "proc_faiss.index"))
        for name, obj in [("spec_bm25.pkl", spec_bm25), ("proc_bm25.pkl", proc_bm25)]:
            with open(self.out_dir / name, "wb") as f:
                pickle.dump(obj, f)

        stats = {
            "spec_chunks": len(spec_chunks),
            "proc_chunks": len(proc_chunks),
            "embedding_model": EmbeddingEngine.MODEL_NAME,
            "embedding_dim": int(spec_embs.shape[1]),
            "spec_avg_chars": sum(len(c.embed_text) for c in spec_chunks) // max(1, len(spec_chunks)),
            "proc_avg_chars": sum(proc_lens) // max(1, len(proc_lens)),
            "proc_max_chars": max(proc_lens) if proc_lens else 0,
        }
        self._jdump(stats, "build_stats.json")
        self._print_summary(spec_chunks, proc_chunks, stats)
        return stats

    def load_retriever(self) -> HybridRetriever:
        import faiss
        spec_chunks = [Chunk(**c) for c in json.loads((self.out_dir / "spec_chunks.json").read_text())]
        proc_chunks = [Chunk(**c) for c in json.loads((self.out_dir / "proc_chunks.json").read_text())]
        spec_faiss  = faiss.read_index(str(self.out_dir / "spec_faiss.index"))
        proc_faiss  = faiss.read_index(str(self.out_dir / "proc_faiss.index"))
        with open(self.out_dir / "spec_bm25.pkl", "rb") as f: spec_bm25 = pickle.load(f)
        with open(self.out_dir / "proc_bm25.pkl", "rb") as f: proc_bm25 = pickle.load(f)
        log.info("Index loaded: %d spec, %d proc chunks", len(spec_chunks), len(proc_chunks))
        return HybridRetriever(spec_chunks, proc_chunks,
                               spec_faiss, proc_faiss, spec_bm25, proc_bm25,
                               EmbeddingEngine())

    def _jdump(self, data, name: str):
        p = self.out_dir / name
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        log.info("      → %s (%d KB)", name, p.stat().st_size // 1024)

    @staticmethod
    def _print_summary(spec_chunks, proc_chunks, stats):
        from collections import Counter
        print("\n" + "=" * 60)
        print("  PHASE 2 BUILD SUMMARY")
        print("=" * 60)
        print(f"\n  Model     : {stats['embedding_model']}  (dim={stats['embedding_dim']})")
        print(f"  Spec index: {stats['spec_chunks']} chunks | avg {stats['spec_avg_chars']} chars")
        print(f"  Proc index: {stats['proc_chunks']} chunks | avg {stats['proc_avg_chars']} chars")
        print(f"             (was up to 93,626 chars before split — now max {stats['proc_max_chars']})")
        print(f"\n  Spec chunks by content type:")
        for t, n in Counter(c.content_type for c in spec_chunks).most_common():
            print(f"    {t:<26} {n:>4}  {'█' * min(n // 3, 25)}")
        print(f"\n  Sample enriched embed texts:")
        for c in spec_chunks[:3]:
            print(f"    [{c.content_type}]  {c.embed_text[:110]}")
        print()


# =============================================================================
# Demo queries
# =============================================================================

def run_demo(retriever: HybridRetriever):
    queries = [
        ("Torque for brake caliper bolts",              None,    True),
        ("Shock absorber lower nuts torque 4WD",        "4WD",   True),
        ("How to install lower control arm",            None,    False),
        ("Stage tightening sequence U-bolt nuts",       None,    True),
        ("Wheel bearing hub bolt torque specification", None,    True),
    ]
    print("\n" + "=" * 60 + "\n  DEMO QUERIES\n" + "=" * 60)
    for query, variant, spec_only in queries:
        print(f"\n  Query : {query}")
        if variant: print(f"  Filter: variant={variant}")
        results = retriever.retrieve(query, k=3, spec_only=spec_only, vehicle_variant=variant)
        for i, r in enumerate(results, 1):
            score = r.get("rrf_score", r.get("_final_rrf", 0))
            print(f"  [{i}] {r['content_type']:<25} sec={r['section_id']:<8} rrf={score:.4f}")
            print(f"       {r['display_text'][:100]}")
            sv = r.get("spec_value") or {}
            if sv.get("value_nm"):
                vals = [f"{sv['value_nm']} Nm"]
                if sv.get("value_lbft"): vals.append(f"{sv['value_lbft']} lb-ft")
                if sv.get("value_lbin"): vals.append(f"{sv['value_lbin']} lb-in")
                print(f"       Spec: {' / '.join(vals)}")
    print()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 2: Chunking + Embedding + Index")
    sub = p.add_subparsers(dest="cmd")

    bp = sub.add_parser("build")
    bp.add_argument("--spec", required=True)
    bp.add_argument("--all",  required=True)
    bp.add_argument("--out",  default="./phase2_index")
    bp.add_argument("--demo", action="store_true")

    qp = sub.add_parser("query")
    qp.add_argument("--index",    default="./phase2_index")
    qp.add_argument("--query",    required=True)
    qp.add_argument("--k",        type=int, default=5)
    qp.add_argument("--variant",  default=None)
    qp.add_argument("--spec-only",action="store_true")
    qp.add_argument("--demo",     action="store_true")

    args = p.parse_args()

    if args.cmd == "build":
        pipe = Phase2Pipeline(args.out)
        pipe.build(args.spec, args.all)
        if args.demo:
            run_demo(pipe.load_retriever())

    elif args.cmd == "query":
        pipe = Phase2Pipeline(args.index)
        r = pipe.load_retriever()
        if args.demo:
            run_demo(r)
        else:
            print(json.dumps(
                r.retrieve(args.query, k=args.k,
                           spec_only=args.spec_only, vehicle_variant=args.variant),
                indent=2, default=str))
    else:
        p.print_help()