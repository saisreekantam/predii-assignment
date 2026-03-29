"""
Phase 3: LLM-Powered Specification Extraction
==============================================
Vehicle Specification Extraction — 2014 F-150 Workshop Manual

Takes a natural language query, retrieves top chunks from Phase 2 indexes,
and returns structured JSON matching the assignment output format:
    [{"component": "...", "spec_type": "...", "value": "...", "unit": "..."}]

Architecture — two extraction paths:

  FAST PATH (no LLM):
    When retrieved chunks are torque_table_row / dimensional_spec with a
    parsed spec_value and a known component, we already have the numbers.
    Just reformat. Zero latency, zero cost, zero hallucination risk.

  LLM PATH (Llama 3.1 8B via Ollama):
    When chunks are inline_torque (empty component), multi_stage sequences,
    or the query needs synthesis. Chain-of-thought prompt extracts the spec.

Post-extraction:
  SpecValidator — deduplicates, normalises units, resolves variants,
                  flags conflicts between sources

Output:
  JSON — assignment required format + extended fields
  CSV  — tabular format for spreadsheet review

Usage:
    python extractor.py demo                               # 8 built-in queries
    python extractor.py query "torque for brake caliper"  # single query
    python extractor.py query "shock absorber 4WD" --variant 4WD
    python extractor.py batch queries.txt --out results.json
    python extractor.py interactive                        # REPL mode
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase3")


# =============================================================================
# Output data model
# =============================================================================

@dataclass
class SpecResult:
    """
    One extracted specification.
    Core 4 fields match the assignment output exactly.
    Extended fields provide full provenance for evaluation.
    """
    # assignment required
    component:  str
    spec_type:  str   # Torque | Dimension | Alignment | Capacity | Other
    value:      str   # string — preserves precision and ranges ("1.5-3.3")
    unit:       str   # Nm | lb-ft | lb-in | mm | in | degrees

    # extended
    section_id:         str       = ""
    section_name:       str       = ""
    vehicle_variant:    List[str] = field(default_factory=list)
    is_safety_critical: bool      = False
    stage_number:       int       = 0
    source:             str       = ""   # table | llm | multi_stage
    confidence:         float     = 1.0
    raw_text:           str       = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_assignment_dict(self) -> dict:
        return {
            "component": self.component,
            "spec_type": self.spec_type,
            "value":     self.value,
            "unit":      self.unit,
        }


# =============================================================================
# Ollama client
# =============================================================================

class OllamaClient:
    """
    Thin wrapper around the Ollama REST API.

    Why Ollama over HuggingFace direct:
      - Models stay resident between queries (no per-query reload)
      - format="json" uses grammar-constrained sampling → zero JSON parse errors
      - Apple MPS acceleration automatic on Mac
      - No GPU memory management code needed

    Model: llama3.1:8b
      - 4.7 GB, fits comfortably in 24 GB RAM alongside FAISS indexes
      - Strong JSON instruction following at 8B scale
      - Assignment explicitly suggests Llama-3
    """

    def __init__(self, model: str = "llama3.1:8b",
                 host: str = "http://localhost:11434", timeout: int = 60):
        self.model   = model
        self.host    = host
        self.timeout = timeout
        self._verify()

    def _verify(self):
        import urllib.request
        try:
            resp = urllib.request.urlopen(f"{self.host}/api/tags", timeout=4)
            data = json.loads(resp.read())
            names = [m["name"] for m in data.get("models", [])]
            base  = self.model.split(":")[0]
            if not any(base in n for n in names):
                log.warning("'%s' not in Ollama. Available: %s", self.model, names)
                log.warning("Pull with: ollama pull %s", self.model)
            else:
                log.info("Ollama ready — %s", self.model)
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}.\n"
                f"Start it with: ollama serve\nError: {e}"
            )

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        import urllib.request, urllib.error
        payload = json.dumps({
            "model":   self.model,
            "prompt":  prompt,
            "format":  "json",      # grammar-constrained → always valid JSON
            "stream":  False,
            "options": {"temperature": temperature, "num_predict": 600, "top_p": 0.9},
        }).encode()

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read()).get("response", "")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama request failed: {e}")


# =============================================================================
# Prompt builder
# =============================================================================

class PromptBuilder:
    """
    Builds chain-of-thought extraction prompts for Llama 3.1 8B.

    Design:
    - Few-shot examples cover all content types (table, inline, stage, dimension)
    - "Think step by step" improves component identification for inline specs
    - JSON schema is explicit so grammar sampling reinforces it
    - Model instructed to return confidence=0 when spec is not found
      rather than hallucinate
    """

    SYSTEM = (
        "You are a precise automotive service manual parser. "
        "Extract vehicle specifications from the given context. "
        "Return ONLY valid JSON. Never add text outside the JSON structure. "
        "If a value cannot be confirmed in the context, set confidence to 0."
    )

    EXAMPLES = """Examples:

Input: "Component: Brake caliper guide pin bolts — Torque: 37.0 Nm — (27.0 lb-ft)"
Query: "brake caliper bolt torque"
Output: {"specs": [{"component": "Brake caliper guide pin bolts", "spec_type": "Torque", "value": "37", "unit": "Nm"}, {"component": "Brake caliper guide pin bolts", "spec_type": "Torque", "value": "27", "unit": "lb-ft"}], "confidence": 0.98, "reasoning": "Found exact table row for brake caliper guide pin bolts."}

Input: "Section: Front Suspension RWD | z Tighten the new nut to 350 Nm (258 lb-ft)."
Query: "lower arm forward rearward nuts torque"
Output: {"specs": [{"component": "Lower arm forward and rearward nuts", "spec_type": "Torque", "value": "350", "unit": "Nm"}, {"component": "Lower arm forward and rearward nuts", "spec_type": "Torque", "value": "258", "unit": "lb-ft"}], "confidence": 0.85, "reasoning": "Inline tighten instruction in front suspension RWD section."}

Input: "Stage 1: Tighten in a cross pattern to 35 Nm (26 lb-ft).\nStage 2: Tighten in a cross pattern to 70 Nm (52 lb-ft).\nStage 3: Tighten in a cross pattern to 100 Nm (74 lb-ft)."
Query: "U-bolt nut tightening sequence"
Output: {"specs": [{"component": "U-bolt nuts stage 1", "spec_type": "Torque", "value": "35", "unit": "Nm"}, {"component": "U-bolt nuts stage 2", "spec_type": "Torque", "value": "70", "unit": "Nm"}, {"component": "U-bolt nuts stage 3", "spec_type": "Torque", "value": "100", "unit": "Nm"}], "confidence": 0.96, "reasoning": "Found 3-stage cross-pattern tightening sequence."}

Input: "Left halfshaft assembled length: 406.45 mm (16.00 in)"
Query: "halfshaft length specification"
Output: {"specs": [{"component": "Left halfshaft", "spec_type": "Dimension", "value": "406.45", "unit": "mm"}], "confidence": 0.99, "reasoning": "Exact dimensional spec for left halfshaft."}"""

    def build(self, query: str, chunks: List[dict]) -> str:
        ctx_parts = []
        for i, c in enumerate(chunks, 1):
            ctype = c.get("content_type", "")
            comp  = c.get("component", "").strip()
            sec   = c.get("section_id", "")
            var   = ", ".join(c.get("vehicle_variant", [])) or "All variants"

            # For inline_torque/spec with empty component, the display_text is a
            # bare "Tighten to X Nm" instruction — no component name.  Use the
            # enriched embed_text instead so the LLM gets the variant + component
            # context injected by Phase 2 (e.g. "SVT Raptor lower shock nut | …").
            if not comp and ctype in ("inline_torque", "multi_stage_torque", "dimensional_spec"):
                text = c.get("embed_text") or c.get("display_text", "")
            else:
                text = c.get("display_text") or c.get("embed_text", "")

            ctx_parts.append(f"[{i}] {ctype} | section={sec} | variant={var}\n    {text}")

        context = "\n".join(ctx_parts)

        return f"""{self.SYSTEM}

{self.EXAMPLES}

Now extract from this context:

USER QUERY: {query}

RETRIEVED CONTEXT:
{context}

Think step by step:
1. Which context blocks are relevant to this query?
2. What is the exact component name? (look in the text itself)
3. What are the exact numeric values and units?
4. For multi-stage specs, number each stage in the component name
5. Only include specs directly relevant to the query

Return this exact JSON structure:
{{
  "specs": [
    {{
      "component": "exact component name",
      "spec_type": "Torque or Dimension or Pressure or Capacity",
      "value": "numeric value as string",
      "unit": "Nm or lb-ft or lb-in or mm or in or degrees"
    }}
  ],
  "confidence": 0.0,
  "reasoning": "one sentence summary"
}}"""


# =============================================================================
# Fast path extractor
# =============================================================================

class FastPathExtractor:
    """
    Extracts specs directly from pre-parsed spec_value fields — no LLM needed.

    Eligible: torque_table_row, multi_stage_torque, dimensional_spec
              — must have a non-empty component AND at least one parsed value.

    Not eligible (→ LLM):
      - inline_torque with empty component (all 109 in this dataset)
      - chunks where spec_value is null
    """

    STYPE = {
        "torque_table_row":   "Torque",
        "multi_stage_torque": "Torque",
        "dimensional_spec":   "Dimension",
        "inline_torque":      "Torque",
    }

    def is_eligible(self, chunk: dict) -> bool:
        sv   = chunk.get("spec_value") or {}
        comp = chunk.get("component", "").strip()
        has  = any(sv.get(k) is not None for k in [
            "value_nm", "value_lbft", "value_lbin", "value_mm", "value_deg"
        ])
        return bool(comp) and has

    def extract(self, chunk: dict) -> List[SpecResult]:
        sv    = chunk.get("spec_value") or {}
        comp  = chunk.get("component", "").strip()
        base  = dict(
            component=comp,
            spec_type=self.STYPE.get(chunk.get("content_type", ""), "Other"),
            section_id=chunk.get("section_id", ""),
            section_name=chunk.get("section_name", ""),
            vehicle_variant=chunk.get("vehicle_variant", []),
            is_safety_critical=chunk.get("is_safety_critical", False),
            stage_number=chunk.get("stage_number", 0),
            source="table",
            confidence=chunk.get("confidence", 0.98),
            raw_text=chunk.get("display_text", ""),
        )

        results = []
        nm, lbft, lbin, mm, deg = (
            sv.get("value_nm"), sv.get("value_lbft"), sv.get("value_lbin"),
            sv.get("value_mm"),  sv.get("value_deg"),
        )

        if nm   is not None: results.append(SpecResult(value=str(nm),   unit="Nm",      **base))
        if lbft is not None: results.append(SpecResult(value=str(lbft), unit="lb-ft",   **base))
        if lbin is not None: results.append(SpecResult(value=str(lbin), unit="lb-in",   **base))
        if mm   is not None and not results:
            dim_base = {**base, "spec_type": "Dimension"}
            results.append(SpecResult(value=str(mm),  unit="mm",      **dim_base))
        if deg  is not None and not results:
            aln_base = {**base, "spec_type": "Alignment"}
            results.append(SpecResult(value=str(deg), unit="degrees", **aln_base))

        return results


# =============================================================================
# LLM extractor
# =============================================================================

class LLMExtractor:
    """
    Uses Llama 3.1 8B via Ollama for chunks the fast path can't handle.
    All ineligible chunks are batched into ONE LLM call per query.
    temperature=0 → deterministic, reproducible output.
    """

    def __init__(self, client: OllamaClient):
        self.client  = client
        self.builder = PromptBuilder()

    def extract(
        self, query: str, chunks: List[dict],
        sec_id: str = "", sec_nm: str = "", var: Optional[List[str]] = None,
    ) -> Tuple[List[SpecResult], float, str]:

        prompt = self.builder.build(query, chunks)
        t0     = time.time()
        raw    = self.client.generate(prompt)
        log.debug("LLM response in %.1fs", time.time() - t0)

        return self._parse(raw, chunks, sec_id, sec_nm, var or [])

    def _parse(self, raw: str, chunks: List[dict],
               sec_id: str, sec_nm: str, var: List[str]):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                try:    data = json.loads(m.group())
                except: return [], 0.0, "JSON parse failed"
            else:
                log.warning("LLM non-JSON response: %s", raw[:200])
                return [], 0.0, "No JSON in response"

        confidence = float(data.get("confidence", 0.8))
        reasoning  = data.get("reasoning", "")
        results    = []

        for s in data.get("specs", []):
            comp  = s.get("component", "").strip()
            value = str(s.get("value", "")).strip()
            unit  = s.get("unit", "").strip()
            stype = s.get("spec_type", "Torque")
            if comp and value and unit:
                results.append(SpecResult(
                    component=comp, spec_type=stype, value=value, unit=unit,
                    section_id=sec_id, section_name=sec_nm,
                    vehicle_variant=var, source="llm",
                    confidence=confidence,
                    raw_text=chunks[0].get("display_text", "") if chunks else "",
                ))

        return results, confidence, reasoning


# =============================================================================
# Spec validator
# =============================================================================

class SpecValidator:
    """
    Post-processes raw SpecResult list:

    1. Value cleaning      — "90.0" → "90", "1.5" stays
    2. Unit normalisation  — aliases → canonical form
    3. Deduplication       — same (component_lower, unit) → keep highest confidence
    4. Conflict detection  — same (component, unit) but different value → flag
    5. Sorting             — table rows first, then by section, then component name
    """

    ALIASES = {
        "nm": "Nm", "n·m": "Nm", "n.m": "Nm", "newton metres": "Nm",
        "newton meters": "Nm", "n-m": "Nm",
        "lb-ft": "lb-ft", "lbft": "lb-ft", "ft-lb": "lb-ft",
        "foot pounds": "lb-ft", "foot-pounds": "lb-ft",
        "lb-in": "lb-in", "lbin": "lb-in", "in-lb": "lb-in",
        "inch pounds": "lb-in",
        "mm": "mm", "millimetres": "mm", "millimeters": "mm",
        "in": "in", "inches": "in", "inch": "in",
        "degrees": "degrees", "deg": "degrees", "°": "degrees",
    }

    def validate(self, results: List[SpecResult]) -> List[dict]:
        for r in results:
            r.unit  = self.ALIASES.get(r.unit.lower().strip(), r.unit)
            r.value = self._clean(r.value)

        seen: Dict[Tuple, SpecResult] = {}
        conflicts: set = set()

        for r in results:
            key = (
                r.component.lower().strip(),
                r.unit,
                int(r.stage_number or 0),
                (r.section_id or "").strip(),
            )
            if key in seen:
                if seen[key].value != r.value:
                    conflicts.add(key)
                elif r.confidence > seen[key].confidence:
                    seen[key] = r
            else:
                seen[key] = r

        out = []
        for key, r in seen.items():
            d = r.to_dict()
            d["is_conflict"] = key in conflicts
            out.append(d)

        out.sort(key=lambda x: (
            0 if x["source"] == "table" else 1,
            x["section_id"],
            x["component"],
        ))
        return out

    @staticmethod
    def _clean(v: str) -> str:
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            return v.strip()


# =============================================================================
# Main pipeline
# =============================================================================

class ExtractionPipeline:
    """
    Connects Phase 2 HybridRetriever → fast/LLM extraction → validation → output.
    """

    def __init__(
        self,
        index_dir:   str = "./phase2_index",
        model:       str = "llama3.1:8b",
        ollama_host: str = "http://localhost:11434",
        k:           int = 8,
    ):
        self.min_top_rrf   = 0.010
        self.min_llm_conf  = 0.30
        self.k         = k
        self.fast_ext  = FastPathExtractor()
        self.validator = SpecValidator()

        # Load Phase 2 retriever
        log.info("Loading Phase 2 index …")
        phase2_dir = Path(__file__).resolve().parents[1] / "phase2_chunking"
        if str(phase2_dir) not in sys.path:
            sys.path.insert(0, str(phase2_dir))
        from chunker_embedder import Phase2Pipeline
        self.retriever = Phase2Pipeline(index_dir).load_retriever()

        # Load LLM (optional — fast path works without it)
        self.llm_ext = None
        try:
            self.llm_ext = LLMExtractor(OllamaClient(model=model, host=ollama_host))
        except RuntimeError as e:
            log.warning("Ollama unavailable — LLM path disabled.\n%s", e)
            log.warning("Fast path handles structured table chunks fine.")

    def run(
        self,
        query:     str,
        variant:   Optional[str] = None,
        spec_only: bool          = True,
        force_llm: bool          = False,
    ) -> dict:
        t0 = time.time()

        # ── Retrieve ──────────────────────────────────────────────────────────
        chunks = self.retriever.retrieve(
            query, k=self.k, spec_only=spec_only, vehicle_variant=variant
        )
        if not chunks:
            log.warning("No chunks retrieved: %s", query)
            return self._empty(query, variant)

        top_score = float(chunks[0].get("_final_rrf", chunks[0].get("rrf_score", 0.0)) or 0.0)
        if top_score < self.min_top_rrf:
            log.info("Top chunk relevance too low (%.4f < %.4f) — returning empty",
                     top_score, self.min_top_rrf)
            return self._empty(query, variant)

        log.info("Retrieved %d | top=%s sec=%s",
                 len(chunks), chunks[0].get("content_type","?"), chunks[0].get("section_id","?"))

        # ── Extract ───────────────────────────────────────────────────────────
        raw: List[SpecResult] = []
        llm_queue: List[dict] = []
        fast_specs_count = 0
        llm_specs_count = 0

        for chunk in chunks:
            if not force_llm and self.fast_ext.is_eligible(chunk):
                specs = self.fast_ext.extract(chunk)
                raw.extend(specs)
                fast_specs_count += len(specs)
                log.debug("fast: %d specs ← %s", len(specs), chunk.get("component","?")[:40])
            else:
                llm_queue.append(chunk)

        if llm_queue:
            if self.llm_ext:
                log.info("LLM: %d chunks → %s", len(llm_queue), self.llm_ext.client.model)
                llm_specs, conf, reason = self.llm_ext.extract(
                    query, llm_queue,
                    sec_id=llm_queue[0].get("section_id", ""),
                    sec_nm=llm_queue[0].get("section_name", ""),
                    var=llm_queue[0].get("vehicle_variant", []),
                )
                kept = [s for s in llm_specs if float(s.confidence) >= self.min_llm_conf]
                dropped = len(llm_specs) - len(kept)
                raw.extend(kept)
                llm_specs_count += len(kept)
                if dropped:
                    log.info("LLM: dropped %d low-confidence specs (< %.2f)",
                             dropped, self.min_llm_conf)
                log.info("LLM: %d specs kept (raw=%d) conf=%.2f — %s",
                         len(kept), len(llm_specs), conf, reason)
            else:
                log.info("Ollama offline — running regex fallback on %d chunks", len(llm_queue))
                fallback = self._regex_fallback(query, llm_queue)
                raw.extend(fallback)
                llm_specs_count += len(fallback)

        # ── Validate ──────────────────────────────────────────────────────────
        validated = self.validator.validate(raw)

        return {
            "query":        query,
            "variant":      variant,
            "results":      [
                {"component": r["component"], "spec_type": r["spec_type"],
                 "value": r["value"], "unit": r["unit"]}
                for r in validated
            ],
            "full_results": validated,
            "metadata": {
                "chunks_retrieved":  len(chunks),
                "fast_path_count":   fast_specs_count,
                "llm_path_count":    llm_specs_count,
                "total_specs_found": len(validated),
                "elapsed_sec":       round(time.time() - t0, 2),
                "top_chunk_type":    chunks[0].get("content_type", ""),
                "top_section":       chunks[0].get("section_id", ""),
                "ollama_online":     self.llm_ext is not None,
            },
        }

    def run_batch(self, queries: List[str], variant: Optional[str] = None,
                  spec_only: bool = True) -> List[dict]:
        results = []
        for i, q in enumerate(queries, 1):
            log.info("[%d/%d] %s", i, len(queries), q)
            results.append(self.run(q, variant=variant, spec_only=spec_only))
        return results

    # ── Regex fallback when Ollama is offline ─────────────────────────────────

    # Broad torque pattern: number + unit, with optional range and imperial parens
    _REGEX_TORQUE = re.compile(
        r"(?P<nm>[\d,]+\.?\d*(?:\s*[-–]\s*[\d,]+\.?\d*)?)\s*"
        r"(?P<nm_unit>N[·.]?m|Nm)"
        r"(?:\s*\((?P<imp>[\d,]+\.?\d*)\s*(?P<imp_unit>kgf[·.]?cm|lb[·.]?ft|ft[·.]?lbf?|lb[·.]?in)\))?",
        re.IGNORECASE,
    )
    _REGEX_LBFT = re.compile(
        r"(?P<val>[\d,]+\.?\d*(?:\s*[-–]\s*[\d,]+\.?\d*)?)\s*"
        r"(?P<unit>lb[·.\-]?ft|ft[·.\-]?lb[sf]?)",
        re.IGNORECASE,
    )

    def _regex_fallback(self, query: str, chunks: List[dict]) -> List[SpecResult]:
        """
        When Ollama is offline, scan retrieved chunk text with regex to pull
        torque values directly.  Confidence is lower than LLM path (0.65).
        """
        found: List[SpecResult] = []
        seen_keys: set = set()

        for chunk in chunks:
            text = chunk.get("display_text", chunk.get("embed_text", ""))
            sec_id   = chunk.get("section_id", "")
            sec_name = chunk.get("section_name", "")

            # Scan each line for torque values
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Try Nm matches
                for m in self._REGEX_TORQUE.finditer(line):
                    nm_str = m.group("nm").replace(",", "")
                    # Take lower bound of range
                    nm_val_str = nm_str.split("-")[0].split("–")[0].strip()
                    try:
                        nm_val = float(nm_val_str)
                    except ValueError:
                        continue
                    if nm_val <= 0:
                        continue

                    # Guess component from the query itself as best-effort
                    component = self._component_from_query(query)
                    unit      = "Nm"
                    key       = (component.lower(), unit, round(nm_val))
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    found.append(SpecResult(
                        component=component, spec_type="Torque",
                        value=nm_val_str, unit=unit,
                        section_id=sec_id, section_name=sec_name,
                        source="regex_fallback", confidence=0.65,
                        raw_text=line,
                    ))

                    # Also add imperial equivalent if present
                    if m.group("imp"):
                        imp_val = m.group("imp").replace(",", "").split("-")[0].split("–")[0].strip()
                        imp_unit_raw = (m.group("imp_unit") or "").lower()
                        if "kgf" in imp_unit_raw:
                            imp_unit = "kgf·cm"
                        elif "ft" in imp_unit_raw:
                            imp_unit = "lb-ft"
                        else:
                            imp_unit = "lb-in"
                        imp_key = (component.lower(), imp_unit, imp_val)
                        if imp_key not in seen_keys:
                            seen_keys.add(imp_key)
                            found.append(SpecResult(
                                component=component, spec_type="Torque",
                                value=imp_val, unit=imp_unit,
                                section_id=sec_id, section_name=sec_name,
                                source="regex_fallback", confidence=0.65,
                                raw_text=line,
                            ))

                # Try standalone lb-ft (no Nm on same line)
                if not self._REGEX_TORQUE.search(line):
                    for m in self._REGEX_LBFT.finditer(line):
                        val_str = m.group("val").replace(",", "").split("-")[0].split("–")[0].strip()
                        try:
                            val = float(val_str)
                        except ValueError:
                            continue
                        if val <= 0:
                            continue
                        component = self._component_from_query(query)
                        key = (component.lower(), "lb-ft", round(val))
                        if key not in seen_keys:
                            seen_keys.add(key)
                            found.append(SpecResult(
                                component=component, spec_type="Torque",
                                value=val_str, unit="lb-ft",
                                section_id=sec_id, section_name=sec_name,
                                source="regex_fallback", confidence=0.60,
                                raw_text=line,
                            ))

        log.info("Regex fallback: %d specs extracted from %d chunks", len(found), len(chunks))
        return found

    @staticmethod
    def _component_from_query(query: str) -> str:
        """
        Best-effort component name from the query string.
        Strips generic verbs so we get 'brake caliper bolts' not 'torque for brake caliper bolts'.
        """
        q = re.sub(
            r"^(what is|what('s| is) the|torque for|tighten|spec|specification|find|give me|tell me)\s+",
            "", query.strip(), flags=re.IGNORECASE,
        )
        q = re.sub(r"\s+(torque|spec|specification|value)$", "", q, flags=re.IGNORECASE)
        return q.strip()[:80] or query[:80]

    @staticmethod
    def _empty(query: str, variant: Optional[str]) -> dict:
        return {"query": query, "variant": variant, "results": [],
                "full_results": [], "metadata": {"chunks_retrieved": 0, "total_specs_found": 0}}


# =============================================================================
# Output helpers
# =============================================================================

def print_results(result: dict):
    print()
    print("=" * 64)
    print(f"  Query   : {result['query']}")
    if result.get("variant"):
        print(f"  Variant : {result['variant']}")
    m = result.get("metadata", {})
    print(f"  Specs   : {m.get('total_specs_found', 0)} found | "
          f"{m.get('chunks_retrieved', 0)} chunks | {m.get('elapsed_sec', 0):.1f}s")
    print("=" * 64)

    for r in result.get("full_results", []):
        conflict = "  [CONFLICT]" if r.get("is_conflict") else ""
        var      = f" [{', '.join(r['vehicle_variant'])}]" if r.get("vehicle_variant") else ""
        stage    = f" stage {r['stage_number']}" if r.get("stage_number") else ""
        print(f"  {r['component']}{stage}{var}{conflict}")
        print(f"    → {r['spec_type']}: {r['value']} {r['unit']}"
              f"  (sec={r['section_id']} | src={r['source']} | conf={r['confidence']:.2f})")

    if not result.get("full_results"):
        print("  No specifications found.")
    print()


def to_csv(results: List[dict], full: bool = False) -> str:
    buf    = StringIO()
    fields = (
        ["query","component","spec_type","value","unit",
         "section_id","vehicle_variant","source","confidence","is_conflict"]
        if full else ["component","spec_type","value","unit"]
    )
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for res in results:
        for spec in res.get("full_results" if full else "results", []):
            row = dict(spec)
            if full:
                row["query"]           = res["query"]
                row["vehicle_variant"] = ", ".join(row.get("vehicle_variant", []))
            w.writerow(row)
    return buf.getvalue()


# =============================================================================
# CLI
# =============================================================================

DEMO_QUERIES = [
    ("Torque for brake caliper bolts",             None),
    ("Shock absorber lower nuts specification",    "4WD"),
    ("Wheel bearing and hub bolt torque",          None),
    ("Stage tightening sequence U-bolt rear",      None),
    ("Upper ball joint nut torque",                "RWD"),
    ("Halfshaft assembled length dimension",       None),
    ("Stabilizer bar link nuts SVT Raptor torque", None),
    ("Tie-rod end nut torque specification",       None),
]


def main():
    p = argparse.ArgumentParser(
        description="Phase 3: LLM Spec Extraction — 2014 F-150 Workshop Manual"
    )
    p.add_argument("--index",     default="./phase2_index")
    p.add_argument("--model",     default="llama3.1:8b")
    p.add_argument("--host",      default="http://localhost:11434")
    p.add_argument("--variant",   default=None, help="RWD | 4WD")
    p.add_argument("--k",         type=int, default=5)
    p.add_argument("--full",      action="store_true", help="Output all fields")
    p.add_argument("--csv",       action="store_true", help="CSV output")
    p.add_argument("--out",       default=None, help="Save to file")
    p.add_argument("--force-llm", action="store_true", help="Always use LLM")

    sub = p.add_subparsers(dest="cmd")

    qp = sub.add_parser("query", help="Single query")
    qp.add_argument("text", nargs="+")

    bp = sub.add_parser("batch", help="Queries from file (one per line)")
    bp.add_argument("file")

    sub.add_parser("interactive", help="Interactive REPL")
    sub.add_parser("demo",        help="Run 8 built-in demo queries")

    args = p.parse_args()

    pipe = ExtractionPipeline(
        index_dir=args.index, model=args.model,
        ollama_host=args.host, k=args.k,
    )

    # ── single query ──────────────────────────────────────────────────────────
    if args.cmd == "query":
        query  = " ".join(args.text)
        result = pipe.run(query, variant=args.variant, force_llm=args.force_llm)
        print_results(result)
        output = to_csv([result], args.full) if args.csv else json.dumps(
            result["full_results"] if args.full else result["results"],
            indent=2, ensure_ascii=False)
        if args.out:
            Path(args.out).write_text(output)
            log.info("Saved → %s", args.out)
        else:
            print(output)

    # ── batch ─────────────────────────────────────────────────────────────────
    elif args.cmd == "batch":
        qs = [l.strip() for l in Path(args.file).read_text().splitlines()
              if l.strip() and not l.startswith("#")]
        log.info("Running %d queries …", len(qs))
        results = pipe.run_batch(qs, variant=args.variant)
        for r in results:
            print_results(r)
        all_specs = [s for r in results for s in r["results"]]
        output = to_csv(results, args.full) if args.csv else json.dumps(all_specs, indent=2)
        if args.out:
            Path(args.out).write_text(output)
            log.info("Saved %d specs → %s", len(all_specs), args.out)
        else:
            print(output)

    # ── interactive ───────────────────────────────────────────────────────────
    elif args.cmd == "interactive":
        print("\n2014 F-150 Specification Extractor (Llama 3.1 8B)")
        print("Commands: :variant RWD|4WD|clear   :full   :quit")
        print("=" * 50)
        variant, show_full = args.variant, False
        while True:
            try:
                raw = input(f"\n[{variant or 'all variants'}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if not raw:             continue
            if raw == ":quit":      break
            if raw == ":full":      show_full = not show_full; print(f"Full output: {show_full}"); continue
            if raw.startswith(":variant "):
                v = raw.split(None, 1)[1].strip()
                variant = None if v == "clear" else v
                print(f"Variant: {variant or 'all'}")
                continue
            result = pipe.run(raw, variant=variant)
            print_results(result)
            if show_full:
                print(json.dumps(result["full_results"], indent=2))
            else:
                print(json.dumps(result["results"], indent=2))

    # ── demo ─────────────────────────────────────────────────────────────────
    elif args.cmd == "demo":
        all_results, total_specs = [], 0
        for query, variant in DEMO_QUERIES:
            r = pipe.run(query, variant=variant)
            print_results(r)
            all_results.append(r)
            total_specs += len(r["results"])

        fast = sum(r["metadata"].get("fast_path_count", 0) for r in all_results)
        llm  = sum(r["metadata"].get("llm_path_count", 0) for r in all_results)
        print(f"Demo summary: {total_specs} specs | {fast} fast-path | {llm} LLM-path")
        print(f"Queries: {len(DEMO_QUERIES)} | Avg specs/query: {total_specs/len(DEMO_QUERIES):.1f}")

        if args.out:
            all_specs = [s for r in all_results for s in r["results"]]
            output    = to_csv(all_results, args.full) if args.csv else json.dumps(all_specs, indent=2)
            Path(args.out).write_text(output)
            log.info("Saved %d specs → %s", len(all_specs), args.out)

    else:
        p.print_help()


if __name__ == "__main__":
    main()
