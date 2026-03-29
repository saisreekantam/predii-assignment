"""
Phase 1: PDF Text Extraction & Structure Parsing
=================================================
Vehicle Specification Extraction — 2014 F-150 Workshop Manual

Pipeline:
    PDF / Text file
        ↓  PDFLoader          — opens real PDFs (PyMuPDF) or plain-text fallback
        ↓  TextCleaner        — strips noise (file paths, page stamps, URLs)
        ↓  StructureParser    — detects SECTION headers, subsection types, page breaks
        ↓  TorqueTableParser  — extracts tabular specs (Description | Nm | lb-ft | lb-in)
        ↓  InlineSpecParser   — extracts specs embedded in procedure steps
        ↓  SegmentBuilder     — assembles rich TextSegment objects with full metadata
        ↓  ExtractionPipeline — orchestrates all stages; saves JSON + cleaned text

Output files (in ./output/):
    *_segments.json   — every segment with metadata, ready for Phase 2 chunking
    *_specs.json      — only spec segments: table rows + inline torque values
    *_cleaned.txt     — cleaned full text for inspection

Usage:
    python pdf_extractor.py sample-service-manual.pdf
    python pdf_extractor.py sample-service-manual.pdf --out-dir ./results
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase1")


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

class ContentType(str, Enum):
    TORQUE_TABLE_ROW  = "torque_table_row"   # single row of spec table: component | Nm | lb-ft | lb-in
    INLINE_TORQUE     = "inline_torque"       # "Tighten to 90 Nm (66 lb-ft)" inside a procedure step
    MULTI_STAGE_TORQUE= "multi_stage_torque"  # Stage 1/2/3/4 tightening sequences
    DIMENSIONAL_SPEC  = "dimensional_spec"    # mm, inches, degrees measurements
    PROCEDURE_STEP    = "procedure_step"      # numbered or bulleted procedure steps
    TABLE_HEADER      = "table_header"        # "Description Nm lb-ft lb-in" header row
    SECTION_HEADER    = "section_header"      # SECTION 204-00: ...
    SUBSECTION_HEADER = "subsection_header"   # DIAGNOSIS AND TESTING, REMOVAL AND INSTALLATION …
    WARNING_NOTICE    = "warning_notice"      # WARNING / NOTICE / NOTE blocks
    DIAGNOSTIC        = "diagnostic"          # Symptom → Possible Sources → Action
    PART_NUMBER       = "part_number"         # tool / material part numbers
    GENERAL_TEXT      = "general_text"        # everything else


@dataclass
class SpecValue:
    """A parsed numeric specification value."""
    value_nm:    Optional[float] = None   # Newton-metres
    value_lbft:  Optional[float] = None   # pound-feet
    value_lbin:  Optional[float] = None   # pound-inches
    value_mm:    Optional[float] = None   # millimetres
    value_deg:   Optional[float] = None   # degrees
    raw:         str = ""                 # original string as found in the text


@dataclass
class TextSegment:
    """
    One extracted segment of the service manual.

    Every segment carries enough metadata that it can be embedded and retrieved
    independently in Phase 2 without needing to look up surrounding context.
    """
    # ── content ──────────────────────────────────────────────────────────────
    text:              str
    content_type:      ContentType

    # ── structural location ──────────────────────────────────────────────────
    section_id:        str   = ""         # e.g. "204-01A"
    section_name:      str   = ""         # e.g. "Front Suspension — RWD"
    subsection_type:   str   = ""         # e.g. "SPECIFICATIONS", "REMOVAL AND INSTALLATION"
    page_number:       int   = 0
    line_number:       int   = 0

    # ── extracted specification fields ───────────────────────────────────────
    component:         str   = ""         # e.g. "Shock absorber lower nuts"
    spec_value:        Optional[SpecValue] = None
    vehicle_variant:   List[str] = field(default_factory=list)  # ["RWD"], ["4WD"], ["SVT Raptor"]
    stage_number:      int   = 0          # for multi-stage torque: 1, 2, 3, 4

    # ── quality flags ────────────────────────────────────────────────────────
    is_safety_critical: bool = False       # contains WARNING or NOTICE
    has_condition:       bool = False      # spec depends on a condition
    confidence:          float = 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["content_type"] = self.content_type.value
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — PDF / text loading
# ─────────────────────────────────────────────────────────────────────────────

class PDFLoader:
    """
    Opens the input file and returns (pages_text, metadata).

    Strategy:
        1. Try PyMuPDF (fitz) — best general text extraction, preserves layout.
        2. Try pdfplumber  — specifically for pages that have structured tables.
        3. Fallback: treat as a pre-extracted UTF-8 text file (handles the
           'compressed' text file supplied with this assignment).

    For table pages pdfplumber is called as a second pass to extract structured
    row data. PyMuPDF remains primary for all other text.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    # ── public ───────────────────────────────────────────────────────────────

    def load(self) -> Tuple[str, dict]:
        """Return (full_text, metadata_dict)."""
        # Detect file type from magic bytes
        with open(self.path, "rb") as fh:
            header = fh.read(8)

        is_pdf = header.startswith(b"%PDF")

        if is_pdf:
            log.info("Detected real PDF — using PyMuPDF + pdfplumber")
            return self._load_real_pdf()
        else:
            log.info("File is pre-extracted text — reading as UTF-8")
            return self._load_text_file()

    # ── private ──────────────────────────────────────────────────────────────

    def _load_real_pdf(self) -> Tuple[str, dict]:
        """
        Two-pass extraction:
            Pass 1 (PyMuPDF)  — all text, page by page with page markers injected.
            Pass 2 (pdfplumber) — table pages get structured table text appended.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            log.warning("PyMuPDF not installed — trying pypdf → pdfplumber")
            return self._load_with_pypdf_fallback()

        doc = fitz.open(str(self.path))
        metadata = {
            "total_pages": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "source_file": self.path.name,
            "extraction_method": "pymupdf+pdfplumber",
        }

        pages_text: List[str] = []
        pages_with_tables: set[int] = set()

        # ── Pass 1: PyMuPDF text ─────────────────────────────────────────────
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")  # "text" mode: preserves reading order
            if text.strip():
                # Inject a consistent page marker so StructureParser can track pages
                pages_text.append(f"<<PAGE {page_num}>>\n{text}")

            # Heuristic: pages with many numbers and short lines likely have tables
            lines = text.split("\n")
            numeric_lines = sum(1 for ln in lines if re.search(r"\d+\s+\d+", ln))
            if numeric_lines > 3:
                pages_with_tables.add(page_num)

        full_text = "\n".join(pages_text)
        log.info(
            "PyMuPDF: %d pages extracted, %d suspected table pages",
            len(doc), len(pages_with_tables)
        )
        doc.close()

        # ── Pass 2: pdfplumber table extraction ──────────────────────────────
        if pages_with_tables:
            table_supplement = self._extract_tables_pdfplumber(pages_with_tables)
            if table_supplement:
                full_text += "\n\n<<STRUCTURED_TABLES>>\n" + table_supplement
                log.info("pdfplumber added %d chars of structured table text",
                         len(table_supplement))

        metadata["total_chars"] = len(full_text)
        return full_text, metadata

    def _extract_tables_pdfplumber(self, target_pages: set[int]) -> str:
        """
        Use pdfplumber to extract tables from target pages.
        Returns formatted text with consistent column alignment.
        """
        try:
            import pdfplumber
        except ImportError:
            log.warning("pdfplumber not installed — skipping table extraction pass")
            return ""

        table_texts: List[str] = []

        with pdfplumber.open(str(self.path)) as pdf:
            for page_num in sorted(target_pages):
                if page_num > len(pdf.pages):
                    continue
                page = pdf.pages[page_num - 1]  # pdfplumber is 0-indexed

                # pdfplumber table settings tuned for Ford-style spec tables
                table_settings = {
                    "vertical_strategy":   "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance":      3,
                    "join_tolerance":      3,
                    "edge_min_length":     30,
                }

                tables = page.extract_tables(table_settings)
                for table in tables:
                    if not table:
                        continue
                    # Format each row as tab-separated
                    rows = []
                    for row in table:
                        if row and any(cell and cell.strip() for cell in row):
                            cells = [str(c).strip() if c else "—" for c in row]
                            rows.append("\t".join(cells))
                    if rows:
                        table_texts.append(
                            f"<<TABLE page={page_num}>>\n" + "\n".join(rows)
                        )

        return "\n\n".join(table_texts)

    def _load_with_pypdf_fallback(self) -> Tuple[str, dict]:
        """
        Fallback chain when PyMuPDF is unavailable:
            1. pypdf   — modern, well-maintained, handles most real PDFs
            2. PyPDF2  — older but sometimes handles edge-case PDFs differently
            3. pdfplumber — last resort for real PDFs
        Each attempt is wrapped so a broken library doesn't kill the pipeline.
        """
        # ── 1. pypdf ─────────────────────────────────────────────────────────
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(self.path))
            parts = []
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(f"<<PAGE {i}>>\n{text}")
            full_text = "\n".join(parts)
            if full_text.strip():
                log.info("pypdf extracted %d chars across %d pages",
                         len(full_text), len(reader.pages))
                return full_text, {
                    "total_chars": len(full_text),
                    "total_pages": len(reader.pages),
                    "source_file": self.path.name,
                    "extraction_method": "pypdf",
                }
        except Exception as e:
            log.warning("pypdf failed (%s) — trying PyPDF2", e)

        # ── 2. PyPDF2 ────────────────────────────────────────────────────────
        try:
            import PyPDF2
            parts = []
            with open(self.path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                for i, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        parts.append(f"<<PAGE {i}>>\n{text}")
            full_text = "\n".join(parts)
            if full_text.strip():
                log.info("PyPDF2 extracted %d chars", len(full_text))
                return full_text, {
                    "total_chars": len(full_text),
                    "source_file": self.path.name,
                    "extraction_method": "PyPDF2",
                }
        except Exception as e:
            log.warning("PyPDF2 failed (%s) — trying pdfplumber", e)

        # ── 3. pdfplumber ────────────────────────────────────────────────────
        return self._load_with_pdfplumber_only()

    def _load_with_pdfplumber_only(self) -> Tuple[str, dict]:
        """Last-resort fallback: pdfplumber for everything."""
        import pdfplumber
        parts: List[str] = []
        with pdfplumber.open(str(self.path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                parts.append(f"<<PAGE {i}>>\n{text}")
        full_text = "\n".join(parts)
        return full_text, {
            "total_chars": len(full_text),
            "source_file": self.path.name,
            "extraction_method": "pdfplumber_only",
        }

    def _load_text_file(self) -> Tuple[str, dict]:
        """Load pre-extracted text (the supplied 'compressed' file in this assignment)."""
        text = self.path.read_text(encoding="utf-8", errors="replace")
        metadata = {
            "total_chars": len(text),
            "total_lines": text.count("\n"),
            "source_file": self.path.name,
            "extraction_method": "text_file_passthrough",
        }
        log.info("Loaded text file: %d chars, %d lines",
                 metadata["total_chars"], metadata["total_lines"])
        return text, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

class TextCleaner:
    """
    Removes noise introduced by the HTML→PDF→text pipeline:
      • Windows file paths (file:///C:/TSO/...)
      • French page stamps (Page X sur Y)
      • Redundant section header repetitions
      • Unicode control characters
      • Excessive blank lines

    Preserves all specification content and document structure markers.
    """

    # ── compiled patterns (compiled once at class load time) ──────────────────
    _FILE_URL     = re.compile(r"file:///[^\n]+")
    _FRENCH_PAGE  = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")   # dates like 2014-03-01
    _UNICODE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
    _MULTI_BLANK  = re.compile(r"\n{4,}")
    _MULTI_SPACE  = re.compile(r"[ \t]{3,}")
    # The page marker "2014 F-150 Workshop Manual Page X sur Y" — keep the
    # section info but strip the page stamp itself
    _PAGE_STAMP   = re.compile(
        r"2014 F-150 Workshop Manual\s+Page\s+\d+\s+sur\s+\d+\s*"
    )
    # "Procedure revision date: …" lines are noise for retrieval
    _PROC_DATE    = re.compile(r"Procedure revision date:\s*[\d/]+\s*")

    def clean(self, text: str) -> str:
        text = self._UNICODE_CTRL.sub("", text)
        text = self._FILE_URL.sub("", text)
        text = self._PAGE_STAMP.sub("", text)
        text = self._PROC_DATE.sub("", text)
        text = self._FRENCH_DATE(text)  # replaces trailing dates on section lines
        text = self._MULTI_SPACE.sub(" ", text)
        text = self._MULTI_BLANK.sub("\n\n\n", text)
        # Normalize curly quotes → straight (safe for number parsing downstream)
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        # Normalize em-dash used as "N/A" in tables
        text = text.replace("\u2014", "—").replace("\u2013", "—")
        return text.strip()

    @staticmethod
    def _FRENCH_DATE(text: str) -> str:
        """Remove trailing date stamps on section header lines."""
        return re.sub(
            r"(SECTION\s+[\d\-]+:[^\n]+?)\s+\d{4}-\d{2}-\d{2}\s*",
            r"\1\n",
            text
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Structure parsing
# ─────────────────────────────────────────────────────────────────────────────

class StructureParser:
    """
    Scans the cleaned text line by line to build a DocumentMap:
    a list of (line_index, line_text, detected_structure) tuples.

    Detected structures feed into the spec parsers so that every extracted
    value carries correct section/subsection/page provenance.
    """

    # ── Ford workshop manual section patterns ─────────────────────────────────
    _RE_SECTION = re.compile(
        r"^SECTION\s+([\d\-]+[A-Z]?):\s*(.+?)(?:\s+2014 F-150 Workshop Manual)?$",
        re.IGNORECASE
    )
    _RE_SUBSECTION = re.compile(
        r"^(DIAGNOSIS AND TESTING|DESCRIPTION AND OPERATION|GENERAL PROCEDURES|"
        r"REMOVAL AND INSTALLATION|DISASSEMBLY AND ASSEMBLY|SPECIFICATIONS|"
        r"INSPECTION AND VERIFICATION|ADJUSTMENTS|COMPONENT TESTS|"
        r"PINPOINT TEST)(?:\s+Procedure revision date:\s*[\d/]+)?\s*$",
        re.IGNORECASE
    )
    _RE_PAGE = re.compile(r"^<<PAGE\s+(\d+)>>$")

    # Content-type quick-classification patterns (order matters)
    _RE_TABLE_HEADER = re.compile(
        r"^(?:Description\s+)?Nm\s+lb[-‑–]ft(?:\s+lb[-‑–]in)?\s*$|"
        r"^(?:Description\s+)?Nm\s+lb[-‑–]in\s*$",
        re.IGNORECASE
    )
    _RE_WARNING = re.compile(r"^(WARNING|NOTICE|NOTE)\b", re.IGNORECASE)
    _RE_STEP_NUM = re.compile(r"^\d{1,2}\.\s+")
    _RE_BULLET   = re.compile(r"^z\s+|^•\s+|^-\s+")

    @dataclass
    class LineInfo:
        idx:             int
        text:            str
        content_type:    ContentType
        section_id:      str
        section_name:    str
        subsection_type: str
        page_number:     int

    def parse(self, text: str) -> List["StructureParser.LineInfo"]:
        """Return a LineInfo for every non-blank line."""
        lines = text.split("\n")
        result: List[StructureParser.LineInfo] = []

        cur_section_id   = ""
        cur_section_name = ""
        cur_subsection   = ""
        cur_page         = 0

        for idx, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue

            # ── page tracking ────────────────────────────────────────────────
            m = self._RE_PAGE.match(line)
            if m:
                cur_page = int(m.group(1))
                continue

            # ── section header ───────────────────────────────────────────────
            m = self._RE_SECTION.match(line)
            if m:
                cur_section_id   = m.group(1).strip()
                cur_section_name = m.group(2).strip()
                cur_subsection   = ""
                result.append(self.LineInfo(
                    idx=idx, text=line,
                    content_type=ContentType.SECTION_HEADER,
                    section_id=cur_section_id, section_name=cur_section_name,
                    subsection_type="", page_number=cur_page,
                ))
                continue

            # ── subsection header ────────────────────────────────────────────
            m = self._RE_SUBSECTION.match(line)
            if m:
                cur_subsection = m.group(1).upper()
                result.append(self.LineInfo(
                    idx=idx, text=line,
                    content_type=ContentType.SUBSECTION_HEADER,
                    section_id=cur_section_id, section_name=cur_section_name,
                    subsection_type=cur_subsection, page_number=cur_page,
                ))
                continue

            # ── classify line content ────────────────────────────────────────
            ctype = self._classify(line)

            result.append(self.LineInfo(
                idx=idx, text=line,
                content_type=ctype,
                section_id=cur_section_id, section_name=cur_section_name,
                subsection_type=cur_subsection, page_number=cur_page,
            ))

        log.info("StructureParser: %d non-blank lines classified", len(result))
        return result

    def _classify(self, line: str) -> ContentType:
        if self._RE_TABLE_HEADER.match(line):
            return ContentType.TABLE_HEADER
        if self._RE_WARNING.match(line):
            return ContentType.WARNING_NOTICE
        if self._RE_STEP_NUM.match(line):
            return ContentType.PROCEDURE_STEP
        if self._RE_BULLET.match(line):
            return ContentType.PROCEDURE_STEP
        return ContentType.GENERAL_TEXT


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4a — Torque table parser
# ─────────────────────────────────────────────────────────────────────────────

class TorqueTableParser:
    """
    Extracts rows from Ford-format torque specification tables:

        Description                   Nm    lb-ft  lb-in
        Brake disc shield bolts       17    —      150
        Shock absorber lower nuts     90    66     —
        Lower arm forward/rearward    350   258    —

    Each row becomes one TORQUE_TABLE_ROW segment.

    The parser is table-aware: it knows a table starts after a TABLE_HEADER
    line and ends when a non-data line is detected.
    """

    # Matches: "Some component text 17 — 150"
    # Also supports range values like: "1.5-3.0"
    # The manual uses SINGLE spaces between columns.
    # Lazy (.+?) backtracks until the last 3 tokens are valid spec values.
    # Groups: (description, nm_val, lbft_val, lbin_val)
    _ROW_RE = re.compile(
        r"^(.+?)\s+"                  # description (lazy — backtracks to find best split)
        r"([\d\.]+(?:\s*[-—–]\s*[\d\.]+)?|[—–-])\s+"  # Nm value/range or dash
        r"([\d\.]+(?:\s*[-—–]\s*[\d\.]+)?|[—–-])\s+"  # lb-ft value/range or dash
        r"([\d\.]+(?:\s*[-—–]\s*[\d\.]+)?|[—–-])\s*$" # lb-in value/range or dash
    )

    # Lines that definitely are NOT table data rows
    _NOT_ROW_RE = re.compile(
        r"^(Description|SECTION|SUBSECTION|Torque Spec|NOTE|WARNING|NOTICE|"
        r"file://|2014 F-150|\d{4}-\d{2}-\d{2}|Page\s+\d)",
        re.IGNORECASE
    )

    def extract(
        self,
        line_infos: List[StructureParser.LineInfo],
    ) -> Iterator[TextSegment]:
        """Yield one TextSegment per torque table row found."""

        in_table   = False
        table_section_id   = ""
        table_section_name = ""
        table_subsection   = ""
        table_page         = 0

        i = 0
        while i < len(line_infos):
            li = line_infos[i]

            if li.content_type == ContentType.TABLE_HEADER:
                in_table           = True
                table_section_id   = li.section_id
                table_section_name = li.section_name
                table_subsection   = li.subsection_type
                table_page         = li.page_number
                i += 1
                continue

            if in_table:
                # Only a new section/subsection boundary ends the table.
                if li.content_type in (
                    ContentType.SECTION_HEADER, ContentType.SUBSECTION_HEADER
                ):
                    in_table = False
                    i += 1
                    continue

                # 1) Normal single-line row: "Component 90 66 —"
                m = self._ROW_RE.match(li.text)
                if m and not self._NOT_ROW_RE.match(li.text):
                    desc  = m.group(1).strip()
                    nm    = self._parse_num(m.group(2))
                    lbft  = self._parse_num(m.group(3))
                    lbin  = self._parse_num(m.group(4))

                    # Sanity: at least one value must be a real number
                    if nm is not None or lbft is not None or lbin is not None:
                        spec_val = SpecValue(
                            value_nm=nm, value_lbft=lbft, value_lbin=lbin,
                            raw=li.text.strip()
                        )
                        readable = self._format_readable(desc, nm, lbft, lbin, table_section_name)

                        yield TextSegment(
                            text=readable,
                            content_type=ContentType.TORQUE_TABLE_ROW,
                            section_id=table_section_id,
                            section_name=table_section_name,
                            subsection_type=table_subsection,
                            page_number=table_page,
                            line_number=li.idx,
                            component=desc,
                            spec_value=spec_val,
                            vehicle_variant=self._extract_variant(table_section_name),
                            is_safety_critical=True,
                            confidence=0.98,
                        )
                    i += 1
                    continue

                # 2) Stacked row (common in pre-extracted text):
                #    Description
                #    Nm
                #    lb-ft
                #    lb-in
                if (
                    i + 3 < len(line_infos)
                    and not self._NOT_ROW_RE.match(li.text)
                    and not self._is_value_token(li.text)
                    and self._is_value_token(line_infos[i + 1].text)
                    and self._is_value_token(line_infos[i + 2].text)
                    and self._is_value_token(line_infos[i + 3].text)
                ):
                    desc = li.text.strip()
                    nm   = self._parse_num(line_infos[i + 1].text)
                    lbft = self._parse_num(line_infos[i + 2].text)
                    lbin = self._parse_num(line_infos[i + 3].text)

                    if nm is not None or lbft is not None or lbin is not None:
                        raw = " ".join([
                            desc,
                            line_infos[i + 1].text.strip(),
                            line_infos[i + 2].text.strip(),
                            line_infos[i + 3].text.strip(),
                        ])
                        spec_val = SpecValue(
                            value_nm=nm, value_lbft=lbft, value_lbin=lbin,
                            raw=raw
                        )
                        readable = self._format_readable(desc, nm, lbft, lbin, table_section_name)

                        yield TextSegment(
                            text=readable,
                            content_type=ContentType.TORQUE_TABLE_ROW,
                            section_id=table_section_id,
                            section_name=table_section_name,
                            subsection_type=table_subsection,
                            page_number=table_page,
                            line_number=li.idx,
                            component=desc,
                            spec_value=spec_val,
                            vehicle_variant=self._extract_variant(table_section_name),
                            is_safety_critical=True,
                            confidence=0.96,
                        )
                        i += 4
                        continue

            # Non-matching lines (NOTE, WARNING, blank-equivalents) → continue
            i += 1

    @staticmethod
    def _is_value_token(s: str) -> bool:
        s = s.strip()
        return bool(re.match(r"^(?:[\d\.]+(?:\s*[-—–]\s*[\d\.]+)?|[—–-]|N/A)$", s, re.IGNORECASE))

    @staticmethod
    def _parse_num(s: str) -> Optional[float]:
        s = s.strip()
        if s in ("—", "-", "", "N/A"):
            return None

        # Support range tokens like "1.5-3.0" / "1.5 — 3.0".
        # We keep a single numeric value by using the lower bound.
        m = re.match(r"^([\d\.]+)\s*[-—–]\s*([\d\.]+)$", s)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None

        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def _format_readable(
        desc: str, nm: Optional[float], lbft: Optional[float], lbin: Optional[float],
        section: str
    ) -> str:
        parts = [f"Component: {desc}"]
        if nm   is not None: parts.append(f"Torque: {nm} Nm")
        if lbft is not None: parts.append(f"({lbft} lb-ft)")
        if lbin is not None: parts.append(f"({lbin} lb-in)")
        if section:          parts.append(f"[{section}]")
        return " — ".join(parts)

    @staticmethod
    def _extract_variant(section_name: str) -> List[str]:
        variants: List[str] = []
        if "RWD" in section_name or "Rear Wheel" in section_name:
            variants.append("RWD")
        if "4WD" in section_name or "Four Wheel" in section_name:
            variants.append("4WD")
        return variants


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4b — Inline spec parser
# ─────────────────────────────────────────────────────────────────────────────

class InlineSpecParser:
    """
    Extracts torque / dimensional specs embedded inside procedure text:

    Patterns handled:
        "Tighten to 90 Nm (66 lb-ft)."
        "tighten the nut to 350 Nm (258 lb-ft)"
        "(SVT Raptor) To install, tighten to 80 Nm (59 lb-ft)"
        "Stage 1: Tighten in a cross pattern to 35 Nm (26 lb-ft)."
        "CV shaft depth minimum: 15.5 mm (0.61 in)"
        "Tighten the valve stem-to-TPMS sensor screw to 1.5 Nm (13 lb-in)."
    """

    # ── torque: "NNN Nm (NNN lb-ft|lb-in)" ───────────────────────────────────
    _TORQUE_RE = re.compile(
        r"(?P<pre>.{0,120}?)"
        r"(?P<nm>[\d\.]+)\s*Nm"
        r"\s*\((?P<imp>[\d\.]+)\s*(?P<unit>lb-ft|lb-in)\)"
        r"(?P<post>.{0,60})",
        re.IGNORECASE
    )

    # ── multi-stage: "Stage N: Tighten … to NNN Nm (NNN lb-ft)" ─────────────
    _STAGE_RE = re.compile(
        r"Stage\s+(?P<stage>\d)\s*:.*?(?P<nm>[\d\.]+)\s*Nm"
        r"\s*\((?P<imp>[\d\.]+)\s*(?P<unit>lb-ft|lb-in)\)",
        re.IGNORECASE
    )

    # ── dimensional: "NNN mm (NNN in)" ───────────────────────────────────────
    _DIM_RE = re.compile(
        r"(?P<val>[\d\.]+)\s*mm\s*\((?P<imp>[\d\.]+)\s*in\)",
        re.IGNORECASE
    )

    # ── variant tags ─────────────────────────────────────────────────────────
    _VARIANT_RE = re.compile(
        r"\((?P<var>SVT Raptor|RWD|4WD|All models?)\)", re.IGNORECASE
    )

    def extract(
        self,
        line_infos: List[StructureParser.LineInfo],
    ) -> Iterator[TextSegment]:
        """Yield one TextSegment per inline spec found."""

        for li in line_infos:
            line = li.text

            # ── multi-stage torque ───────────────────────────────────────────
            m = self._STAGE_RE.search(line)
            if m:
                stage  = int(m.group("stage"))
                nm_val = float(m.group("nm"))
                imp    = float(m.group("imp"))
                unit   = m.group("unit")

                lbft = imp if unit == "lb-ft" else None
                lbin = imp if unit == "lb-in" else None

                yield TextSegment(
                    text=line.strip(),
                    content_type=ContentType.MULTI_STAGE_TORQUE,
                    section_id=li.section_id,
                    section_name=li.section_name,
                    subsection_type=li.subsection_type,
                    page_number=li.page_number,
                    line_number=li.idx,
                    component=self._guess_component(line),
                    spec_value=SpecValue(value_nm=nm_val, value_lbft=lbft,
                                        value_lbin=lbin, raw=line.strip()),
                    vehicle_variant=self._extract_variant(line),
                    stage_number=stage,
                    is_safety_critical=True,
                    confidence=0.96,
                )
                continue  # don't double-parse this line

            # ── standard inline torque ───────────────────────────────────────
            for m in self._TORQUE_RE.finditer(line):
                nm_val = float(m.group("nm"))
                imp    = float(m.group("imp"))
                unit   = m.group("unit")

                lbft = imp if unit == "lb-ft" else None
                lbin = imp if unit == "lb-in" else None

                component = self._guess_component(m.group("pre") + m.group("post"))

                yield TextSegment(
                    text=line.strip(),
                    content_type=ContentType.INLINE_TORQUE,
                    section_id=li.section_id,
                    section_name=li.section_name,
                    subsection_type=li.subsection_type,
                    page_number=li.page_number,
                    line_number=li.idx,
                    component=component,
                    spec_value=SpecValue(value_nm=nm_val, value_lbft=lbft,
                                        value_lbin=lbin, raw=line.strip()),
                    vehicle_variant=self._extract_variant(line),
                    is_safety_critical=True,
                    has_condition="condition" in line.lower() or "when" in line.lower(),
                    confidence=0.92,
                )

            # ── dimensional specs ────────────────────────────────────────────
            for m in self._DIM_RE.finditer(line):
                mm_val = float(m.group("val"))
                yield TextSegment(
                    text=line.strip(),
                    content_type=ContentType.DIMENSIONAL_SPEC,
                    section_id=li.section_id,
                    section_name=li.section_name,
                    subsection_type=li.subsection_type,
                    page_number=li.page_number,
                    line_number=li.idx,
                    component=self._guess_component(line),
                    spec_value=SpecValue(value_mm=mm_val, raw=line.strip()),
                    vehicle_variant=self._extract_variant(line),
                    confidence=0.90,
                )

    @staticmethod
    def _extract_variant(text: str) -> List[str]:
        found: List[str] = []
        text_lower = text.lower()
        if "svt raptor" in text_lower:    found.append("SVT Raptor")
        if "rwd" in text_lower:           found.append("RWD")
        if "4wd" in text_lower:           found.append("4WD")
        if "all model" in text_lower:     found.append("All")
        return found

    @staticmethod
    def _guess_component(context: str) -> str:
        """
        Try to infer the component name from surrounding text.
        Looks for noun phrases near "tighten" or "bolt/nut/screw".
        """
        # Remove bullet/step markers
        cleaned = re.sub(r"^z\s+|^\d+\.\s+", "", context.strip())

        # Look for "tighten the <component>" pattern
        m = re.search(
            r"tighten(?:\s+the)?\s+(?:new\s+)?(?:three\s+new\s+)?(.{5,60?}?)"
            r"(?:\s+to\s+[\d\.]+\s*Nm|\s*$)",
            cleaned, re.IGNORECASE
        )
        if m:
            return m.group(1).strip().rstrip(",.")

        # Look for nouns before "to [number] Nm"
        m = re.search(r"(.{3,50}?)\s+to\s+[\d\.]+\s*Nm", cleaned, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            # Skip common verbs that aren't component names
            if not re.match(r"^(install|tighten|torque|apply|check|ensure)", candidate, re.I):
                return candidate[:80]

        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Procedure segment builder
# ─────────────────────────────────────────────────────────────────────────────

class ProcedureSegmentBuilder:
    """
    Groups consecutive procedure lines (numbered steps + bullet sub-steps)
    into a single PROCEDURE_STEP segment so that the procedure context
    travels with any embedded spec values in Phase 2.

    A procedure group ends when:
        • A new numbered step is encountered after 3+ lines
        • A section/subsection header appears
        • More than 2 blank lines separate lines

    Procedure segments include their parent section context so they can be
    retrieved alongside the torque spec they contain.
    """

    _STEP_START = re.compile(r"^\d{1,2}\.\s+")

    def build(
        self,
        line_infos: List[StructureParser.LineInfo],
    ) -> Iterator[TextSegment]:
        """Yield one TextSegment per logical procedure block."""

        buffer: List[str] = []
        ctx: Optional[StructureParser.LineInfo] = None

        def flush():
            if buffer and ctx:
                text = "\n".join(buffer)
                yield TextSegment(
                    text=text,
                    content_type=ContentType.PROCEDURE_STEP,
                    section_id=ctx.section_id,
                    section_name=ctx.section_name,
                    subsection_type=ctx.subsection_type,
                    page_number=ctx.page_number,
                    line_number=ctx.idx,
                    is_safety_critical=any(
                        kw in text.upper()
                        for kw in ("WARNING", "NOTICE", "CAUTION")
                    ),
                    confidence=1.0,
                )

        for li in line_infos:
            if li.content_type not in (
                ContentType.PROCEDURE_STEP, ContentType.WARNING_NOTICE, ContentType.GENERAL_TEXT
            ):
                yield from flush()
                buffer = []
                ctx = None
                continue

            if li.content_type in (ContentType.PROCEDURE_STEP, ContentType.WARNING_NOTICE):
                if self._STEP_START.match(li.text) and buffer:
                    yield from flush()
                    buffer = []
                buffer.append(li.text)
                if ctx is None:
                    ctx = li
            elif li.content_type == ContentType.GENERAL_TEXT and buffer:
                # Allow short general-text lines to join the current step (continuation)
                buffer.append(li.text)

        yield from flush()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionPipeline:
    """
    Orchestrates all extraction stages and saves outputs.

    Output structure:
        output/
            <stem>_cleaned.txt       — cleaned full text (inspection)
            <stem>_all_segments.json — ALL segments (for Phase 2 chunking)
            <stem>_spec_segments.json — ONLY spec segments (quick validation)
            <stem>_metadata.json     — document-level stats
    """

    def __init__(self, pdf_path: str | Path, out_dir: str | Path = "./output"):
        self.pdf_path = Path(pdf_path)
        self.out_dir  = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Stage instances
        self.loader    = PDFLoader(self.pdf_path)
        self.cleaner   = TextCleaner()
        self.structure = StructureParser()
        self.table_p   = TorqueTableParser()
        self.inline_p  = InlineSpecParser()
        self.proc_b    = ProcedureSegmentBuilder()

    # ── public ───────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute full pipeline. Returns a summary dict."""

        log.info("=" * 60)
        log.info("Phase 1 Extraction — %s", self.pdf_path.name)
        log.info("=" * 60)

        # ── Stage 1: Load ────────────────────────────────────────────────────
        log.info("[1/5] Loading document …")
        raw_text, metadata = self.loader.load()

        # ── Stage 2: Clean ───────────────────────────────────────────────────
        log.info("[2/5] Cleaning text …")
        clean_text = self.cleaner.clean(raw_text)
        metadata["cleaned_chars"] = len(clean_text)
        log.info("      %d → %d chars (%.1f%% retained)",
                 metadata.get("total_chars", len(raw_text)),
                 len(clean_text),
                 100 * len(clean_text) / max(1, metadata.get("total_chars", len(raw_text))))

        # ── Stage 3: Structural parse ────────────────────────────────────────
        log.info("[3/5] Parsing document structure …")
        line_infos = self.structure.parse(clean_text)
        metadata["total_classified_lines"] = len(line_infos)

        # ── Stage 4: Spec extraction ─────────────────────────────────────────
        log.info("[4/5] Extracting specifications …")
        table_segs  = list(self.table_p.extract(line_infos))
        inline_segs = list(self.inline_p.extract(line_infos))
        proc_segs   = list(self.proc_b.build(line_infos))

        spec_segments = table_segs + inline_segs
        all_segments  = spec_segments + proc_segs

        # De-duplicate specs by (component, nm_value, section) to remove
        # repeated specs scattered across multiple references
        deduped_specs = self._deduplicate(spec_segments)

        log.info("      Torque table rows : %d", len(table_segs))
        log.info("      Inline torque     : %d", len(inline_segs))
        log.info("      Procedure blocks  : %d", len(proc_segs))
        log.info("      Unique specs      : %d", len(deduped_specs))
        
        # ── Fallback: If no specs found, use all text as segments ──────────
        if not deduped_specs:
            log.info("      ⚠ No specific specs found. Using full text as segments.")
            deduped_specs = all_segments if all_segments else self._text_to_paragraphs(clean_text)
        if not all_segments:
            all_segments = list(deduped_specs)

        metadata.update({
            "torque_table_rows":  len(table_segs),
            "inline_torque_segs": len(inline_segs),
            "procedure_segments": len(proc_segs),
            "unique_spec_count":  len(deduped_specs),
        })

        # ── Stage 5: Save ────────────────────────────────────────────────────
        log.info("[5/5] Saving outputs …")
        stem = self.pdf_path.stem.replace(" ", "_")

        paths = {
            "cleaned_text":   self._save_text(clean_text, f"{stem}_cleaned.txt"),
            "all_segments":   self._save_json(
                [s.to_dict() for s in all_segments], f"{stem}_all_segments.json"
            ),
            "spec_segments":  self._save_json(
                [s.to_dict() for s in deduped_specs], f"{stem}_spec_segments.json"
            ),
            "metadata":       self._save_json(metadata, f"{stem}_metadata.json"),
        }

        return {
            "metadata":      metadata,
            "paths":         paths,
            "all_segments":  all_segments,
            "spec_segments": deduped_specs,
        }

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _text_to_paragraphs(text: str) -> List[TextSegment]:
        """Convert raw text into paragraph segments when no specs found."""
        segments = []
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs and text.strip():
            paragraphs = [text.strip()]
        
        for i, para in enumerate(paragraphs[:100]):  # limit to 100 paragraphs
            seg = TextSegment(
                text=para,
                content_type=ContentType.PROCEDURE_STEP,
                component="General Text",
                section_id=f"para_{i}",
                section_name="Extracted Text",
                page_number=0,
            )
            segments.append(seg)
        
        return segments

    @staticmethod
    def _deduplicate(segments: List[TextSegment]) -> List[TextSegment]:
        """
        Remove near-duplicate specs.
        Key: (component_lower, nm_value, section_id)
        When duplicates exist, prefer the TABLE_ROW version (higher confidence).
        """
        seen: dict = {}
        for seg in segments:
            nm = seg.spec_value.value_nm if seg.spec_value else None
            key = (
                seg.component.lower().strip()[:60],
                nm,
                seg.section_id,
            )
            if key not in seen:
                seen[key] = seg
            elif seg.content_type == ContentType.TORQUE_TABLE_ROW:
                seen[key] = seg  # prefer table row
        return list(seen.values())

    def _save_json(self, data: list | dict, filename: str) -> str:
        path = self.out_dir / filename
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
        log.info("      → %s (%d KB)", path.name, path.stat().st_size // 1024)
        return str(path)

    def _save_text(self, text: str, filename: str) -> str:
        path = self.out_dir / filename
        path.write_text(text, encoding="utf-8")
        log.info("      → %s (%d KB)", path.name, path.stat().st_size // 1024)
        return str(path)

    @staticmethod
    def _print_summary(specs: List[TextSegment], meta: dict) -> None:
        print()
        print("=" * 60)
        print("  PHASE 1 EXTRACTION SUMMARY")
        print("=" * 60)

        # Count by type
        by_type: dict = {}
        by_section: dict = {}
        for s in specs:
            by_type[s.content_type.value] = by_type.get(s.content_type.value, 0) + 1
            key = f"{s.section_id} — {s.section_name}"[:50]
            by_section[key] = by_section.get(key, 0) + 1

        print(f"\n  Document stats:")
        print(f"    Source file    : {meta.get('source_file', '?')}")
        print(f"    Extraction     : {meta.get('extraction_method', '?')}")
        print(f"    Characters     : {meta.get('total_chars', '?'):,} raw → "
              f"{meta.get('cleaned_chars', '?'):,} cleaned")

        print(f"\n  Spec segments by type:")
        for ctype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            bar = "█" * min(count // 2, 30)
            print(f"    {ctype:<25} {count:>4}  {bar}")

        print(f"\n  Specs by section (top 10):")
        for sec, count in sorted(by_section.items(), key=lambda x: -x[1])[:10]:
            print(f"    {sec:<50} {count:>4}")

        # Sample table rows
        table_rows = [s for s in specs if s.content_type == ContentType.TORQUE_TABLE_ROW]
        if table_rows:
            print(f"\n  Sample extracted table specs (first 5):")
            for s in table_rows[:5]:
                sv = s.spec_value
                nm   = f"{sv.value_nm} Nm" if sv and sv.value_nm else "—"
                lbft = f"{sv.value_lbft} lb-ft" if sv and sv.value_lbft else ""
                lbin = f"{sv.value_lbin} lb-in" if sv and sv.value_lbin else ""
                unit_str = " / ".join(filter(None, [nm, lbft, lbin]))
                print(f"    • {s.component:<45} → {unit_str}")

        print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1: Extract text & specs from a vehicle service manual PDF"
    )
    parser.add_argument("pdf", help="Path to the PDF or text file")
    parser.add_argument("--out-dir", default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = ExtractionPipeline(args.pdf, out_dir=args.out_dir)
    results  = pipeline.run()

    # Always exit 0 - even if no specs found, we output text segments
    sys.exit(0)