"""
Phase 1 — Generic Automotive PDF Extractor
===========================================
Handles any vehicle service manual, not just the F-150.

Supported unit systems detected automatically:
  • SI:        N·m  /  kgf·cm  /  kN  (Toyota, Honda, Nissan, …)
  • Imperial:  lb-ft  /  lb-in  /  ft·lbf  (US domestic)
  • Mixed:     Nm  lb-ft  lb-in  (Ford-style)
  • With ranges:  1.5–3.0 N·m  /  25-35 lb-ft

Output schema is identical to pdf_extractor.py so Phase 2 can consume it.

Usage:
    python pdf_extractor_generic.py  manual.pdf
    python pdf_extractor_generic.py  manual.pdf  --out-dir ./results
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase1g")


# ─────────────────────────────────────────────────────────────────────────────
# Data models  (same as pdf_extractor.py so Phase 2 schema is unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class ContentType(str, Enum):
    TORQUE_TABLE_ROW   = "torque_table_row"
    INLINE_TORQUE      = "inline_torque"
    MULTI_STAGE_TORQUE = "multi_stage_torque"
    DIMENSIONAL_SPEC   = "dimensional_spec"
    PROCEDURE_STEP     = "procedure_step"
    TABLE_HEADER       = "table_header"
    SECTION_HEADER     = "section_header"
    SUBSECTION_HEADER  = "subsection_header"
    WARNING_NOTICE     = "warning_notice"
    DIAGNOSTIC         = "diagnostic"
    PART_NUMBER        = "part_number"
    GENERAL_TEXT       = "general_text"


@dataclass
class SpecValue:
    value_nm:   Optional[float] = None
    value_lbft: Optional[float] = None
    value_lbin: Optional[float] = None
    value_kgcm: Optional[float] = None   # kgf·cm (Toyota/JDM)
    value_mm:   Optional[float] = None
    value_deg:  Optional[float] = None
    raw:        str = ""


@dataclass
class TextSegment:
    text:              str
    content_type:      ContentType
    section_id:        str  = ""
    section_name:      str  = ""
    subsection_type:   str  = ""
    page_number:       int  = 0
    line_number:       int  = 0
    component:         str  = ""
    spec_value:        Optional[SpecValue] = None
    vehicle_variant:   List[str] = field(default_factory=list)
    stage_number:      int  = 0
    is_safety_critical: bool = False
    has_condition:      bool = False
    confidence:         float = 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["content_type"] = self.content_type.value
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — PDF / text loading  (same waterfall as pdf_extractor.py)
# ─────────────────────────────────────────────────────────────────────────────

class PDFLoader:
    """
    Waterfall:
      1. PyMuPDF (fitz)     — primary, handles broken xrefs
      2. pdfminer.six       — secondary
      3. pdfplumber         — tertiary, good table layout
      4. pdftotext subprocess — quaternary (poppler)
      5. Plain text fallback — if file is already a .txt
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> Tuple[str, dict]:
        with open(self.path, "rb") as fh:
            header = fh.read(8)
        is_pdf = header.startswith(b"%PDF")

        if is_pdf:
            return self._load_pdf()
        else:
            log.info("Not a PDF — reading as text file")
            return self._load_text()

    # ── PDF paths ─────────────────────────────────────────────────────────────

    def _load_pdf(self) -> Tuple[str, dict]:
        # 1. PyMuPDF
        text = self._try_pymupdf()
        if len(text.strip()) > 1000:
            log.info("PyMuPDF: %d chars extracted", len(text))
            return text, self._meta(text, "pymupdf")

        # 2. pdfminer.six
        log.info("PyMuPDF yielded too little — trying pdfminer.six …")
        text = self._try_pdfminer()
        if len(text.strip()) > 1000:
            log.info("pdfminer: %d chars extracted", len(text))
            return text, self._meta(text, "pdfminer")

        # 3. pdfplumber
        log.info("pdfminer yielded too little — trying pdfplumber …")
        text = self._try_pdfplumber()
        if len(text.strip()) > 1000:
            log.info("pdfplumber: %d chars extracted", len(text))
            return text, self._meta(text, "pdfplumber")

        # 4. pdftotext subprocess
        log.info("pdfplumber yielded too little — trying pdftotext …")
        text = self._try_pdftotext()
        if len(text.strip()) > 1000:
            log.info("pdftotext: %d chars extracted", len(text))
            return text, self._meta(text, "pdftotext")

        # 5. Neighbouring .txt files
        log.warning("All PDF extraction methods failed — searching for companion .txt …")
        for candidate in [
            self.path.with_suffix(".txt"),
            self.path.parent / (self.path.stem + "_cleaned.txt"),
            self.path.parent / "extracted_text_cleaned.txt",
            self.path.parent / "extracted_text.txt",
        ]:
            if candidate.exists() and candidate != self.path:
                log.info("Using companion text file: %s", candidate.name)
                return self._load_text_from(candidate)

        log.error("No usable text extracted from %s", self.path.name)
        return "", self._meta("", "failed")

    def _try_pymupdf(self) -> str:
        try:
            import fitz
            doc = fitz.open(str(self.path))
            parts = []
            for i, page in enumerate(doc, 1):
                t = page.get_text("text")
                if t.strip():
                    parts.append(f"<<PAGE {i}>>\n{t}")
            doc.close()
            return "\n".join(parts)
        except Exception as e:
            log.debug("PyMuPDF failed: %s", e)
            return ""

    def _try_pdfminer(self) -> str:
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(str(self.path))
            return text or ""
        except Exception as e:
            log.debug("pdfminer failed: %s", e)
            return ""

    def _try_pdfplumber(self) -> str:
        try:
            import pdfplumber
            parts = []
            with pdfplumber.open(str(self.path)) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    t = page.extract_text() or ""
                    if t.strip():
                        parts.append(f"<<PAGE {i}>>\n{t}")
            return "\n".join(parts)
        except Exception as e:
            log.debug("pdfplumber failed: %s", e)
            return ""

    def _try_pdftotext(self) -> str:
        import subprocess
        try:
            r = subprocess.run(
                ["pdftotext", "-layout", "-enc", "UTF-8", str(self.path), "-"],
                capture_output=True, text=True, timeout=120,
            )
            text = r.stdout or ""
            # Reject output that's only control chars (garbage)
            printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
            if r.returncode == 0 and printable > 500:
                return text
            return ""
        except Exception as e:
            log.debug("pdftotext failed: %s", e)
            return ""

    # ── Text paths ────────────────────────────────────────────────────────────

    def _load_text(self) -> Tuple[str, dict]:
        return self._load_text_from(self.path)

    def _load_text_from(self, p: Path) -> Tuple[str, dict]:
        text = p.read_text(encoding="utf-8", errors="replace")
        log.info("Text file: %d chars, %d lines", len(text), text.count("\n"))
        return text, self._meta(text, "text_file")

    def _meta(self, text: str, method: str) -> dict:
        return {
            "source_file":       self.path.name,
            "extraction_method": method,
            "total_chars":       len(text),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1b — pdfplumber coordinate-based table extraction (parallel pass)
# ─────────────────────────────────────────────────────────────────────────────

class PDFTableExtractor:
    """
    Second extraction pass using pdfplumber's geometry-aware table detection.
    Returns structured rows even when text-mode extraction misses column alignment.
    """

    # Units we consider "torque-bearing" — broad to catch any manual
    TORQUE_UNITS = re.compile(
        r"\b(?:N[·.]?m|Nm|kgf[·.]?cm|kgf\.cm|ft[·.]?lb[fs]?|lb[·.]?ft|lb[·.]?in|in[·.]?lb)\b",
        re.IGNORECASE,
    )

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def extract(self) -> List[dict]:
        """Return list of {page_num, is_torque_table, rows: [[cell,...],...]}"""
        results = []
        try:
            import pdfplumber
        except ImportError:
            return results

        try:
            with pdfplumber.open(str(self.path)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Try with explicit line strategies first, fall back to text
                    for settings in [
                        {"vertical_strategy": "lines",  "horizontal_strategy": "lines"},
                        {"vertical_strategy": "text",   "horizontal_strategy": "text"},
                    ]:
                        tables = page.extract_tables(settings)
                        if tables:
                            break
                    for table in (tables or []):
                        if not table:
                            continue
                        is_torque = any(
                            self.TORQUE_UNITS.search(str(cell))
                            for row in table for cell in (row or []) if cell
                        )
                        clean_rows = []
                        for row in table:
                            clean_row = [str(c).strip() if c else "" for c in (row or [])]
                            if any(clean_row):
                                clean_rows.append(clean_row)
                        if clean_rows:
                            results.append({
                                "page_num":       page_num,
                                "is_torque_table": is_torque,
                                "rows":           clean_rows,
                            })
        except Exception as e:
            log.debug("PDFTableExtractor failed: %s", e)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Generic text cleaner
# ─────────────────────────────────────────────────────────────────────────────

class TextCleaner:
    _CTRL    = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
    _URL     = re.compile(r"file:///[^\n]+|https?://\S+")
    _BLANKS  = re.compile(r"\n{4,}")
    _SPACES  = re.compile(r"[ \t]{3,}")

    def clean(self, text: str) -> str:
        text = self._CTRL.sub("", text)
        text = self._URL.sub("", text)
        text = self._SPACES.sub(" ", text)
        text = self._BLANKS.sub("\n\n\n", text)
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        # Normalise various dashes to ASCII for easier regex matching later
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        # Normalise middle dot variants for N·m etc.
        text = text.replace("\u00b7", "\u00b7")  # keep middle-dot (U+00B7) as-is
        return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Generic structure parser
# ─────────────────────────────────────────────────────────────────────────────

class StructureParser:
    """
    Recognises section/subsection/page structure in any automotive manual.
    Far more permissive than the F-150-specific version.
    """

    # <<PAGE N>> markers injected by PyMuPDF loader
    _RE_PAGE = re.compile(r"^<<PAGE\s+(\d+)>>$")

    # Ford-style: "SECTION 204-01A: Front Suspension"
    _RE_FORD_SECTION = re.compile(
        r"^SECTION\s+([\w\-]+):\s*(.+)$", re.IGNORECASE
    )

    # Toyota/JDM-style: "SS-40  SERVICE SPECIFICATIONS - SUSPENSION AND AXLE"
    # Also matches: "EM-1  ENGINE MECHANICAL" or "BR-12  BRAKES"
    _RE_JDM_SECTION = re.compile(
        r"^([A-Z]{1,4}[-–]\d+[A-Z]?)\s{2,}(.+)$"
    )

    # All-caps heading, 3-60 chars — subsection or major header
    _RE_ALLCAPS = re.compile(r"^([A-Z][A-Z\s/&,()\-]{2,58})$")

    # Subsection keywords (universal)
    _SUBSECTIONS = re.compile(
        r"^(?:SPECIFICATIONS?|DIAGNOSIS AND TESTING|DESCRIPTION AND OPERATION|"
        r"REMOVAL AND INSTALLATION|DISASSEMBLY AND ASSEMBLY|GENERAL PROCEDURES?|"
        r"SERVICE SPECIFICATIONS?|INSPECTION AND VERIFICATION|ADJUSTMENTS?|"
        r"TORQUE SPECIFICATIONS?|FASTENER SPECIFICATIONS?|TIGHTENING TORQUES?)\s*$",
        re.IGNORECASE,
    )

    _RE_WARNING  = re.compile(r"^(WARNING|NOTICE|NOTE|CAUTION)\b", re.IGNORECASE)
    _RE_STEP_NUM = re.compile(r"^\d{1,2}\.\s+")
    _RE_BULLET   = re.compile(r"^[z•\-\*]\s+")

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
        lines  = text.split("\n")
        result: List[StructureParser.LineInfo] = []

        cur_section_id   = ""
        cur_section_name = ""
        cur_subsection   = ""
        cur_page         = 0

        for idx, raw in enumerate(lines):
            line = raw.strip()
            if not line:
                continue

            # Page marker
            m = self._RE_PAGE.match(line)
            if m:
                cur_page = int(m.group(1))
                continue

            # Ford-style section
            m = self._RE_FORD_SECTION.match(line)
            if m:
                cur_section_id   = m.group(1).strip()
                cur_section_name = m.group(2).strip()
                cur_subsection   = ""
                result.append(self.LineInfo(idx, line, ContentType.SECTION_HEADER,
                                            cur_section_id, cur_section_name,
                                            "", cur_page))
                continue

            # JDM-style section header (e.g. "SS-40  SERVICE SPECIFICATIONS")
            m = self._RE_JDM_SECTION.match(line)
            if m:
                cur_section_id   = m.group(1).strip()
                cur_section_name = m.group(2).strip()
                cur_subsection   = ""
                result.append(self.LineInfo(idx, line, ContentType.SECTION_HEADER,
                                            cur_section_id, cur_section_name,
                                            "", cur_page))
                continue

            # Subsection keyword
            if self._SUBSECTIONS.match(line):
                cur_subsection = line.upper()
                result.append(self.LineInfo(idx, line, ContentType.SUBSECTION_HEADER,
                                            cur_section_id, cur_section_name,
                                            cur_subsection, cur_page))
                continue

            ctype = self._classify(line)
            result.append(self.LineInfo(idx, line, ctype,
                                        cur_section_id, cur_section_name,
                                        cur_subsection, cur_page))

        log.info("StructureParser: %d non-blank lines classified", len(result))
        return result

    def _classify(self, line: str) -> ContentType:
        if self._RE_WARNING.match(line):
            return ContentType.WARNING_NOTICE
        if self._RE_STEP_NUM.match(line):
            return ContentType.PROCEDURE_STEP
        if self._RE_BULLET.match(line):
            return ContentType.PROCEDURE_STEP
        if self._RE_ALLCAPS.match(line) and len(line) < 60:
            return ContentType.SUBSECTION_HEADER
        return ContentType.GENERAL_TEXT


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4a — Generic torque table parser
# ─────────────────────────────────────────────────────────────────────────────

class GenericTorqueTableParser:
    """
    Detects torque tables in any unit system and extracts rows.

    Supported table header patterns:
      Ford:   "Description  Nm  lb-ft  lb-in"
      Toyota: "Part tightened  N·m  kgf·cm  ft·lbf"
      Honda:  "Torque  N·m (kgf·m, ft·lbf)"
      Generic: any line with 2+ torque unit tokens

    Row pattern: text + 1..4 numeric columns (numbers or dashes)
    """

    # Torque unit tokens (broad — catch N·m, N.m, Nm, kgf·cm, ft·lbf, lb-ft, …)
    _UNIT_TOK = re.compile(
        r"N[·.]?m|kgf[·.]?cm|ft[·.]?lb[fs]?|lb[·.]?ft|lb[·.]?in|in[·.]?lb|"
        r"ft[·.]?lbf|lbf[·.]?in|kN[·.]?m",
        re.IGNORECASE,
    )

    # A table header has >= 2 unit tokens on a single line
    def _is_table_header(self, line: str) -> bool:
        return len(self._UNIT_TOK.findall(line)) >= 2

    # A table header that only lists units (no description prefix)
    _PURE_HEADER = re.compile(
        r"^\s*(?:Description|Part(?:\s+tightened)?|Component|Fastener)?"
        r"\s*N[·.]?m"
        r"[\s\S]{0,60}(?:lb[·.]?ft|kgf|ft[·.]?lb)",
        re.IGNORECASE,
    )

    # Numeric cell: a number (with optional range) or a dash
    _NUM = re.compile(r"^([\d,]+(?:\.\d+)?(?:\s*[-–]\s*[\d,]+(?:\.\d+)?)?|[-–])$")

    # Row: "Some text ... value value value"
    _ROW_RE = re.compile(
        r"^(.+?)\s+"
        r"([\d,]+(?:\.\d+)?(?:\s*[-–]\s*[\d,]+(?:\.\d+)?)?|[-–])\s+"
        r"([\d,]+(?:\.\d+)?(?:\s*[-–]\s*[\d,]+(?:\.\d+)?)?|[-–])"
        r"(?:\s+([\d,]+(?:\.\d+)?(?:\s*[-–]\s*[\d,]+(?:\.\d+)?)?|[-–]))?"
        r"\s*$"
    )

    # Lines that cannot be table data rows
    _NOT_ROW = re.compile(
        r"^(?:SECTION|Note|WARNING|NOTICE|file://|\d{4}-\d{2}-\d{2}|Page\s)",
        re.IGNORECASE,
    )

    def extract(
        self,
        line_infos: List[StructureParser.LineInfo],
        unit_map: "TableUnitMap",
    ) -> Iterator[TextSegment]:

        in_table          = False
        col_units: List[str] = []
        table_sec_id   = ""
        table_sec_name = ""
        table_subsec   = ""
        table_page     = 0

        i = 0
        while i < len(line_infos):
            li = line_infos[i]

            # Detect a new table header
            if self._is_table_header(li.text):
                in_table       = True
                col_units      = self._UNIT_TOK.findall(li.text)
                table_sec_id   = li.section_id
                table_sec_name = li.section_name
                table_subsec   = li.subsection_type
                table_page     = li.page_number
                result_li = self.LineInfo_factory(li, ContentType.TABLE_HEADER)
                i += 1
                continue

            if in_table:
                # Section boundary ends the table
                if li.content_type in (ContentType.SECTION_HEADER,
                                       ContentType.SUBSECTION_HEADER):
                    in_table = False
                    i += 1
                    continue

                m = self._ROW_RE.match(li.text)
                if m and not self._NOT_ROW.match(li.text):
                    desc = m.group(1).strip()
                    vals = [m.group(2), m.group(3), m.group(4)]

                    # Parse up to 3 column values
                    parsed = [self._parse_num(v) for v in vals]
                    col0, col1, col2 = parsed[0], parsed[1], parsed[2]

                    if col0 is None and col1 is None:
                        i += 1
                        continue

                    sv = self._build_spec_value(col_units, col0, col1, col2, li.text)

                    if sv.value_nm is not None or sv.value_lbft is not None or sv.value_kgcm is not None:
                        readable = self._format(desc, sv, table_sec_name, col_units)
                        yield TextSegment(
                            text=readable,
                            content_type=ContentType.TORQUE_TABLE_ROW,
                            section_id=table_sec_id,
                            section_name=table_sec_name,
                            subsection_type=table_subsec,
                            page_number=table_page,
                            line_number=li.idx,
                            component=desc,
                            spec_value=sv,
                            is_safety_critical=True,
                            confidence=0.97,
                        )
                        i += 1
                        continue

                # Stacked format: description line, then 2-3 value lines
                if (i + 2 < len(line_infos)
                        and not self._NOT_ROW.match(li.text)
                        and not self._is_value_tok(li.text)
                        and self._is_value_tok(line_infos[i + 1].text)
                        and self._is_value_tok(line_infos[i + 2].text)):
                    desc = li.text.strip()
                    col0 = self._parse_num(line_infos[i + 1].text)
                    col1 = self._parse_num(line_infos[i + 2].text)
                    col2 = (self._parse_num(line_infos[i + 3].text)
                            if i + 3 < len(line_infos)
                               and self._is_value_tok(line_infos[i + 3].text)
                            else None)

                    if col0 is not None or col1 is not None:
                        sv = self._build_spec_value(col_units, col0, col1, col2, li.text)
                        if sv.value_nm is not None or sv.value_lbft is not None or sv.value_kgcm is not None:
                            readable = self._format(desc, sv, table_sec_name, col_units)
                            yield TextSegment(
                                text=readable,
                                content_type=ContentType.TORQUE_TABLE_ROW,
                                section_id=table_sec_id,
                                section_name=table_sec_name,
                                subsection_type=table_subsec,
                                page_number=table_page,
                                line_number=li.idx,
                                component=desc,
                                spec_value=sv,
                                is_safety_critical=True,
                                confidence=0.95,
                            )
                            i += (4 if col2 is not None else 3)
                            continue
            i += 1

    @staticmethod
    def LineInfo_factory(li, ctype):
        """Dummy — we don't actually use this; kept for clarity."""
        return li

    def _build_spec_value(
        self,
        col_units: List[str],
        col0: Optional[float],
        col1: Optional[float],
        col2: Optional[float],
        raw: str,
    ) -> SpecValue:
        """
        Map column values to physical quantities using the detected column header units.
        Column order mirrors the header: typically col0=Nm, col1=kgf·cm or lb-ft, col2=ft·lbf or lb-in
        """
        sv = SpecValue(raw=raw.strip())

        def _norm(u: str) -> str:
            return u.lower().replace("·", ".").replace(" ", "")

        normed = [_norm(u) for u in col_units]

        for j, (val, unit_raw) in enumerate(zip([col0, col1, col2], normed)):
            if val is None:
                continue
            u = unit_raw
            if "n.m" in u or u == "nm" or u == "n·m":
                sv.value_nm = val
            elif "kgf" in u:
                sv.value_kgcm = val
            elif "lb.ft" in u or "ft.lb" in u or "lbft" in u or "ft.lbf" in u:
                sv.value_lbft = val
            elif "lb.in" in u or "in.lb" in u or "lbin" in u:
                sv.value_lbin = val

        # Fallback: if we couldn't map any column headers but got numbers, treat
        # col0 as Nm and col1 as lb-ft (the most common layout).
        if sv.value_nm is None and sv.value_lbft is None and sv.value_kgcm is None:
            if col0 is not None:
                sv.value_nm = col0
            if col1 is not None:
                sv.value_lbft = col1

        return sv

    @staticmethod
    def _parse_num(s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        s = s.strip()
        if s in ("—", "-", "", "N/A", "—"):
            return None
        # Remove thousands comma
        s = s.replace(",", "")
        m = re.match(r"^([\d.]+)\s*[-–]\s*([\d.]+)$", s)
        if m:
            try:
                return float(m.group(1))   # lower bound of range
            except ValueError:
                return None
        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def _is_value_tok(s: str) -> bool:
        s = s.strip().replace(",", "")
        return bool(re.match(
            r"^(?:[\d.]+(?:\s*[-–]\s*[\d.]+)?|[-–]|N/A)$", s, re.IGNORECASE
        ))

    @staticmethod
    def _format(desc: str, sv: SpecValue, section: str, units: List[str]) -> str:
        parts = [f"Component: {desc}"]
        if sv.value_nm   is not None: parts.append(f"Torque: {sv.value_nm} N·m")
        if sv.value_kgcm is not None: parts.append(f"({sv.value_kgcm} kgf·cm)")
        if sv.value_lbft is not None: parts.append(f"({sv.value_lbft} lb-ft)")
        if sv.value_lbin is not None: parts.append(f"({sv.value_lbin} lb-in)")
        if section:                   parts.append(f"[{section}]")
        return " — ".join(parts)


class TableUnitMap:
    """Placeholder — unit mapping is resolved directly in GenericTorqueTableParser."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4a-extra — pdfplumber table rows → TextSegments
# ─────────────────────────────────────────────────────────────────────────────

class PlumberTableConverter:
    """
    Converts raw pdfplumber tables (list of rows) into TextSegments.
    Torque tables → TORQUE_TABLE_ROW, others → GENERAL_TEXT with is_table_content.
    """

    _TORQUE_UNIT = re.compile(
        r"\b(?:N[·.]?m|Nm|kgf[·.]?cm|ft[·.]?lbf?|lb[·.]?ft|lb[·.]?in)\b",
        re.IGNORECASE,
    )
    _NUM_RE = re.compile(r"^\d[\d,\.]*$")

    def convert(self, tables: List[dict]) -> List[TextSegment]:
        segments: List[TextSegment] = []
        for tbl in tables:
            page     = tbl["page_num"]
            is_torq  = tbl["is_torque_table"]
            rows     = tbl["rows"]
            if not rows:
                continue

            # Try to detect column headers from first row
            header_row = rows[0] if rows else []

            for r_idx, row in enumerate(rows[1:] if len(rows) > 1 else rows, 1):
                if not any(c.strip() for c in row):
                    continue

                text = " | ".join(c if c else "—" for c in row)

                if is_torq:
                    # Parse out numeric values
                    sv = self._parse_torque_row(header_row, row)
                    desc = row[0].strip() if row else ""
                    ctype = ContentType.TORQUE_TABLE_ROW
                else:
                    sv   = None
                    desc = ""
                    ctype = ContentType.GENERAL_TEXT

                segments.append(TextSegment(
                    text=text,
                    content_type=ctype,
                    page_number=page,
                    component=desc,
                    spec_value=sv,
                    is_safety_critical=is_torq,
                    confidence=0.90,
                ))

        return segments

    def _parse_torque_row(self, header: List[str], row: List[str]) -> Optional[SpecValue]:
        sv = SpecValue(raw=" | ".join(row))
        for col_i, cell in enumerate(row[1:], 1):
            if col_i >= len(header):
                break
            hdr = header[col_i] if header else ""
            val_str = cell.replace(",", "").strip()
            val = None
            try:
                val = float(val_str)
            except (ValueError, AttributeError):
                pass

            if val is None:
                continue

            hdr_l = hdr.lower()
            if "n·m" in hdr_l or "n.m" in hdr_l or hdr_l in ("nm", "n·m"):
                sv.value_nm = val
            elif "kgf" in hdr_l:
                sv.value_kgcm = val
            elif "lb-ft" in hdr_l or "ft-lb" in hdr_l or "ft·lbf" in hdr_l:
                sv.value_lbft = val
            elif "lb-in" in hdr_l or "in-lb" in hdr_l:
                sv.value_lbin = val

        if sv.value_nm is None and sv.value_lbft is None and sv.value_kgcm is None:
            return None
        return sv


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4b — Generic inline spec parser
# ─────────────────────────────────────────────────────────────────────────────

class GenericInlineSpecParser:
    """
    Extracts torque/dimensional specs from procedure text.

    Handles:
      • "Tighten to 90 Nm (66 lb-ft)"               — Ford style
      • "Tighten to 209 N·m (2,131 kgf·cm, 154 ft·lbf)"  — Toyota style
      • "Install and tighten to 35 N·m."             — bare Nm
      • "1.5–3.0 N·m"                               — range value
      • "Stage 1: Tighten to 35 Nm (26 lb-ft)"       — multi-stage
      • "15.5 mm (0.61 in)"                          — dimensional
    """

    # Primary: NM with optional imperial equivalent in parens
    # Handles both "Nm" and "N·m" and ranges like "1.5-3.0"
    _TORQUE_RE = re.compile(
        r"(?P<pre>[^.]{0,120}?)"
        r"(?P<nm>[\d,]+\.?\d*(?:\s*[-–]\s*[\d,]+\.?\d*)?)\s*"
        r"(?P<nm_unit>N[·.]?m|Nm)"
        r"(?:\s*\((?P<imp1>[\d,]+\.?\d*(?:\s*[-–]\s*[\d,]+\.?\d*)?)\s*"
        r"(?P<imp1_unit>kgf[·.]?cm|kgf\.cm|kgf·cm)(?:,\s*"
        r"(?P<imp2>[\d,]+\.?\d*(?:\s*[-–]\s*[\d,]+\.?\d*)?)\s*"
        r"(?P<imp2_unit>ft[·.]?lbf?|lb[·.]?ft))?\))?"
        r"(?:\s*\((?P<lb>[\d,]+\.?\d*(?:\s*[-–]\s*[\d,]+\.?\d*)?)\s*"
        r"(?P<lb_unit>lb[·.]?ft|lb[·.]?in|ft[·.]?lb[fs]?)\))?",
        re.IGNORECASE,
    )

    # Stage pattern: "Stage N: ... X Nm"
    _STAGE_RE = re.compile(
        r"Stage\s+(?P<stage>\d)\s*:.*?"
        r"(?P<nm>[\d,]+\.?\d*)\s*(?P<nm_unit>N[·.]?m|Nm)"
        r"(?:\s*\((?P<imp>[\d,]+\.?\d*)\s*(?P<imp_unit>lb[·.]?ft|ft[·.]?lbf?|kgf[·.]?cm)\))?",
        re.IGNORECASE,
    )

    # Dimensional: "15.5 mm (0.61 in)"
    _DIM_RE = re.compile(
        r"(?P<val>[\d.]+)\s*mm\s*\((?P<imp>[\d.]+)\s*in\)",
        re.IGNORECASE,
    )

    def extract(
        self, line_infos: List[StructureParser.LineInfo],
    ) -> Iterator[TextSegment]:

        for li in line_infos:
            line = li.text

            # Multi-stage first
            m = self._STAGE_RE.search(line)
            if m:
                nm  = self._fnum(m.group("nm"))
                imp = self._fnum(m.group("imp")) if m.group("imp") else None
                unit = (m.group("imp_unit") or "").lower()
                lbft = imp if "lb" in unit and "in" not in unit else None
                lbin = imp if "in" in unit else None
                kgcm = imp if "kgf" in unit else None
                yield TextSegment(
                    text=line.strip(),
                    content_type=ContentType.MULTI_STAGE_TORQUE,
                    section_id=li.section_id, section_name=li.section_name,
                    subsection_type=li.subsection_type,
                    page_number=li.page_number, line_number=li.idx,
                    component=self._guess_component(line),
                    spec_value=SpecValue(value_nm=nm, value_lbft=lbft, value_lbin=lbin,
                                        value_kgcm=kgcm, raw=line.strip()),
                    stage_number=int(m.group("stage")),
                    is_safety_critical=True, confidence=0.95,
                )
                continue

            # Standard torque
            for m in self._TORQUE_RE.finditer(line):
                nm_str = m.group("nm") or ""
                nm     = self._fnum(nm_str.split("-")[0].split("–")[0])  # lower of range
                if nm is None or nm <= 0:
                    continue

                lbft  = self._fnum(m.group("imp2")) if m.group("imp2") else None
                if lbft is None:
                    lb_str  = m.group("lb") or ""
                    lb_unit = (m.group("lb_unit") or "").lower()
                    lbft = self._fnum(lb_str) if "ft" in lb_unit and "in" not in lb_unit else None
                    lbin = self._fnum(lb_str) if "in" in lb_unit else None
                else:
                    lbin = None

                kgcm = self._fnum(m.group("imp1")) if m.group("imp1") else None

                yield TextSegment(
                    text=line.strip(),
                    content_type=ContentType.INLINE_TORQUE,
                    section_id=li.section_id, section_name=li.section_name,
                    subsection_type=li.subsection_type,
                    page_number=li.page_number, line_number=li.idx,
                    component=self._guess_component(m.group("pre") or ""),
                    spec_value=SpecValue(value_nm=nm, value_lbft=lbft, value_lbin=lbin,
                                        value_kgcm=kgcm, raw=line.strip()),
                    is_safety_critical=True,
                    has_condition="when" in line.lower() or "if " in line.lower(),
                    confidence=0.90,
                )

            # Dimensional
            for m in self._DIM_RE.finditer(line):
                mm = self._fnum(m.group("val"))
                if mm:
                    yield TextSegment(
                        text=line.strip(),
                        content_type=ContentType.DIMENSIONAL_SPEC,
                        section_id=li.section_id, section_name=li.section_name,
                        subsection_type=li.subsection_type,
                        page_number=li.page_number, line_number=li.idx,
                        component=self._guess_component(line),
                        spec_value=SpecValue(value_mm=mm, raw=line.strip()),
                        confidence=0.88,
                    )

    @staticmethod
    def _fnum(s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        s = str(s).strip().replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def _guess_component(ctx: str) -> str:
        ctx = re.sub(r"^[z•\-\*]\s+|^\d+\.\s+", "", ctx.strip())
        m = re.search(
            r"tighten(?:\s+the)?\s+(?:new\s+)?(.{5,60?}?)"
            r"(?:\s+to\s+[\d,\.]+\s*N[·.]?m|\s*$)",
            ctx, re.IGNORECASE
        )
        if m:
            return m.group(1).strip().rstrip(",.")
        m = re.search(r"(.{3,50}?)\s+to\s+[\d,\.]+\s*N[·.]?m", ctx, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if not re.match(r"^(install|tighten|torque|apply|check)", candidate, re.I):
                return candidate[:80]
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Procedure segment builder
# ─────────────────────────────────────────────────────────────────────────────

class ProcedureSegmentBuilder:
    _STEP = re.compile(r"^\d{1,2}\.\s+")

    def build(self, line_infos: List[StructureParser.LineInfo]) -> Iterator[TextSegment]:
        buffer: List[str] = []
        ctx: Optional[StructureParser.LineInfo] = None

        def _flush():
            if buffer and ctx:
                text = "\n".join(buffer)
                yield TextSegment(
                    text=text,
                    content_type=ContentType.PROCEDURE_STEP,
                    section_id=ctx.section_id, section_name=ctx.section_name,
                    subsection_type=ctx.subsection_type,
                    page_number=ctx.page_number, line_number=ctx.idx,
                    is_safety_critical=any(
                        kw in text.upper() for kw in ("WARNING", "NOTICE", "CAUTION")
                    ),
                    confidence=1.0,
                )

        for li in line_infos:
            if li.content_type not in (
                ContentType.PROCEDURE_STEP,
                ContentType.WARNING_NOTICE,
                ContentType.GENERAL_TEXT,
            ):
                yield from _flush(); buffer = []; ctx = None; continue

            if li.content_type in (ContentType.PROCEDURE_STEP, ContentType.WARNING_NOTICE):
                if self._STEP.match(li.text) and buffer:
                    yield from _flush(); buffer = []
                buffer.append(li.text)
                if ctx is None:
                    ctx = li
            elif li.content_type == ContentType.GENERAL_TEXT and buffer:
                buffer.append(li.text)

        yield from _flush()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionPipeline:
    """
    Same public interface + output file names as pdf_extractor.py.
    api.py can call this interchangeably.
    """

    def __init__(self, pdf_path: str | Path, out_dir: str | Path = "./output"):
        self.pdf_path = Path(pdf_path)
        self.out_dir  = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.loader       = PDFLoader(self.pdf_path)
        self.tbl_extractor = PDFTableExtractor(self.pdf_path)
        self.cleaner      = TextCleaner()
        self.structure    = StructureParser()
        self.table_p      = GenericTorqueTableParser()
        self.inline_p     = GenericInlineSpecParser()
        self.proc_b       = ProcedureSegmentBuilder()
        self.plumber_conv = PlumberTableConverter()

    def run(self) -> dict:
        log.info("=" * 60)
        log.info("Phase 1 (Generic) — %s", self.pdf_path.name)
        log.info("=" * 60)

        # Stage 1: Load text
        log.info("[1/5] Loading document …")
        raw_text, metadata = self.loader.load()

        # Stage 1b: pdfplumber coordinate tables (parallel pass)
        log.info("[1b]  Extracting coordinate-based tables …")
        plumber_tables = self.tbl_extractor.extract()
        plumber_segs   = self.plumber_conv.convert(plumber_tables)
        log.info("      pdfplumber tables: %d rows across %d tables",
                 len(plumber_segs), len(plumber_tables))

        # Stage 2: Clean
        log.info("[2/5] Cleaning text …")
        clean_text = self.cleaner.clean(raw_text)
        metadata["cleaned_chars"] = len(clean_text)
        log.info("      %d → %d chars (%.1f%% retained)",
                 metadata.get("total_chars", len(raw_text)), len(clean_text),
                 100 * len(clean_text) / max(1, metadata.get("total_chars", len(raw_text))))

        # Stage 3: Parse structure
        log.info("[3/5] Parsing document structure …")
        line_infos = self.structure.parse(clean_text)
        metadata["total_classified_lines"] = len(line_infos)

        # Stage 4: Extract specs
        log.info("[4/5] Extracting specifications …")
        table_segs  = list(self.table_p.extract(line_infos, TableUnitMap()))
        inline_segs = list(self.inline_p.extract(line_infos))
        proc_segs   = list(self.proc_b.build(line_infos))

        # Merge pdfplumber torque rows into table_segs
        plumber_torque = [s for s in plumber_segs
                          if s.content_type == ContentType.TORQUE_TABLE_ROW]
        table_segs = table_segs + plumber_torque

        spec_segments = table_segs + inline_segs
        all_segments  = spec_segments + proc_segs

        deduped_specs = self._deduplicate(spec_segments)

        log.info("      Torque table rows : %d  (+%d from pdfplumber)",
                 len(table_segs) - len(plumber_torque), len(plumber_torque))
        log.info("      Inline torque     : %d", len(inline_segs))
        log.info("      Procedure blocks  : %d", len(proc_segs))
        log.info("      Unique specs      : %d", len(deduped_specs))

        # Fallback: if no specs, use procedure segments or raw paragraphs
        if not deduped_specs:
            log.warning("No structured specs found — using full text segments as fallback")
            deduped_specs = all_segments if all_segments else self._paragraphs(clean_text)
        if not all_segments:
            all_segments = list(deduped_specs)

        metadata.update({
            "torque_table_rows":    len(table_segs),
            "inline_torque_segs":   len(inline_segs),
            "procedure_segments":   len(proc_segs),
            "unique_spec_count":    len(deduped_specs),
            "plumber_table_rows":   len(plumber_torque),
        })

        # Stage 5: Save
        log.info("[5/5] Saving outputs …")
        stem = self.pdf_path.stem.replace(" ", "_")

        paths = {
            "cleaned_text":  self._save_text(clean_text, f"{stem}_cleaned.txt"),
            "all_segments":  self._save_json(
                [s.to_dict() for s in all_segments], f"{stem}_all_segments.json"
            ),
            "spec_segments": self._save_json(
                [s.to_dict() for s in deduped_specs], f"{stem}_spec_segments.json"
            ),
            "metadata":      self._save_json(metadata, f"{stem}_metadata.json"),
        }

        log.info("Done. %d spec segments, %d total.", len(deduped_specs), len(all_segments))
        return {"metadata": metadata, "paths": paths,
                "all_segments": all_segments, "spec_segments": deduped_specs}

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(segs: List[TextSegment]) -> List[TextSegment]:
        seen: dict = {}
        for s in segs:
            nm  = s.spec_value.value_nm if s.spec_value else None
            key = (s.component.lower().strip()[:60], nm, s.section_id)
            if key not in seen:
                seen[key] = s
            elif s.content_type == ContentType.TORQUE_TABLE_ROW:
                seen[key] = s
        return list(seen.values())

    @staticmethod
    def _paragraphs(text: str) -> List[TextSegment]:
        segs = []
        for i, para in enumerate(re.split(r"\n\s*\n", text)):
            para = para.strip()
            if para:
                segs.append(TextSegment(
                    text=para, content_type=ContentType.PROCEDURE_STEP,
                    component="General Text", section_id=f"para_{i}",
                    section_name="Extracted Text",
                ))
        return segs[:200]

    def _save_json(self, data, filename: str) -> str:
        p = self.out_dir / filename
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
        log.info("      → %s (%d KB)", p.name, p.stat().st_size // 1024)
        return str(p)

    def _save_text(self, text: str, filename: str) -> str:
        p = self.out_dir / filename
        p.write_text(text, encoding="utf-8")
        log.info("      → %s (%d KB)", p.name, p.stat().st_size // 1024)
        return str(p)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generic automotive PDF extractor")
    ap.add_argument("pdf", help="Path to the PDF or text file")
    ap.add_argument("--out-dir", default="./output", help="Output directory")
    args = ap.parse_args()

    pipe = ExtractionPipeline(args.pdf, args.out_dir)
    result = pipe.run()
    m = result["metadata"]

    print()
    print("=" * 60)
    print("  PHASE 1 (GENERIC) SUMMARY")
    print("=" * 60)
    print(f"  Source         : {m.get('source_file')}")
    print(f"  Method         : {m.get('extraction_method')}")
    print(f"  Characters     : {m.get('total_chars',0):,} raw → {m.get('cleaned_chars',0):,} cleaned")
    print(f"  Lines          : {m.get('total_classified_lines',0):,}")
    print(f"  Table rows     : {m.get('torque_table_rows',0):,}  "
          f"(+{m.get('plumber_table_rows',0)} from pdfplumber)")
    print(f"  Inline torque  : {m.get('inline_torque_segs',0):,}")
    print(f"  Unique specs   : {m.get('unique_spec_count',0):,}")
    print()

    specs = result["spec_segments"]
    if specs:
        print("  Sample specs:")
        for s in specs[:5]:
            print(f"    [{s.content_type.value}] {s.text[:80]}")
    print()
