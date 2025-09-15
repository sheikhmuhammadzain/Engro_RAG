#!/usr/bin/env python3
"""
xlsx_to_json_pdf.py

Convert every sheet and *every row* of an Excel workbook into:
1) JSONL (one JSON object per row, with metadata) — ideal for RAG ingestion.
2) Chunked TXT files (one chunk per file) — simple text format for embedding.
3) Chunked PDFs (one chunk per page) — readable archive you can embed if desired.

USAGE
-----
python xlsx_to_json_pdf.py --xlsx /path/to/workbook.xlsx --out ./export --max-chars 2000

DEPENDENCIES
------------
pip install pandas openpyxl reportlab

NOTES
-----
- Dates are converted to ISO 8601 strings.
- Each row is converted to a "document" with 'sheet', 'row_index', 'id', 'text', and 'metadata'.
- 'text' consolidates all row fields as "Column: Value" lines.
- Long rows are split into chunks based on --max-chars (default 2000).
- JSONL gets one object per **row** (not per chunk) to keep the canonical record,
  while TXT/PDF are produced **per chunk** for embedding.
"""

import argparse
import json
import re
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Iterable, List

import pandas as pd

# Optional PDF support
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.lib.utils import simpleSplit
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


def iso(v):
    """Convert datelike values to ISO 8601 strings; pass others through as-is."""
    if pd.isna(v):
        return None
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    # Pandas often keeps Excel dates as Timestamp
    if hasattr(v, "to_pydatetime"):
        try:
            return v.to_pydatetime().isoformat()
        except Exception:
            pass
    # Attempt parse for strings that look like dates
    if isinstance(v, str):
        s = v.strip()
        # Cheap check for date-ish strings
        if re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", s):
            try:
                # Let pandas parse broadly
                dt = pd.to_datetime(s, errors="raise")
                return dt.to_pydatetime().isoformat()
            except Exception:
                return s
        return s
    return v


def normalize_col(col: str) -> str:
    """Normalize column names to a stable snake_case key."""
    s = re.sub(r"\s+", " ", str(col or "")).strip()
    s = s.replace("(", "").replace(")", "").replace("#", "number")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "unnamed_column"


def row_to_text(row_dict: Dict[str, Any]) -> str:
    """Create a human-readable 'Field: Value' block for a single row."""
    lines = []
    for k, v in row_dict.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if isinstance(v, (list, dict)):
            v_str = json.dumps(v, ensure_ascii=False)
        else:
            v_str = str(v)
        if v_str.strip() == "":
            continue
        lines.append(f"{k}: {v_str}")
    return "\n".join(lines).strip()


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    Chunk text into roughly max_chars chunks with a small overlap to preserve context.
    Splits on paragraph boundaries first, then falls back to sentences/words.
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paras = text.split("\n")
    chunks = []
    buf = []

    def flush_buf():
        if not buf:
            return
        chunk = "\n".join(buf).strip()
        if chunk:
            chunks.append(chunk)

    cur_len = 0
    for p in paras:
        p = p.strip()
        if not p:
            p = ""  # keep blank lines
        inc = len(p) + 1  # newline
        if cur_len + inc > max_chars and buf:
            # close current chunk with overlap
            flush_buf()
            # add overlap: last ~overlap chars from the previous chunk
            if overlap and chunks:
                tail = chunks[-1][-overlap:]
                buf = [tail] if tail else []
                cur_len = len(tail)
            else:
                buf = []
                cur_len = 0
        else:
            pass
        if p:
            buf = (buf + [p]) if buf else [p]
            cur_len = sum(len(x) + 1 for x in buf)
    flush_buf()
    return chunks


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_txt_chunk(out_dir: Path, doc_id: str, chunk_idx: int, text: str):
    out_path = out_dir / f"{doc_id}__chunk{chunk_idx:04d}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def write_pdf_chunks(out_dir: Path, doc_id: str, chunks: List[str], title: str = "") -> List[Path]:
    """
    Write chunks into a PDF with one chunk per page. Returns list with single PDF path.
    """
    if not REPORTLAB_OK:
        return []
    out_path = out_dir / f"{doc_id}.pdf"
    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    margin = 15 * mm
    usable_w = width - 2 * margin
    usable_h = height - 2 * margin
    text_y_start = height - margin

    # Title page / metadata on first page header
    for i, chunk in enumerate(chunks):
        c.setFont("Helvetica-Bold", 10)
        header = f"{title} — {doc_id} (chunk {i+1}/{len(chunks)})"
        c.drawString(margin, height - margin + 2*mm, header)

        c.setFont("Helvetica", 9)
        lines = simpleSplit(chunk, "Helvetica", 9, usable_w)
        y = text_y_start - 12
        for line in lines:
            if y < margin + 10:
                c.showPage()
                c.setFont("Helvetica-Bold", 10)
                c.drawString(margin, height - margin + 2*mm, header + " (cont.)")
                c.setFont("Helvetica", 9)
                y = text_y_start - 12
            c.drawString(margin, y, line)
            y -= 12
        c.showPage()
    c.save()
    return [out_path]


def process_workbook(xlsx_path: Path, out_dir: Path, max_chars: int = 2000, overlap: int = 200):
    ensure_dir(out_dir)
    jsonl_dir = out_dir / "jsonl"
    txt_dir   = out_dir / "txt_chunks"
    pdf_dir   = out_dir / "pdf_chunks"
    for d in [jsonl_dir, txt_dir, pdf_dir]:
        ensure_dir(d)

    # Read all sheets
    xls = pd.ExcelFile(xlsx_path)
    for sheet_name in xls.sheet_names:
        print(f"Processing sheet: {sheet_name}")
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")

        # Keep original column names for human text; normalized for keys
        orig_cols = list(df.columns)
        norm_cols = [normalize_col(c) for c in orig_cols]
        df.columns = norm_cols

        # Coerce all values to JSON-serializable (dates -> ISO, NaN -> None)
        df = df.where(pd.notnull(df), None)

        # Build JSONL (one object per row) + chunk to TXT/PDF
        jsonl_path = jsonl_dir / f"{sheet_name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as jf:
            for idx, row in df.iterrows():
                row_dict = {k: iso(row[k]) for k in df.columns}
                # Build canonical doc for the row
                doc_id = f"{sheet_name}__row{idx+1:06d}"
                text_block = row_to_text(row_dict)

                doc = {
                    "id": doc_id,
                    "sheet": sheet_name,
                    "row_index": int(idx),
                    "text": text_block,
                    "metadata": {
                        "source_file": str(xlsx_path.name),
                        "sheet": sheet_name,
                        "row_index": int(idx),
                        "columns": df.columns.tolist(),
                    },
                }
                jf.write(json.dumps(doc, ensure_ascii=False) + "\n")

                # Chunk and write TXT/PDF for embedding
                chunks = chunk_text(text_block, max_chars=max_chars, overlap=overlap)
                # TXT chunks
                for ci, ch in enumerate(chunks):
                    write_txt_chunk(txt_dir, doc_id, ci, ch)
                # PDF chunks (one PDF per row; each chunk on a new page)
                if REPORTLAB_OK:
                    write_pdf_chunks(pdf_dir, doc_id, chunks, title=sheet_name)
                else:
                    # If reportlab not available, skip PDF creation
                    pass

        print(f"  Wrote JSONL: {jsonl_path}")
        print(f"  TXT chunks dir: {txt_dir}")
        if REPORTLAB_OK:
            print(f"  PDF chunks dir: {pdf_dir}")
        else:
            print("  Skipped PDF (install 'reportlab' to enable PDF output).")

    print("Done!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to the input .xlsx workbook")
    ap.add_argument("--out", default="./export", help="Output directory (will be created if missing)")
    ap.add_argument("--max-chars", type=int, default=2000, help="Maximum characters per chunk")
    ap.add_argument("--overlap", type=int, default=200, help="Overlap characters between chunks")
    args = ap.parse_args()

    xlsx_path = Path(args.xlsx).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not xlsx_path.exists():
        raise SystemExit(f"Input not found: {xlsx_path}")

    process_workbook(xlsx_path, out_dir, max_chars=args.max_chars, overlap=args.overlap)


if __name__ == "__main__":
    main()
