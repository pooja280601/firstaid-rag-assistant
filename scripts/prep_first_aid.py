#!/usr/bin/env python3
"""
prep_first_aid.py
Convert PDFs in --in to chunked JSONL in --out for RAG indexing.

Usage (Option B layout):
  python scripts/prep_first_aid.py --in data/corpus/raw_pdfs \
                                   --out data/corpus/processed_text \
                                   --meta data/corpus/metadata.csv \
                                   --chunk_words 700 --overlap 120
"""
import argparse
import csv
import json
import re
from pathlib import Path
from pypdf import PdfReader


def clean_text(t: str) -> str:
    """Basic cleanup: nulls, multi-spaces, excessive newlines, trim."""
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def chunk_words(text: str, chunk_words: int = 700, overlap: int = 120):
    """Word-based chunking with overlap."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(n, start + chunk_words)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def load_meta(meta_csv: Path):
    """Load metadata CSV (file_name, doc_id, title) into a dict keyed by file_name."""
    meta = {}
    if meta_csv and meta_csv.exists():
        with open(meta_csv, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                meta[row["file_name"]] = row
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Directory of source PDFs")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output directory for chunks.jsonl")
    ap.add_argument("--meta", dest="meta_csv", required=False, default=None, help="Optional metadata CSV")
    ap.add_argument("--chunk_words", type=int, default=700)
    ap.add_argument("--overlap", type=int, default=120)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(Path(args.meta_csv)) if args.meta_csv else {}

    out_jsonl = out_dir / "chunks.jsonl"
    count = 0

    with open(out_jsonl, "w", encoding="utf-8") as w:
        for pdf_path in sorted(in_dir.glob("*.pdf")):
            try:
                reader = PdfReader(str(pdf_path))
            except Exception as e:
                print(f"[ERROR] Failed to open {pdf_path.name}: {e}")
                continue

            all_pages = []
            for i, page in enumerate(reader.pages):
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                txt = clean_text(txt)
                if txt:
                    all_pages.append((i + 1, txt))

            # Warn if nothing extracted (likely scanned PDF → needs OCR)
            if not all_pages:
                print(
                    f"[WARN] No extractable text in {pdf_path.name}. "
                    "It may be a scanned PDF; consider OCR."
                )
                continue

            # Combine pages, preserving page markers for later citation
            combined = []
            for pg, txt in all_pages:
                combined.append(f"<<PAGE {pg}>> {txt}")
            combined_text = "\n".join(combined)

            chunks = chunk_words(combined_text, args.chunk_words, args.overlap)
            file_name = pdf_path.name
            m = meta.get(file_name, {})
            # Normalize doc_id to uppercase for consistent eval comparisons
            doc_id = m.get("doc_id", file_name.rsplit(".", 1)[0]).strip().upper()
            title = m.get("title", doc_id.replace("_", " "))

            for idx, ch in enumerate(chunks):
                record = {
                    "chunk_id": f"{doc_id}_{idx:04d}",
                    "doc_id": doc_id,
                    "title": title,
                    "source_file": file_name,
                    "text": ch,
                }
                w.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote {count} chunks to {out_jsonl}")


if __name__ == "__main__":
    main()
