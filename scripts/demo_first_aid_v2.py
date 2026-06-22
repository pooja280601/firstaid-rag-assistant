#!/usr/bin/env python3
import os, json, argparse, textwrap, requests, re, sys
from pathlib import Path
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

SYSTEM_PROMPT = """You are "First Aid Assistant". Use ONLY the provided context from trusted first-aid manuals.
If the context does not contain the answer, say you don't know.

Respond in concise, numbered steps (Markdown list), suitable for a non-expert.
- Keep each step to one short, imperative sentence.
- 6–10 steps max.
- If there are key cautions, add a brief Notes section as bullet points (up to 3 bullets).
- Do not add long paragraphs or extra fluff.
- Always end with a one-line Safety Disclaimer.
When a topic has multiple severities (e.g., minor/superficial vs severe/full-thickness), answer for the asked severity and clearly differentiate if both are present in the context.

Safety rules:
- No diagnosis or drug advice.
- If the situation is life-threatening or unclear, instruct to call emergency services immediately.
- Do not infer beyond the cited manuals.
"""

# Out-of-scope / safety filters
FORBIDDEN_PATTERNS = [
    r"\bprescrib(e|ing|er|ption)\b",
    r"\bantibiotic(s)?\b",
    r"\bdose|dosage|mg\b",
    r"\bdiagnos(e|is|ing)\b",
    r"\bmedication(s)?\b",
    r"\bdrug(s)?\b",
    r"\brx\b",
]
FORBIDDEN_RE = re.compile("|".join(FORBIDDEN_PATTERNS), flags=re.I)

# Page tag for citations
PAGE_RE = re.compile(r"<<PAGE\s+(\d+)>>")

def load_index(index_dir: Path):
    idx_path = index_dir / "faiss.index"
    meta_path = index_dir / "meta.jsonl"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Index files not found in {index_dir}. Expected faiss.index and meta.jsonl")
    index = faiss.read_index(str(idx_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = [json.loads(x) for x in f if x.strip()]
    return index, meta

def load_chunks(chunks_path: Path):
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def first_page_from_text(txt: str):
    m = PAGE_RE.search(txt or "")
    return int(m.group(1)) if m else None

def build_context(hit_ids, meta, chunks, max_chars=7000, max_per_source=2):
    """Assemble context allowing multiple chunks per source (up to max_per_source)."""
    parts = []
    per_source = {}  # (title, source_file) -> count
    total_len = 0

    for i in hit_ids:
        if i is None or i < 0:
            continue
        if i >= len(chunks) or i >= len(meta):
            continue
        m, c = meta[i], chunks[i]
        key = (m.get("title"), m.get("source_file"))
        cnt = per_source.get(key, 0)
        if cnt >= max_per_source:
            continue

        text = (c.get("text") or "").strip()
        if not text:
            continue

        piece = f"[Source: {m.get('title')} - {m.get('source_file')}]\n{text}"
        new_total = total_len + len(piece)
        if new_total > max_chars:
            break

        parts.append(piece)
        per_source[key] = cnt + 1
        total_len = new_total

    return "\n\n-----\n\n".join(parts)

def build_citations(hit_ids, meta, chunks):
    """One citation per (title, source_file); show the lowest page number seen."""
    by_source = {}
    for i in hit_ids:
        if i is None or i < 0 or i >= len(chunks) or i >= len(meta):
            continue
        m, c = meta[i], chunks[i]
        title = m.get("title") or "Unknown"
        src = m.get("source_file") or "?"
        key = (title, src)

        txt = c.get("text") or ""
        mpage = PAGE_RE.search(txt)
        pg = int(mpage.group(1)) if mpage else None

        if key not in by_source:
            by_source[key] = pg
        else:
            if pg is not None and (by_source[key] is None or pg < by_source[key]):
                by_source[key] = pg

    cites = []
    for (title, src), pg in by_source.items():
        label = f"{title}" + (f" - Page {pg}" if pg else "") + f" ({src})"
        cites.append(label)
    return cites

def ask_openai(system, question, context):
    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY not set. Use ollama mode or set your key."
    from openai import OpenAI
    client = OpenAI()
    prompt = (
        "Use the context to answer the question strictly in numbered steps.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    rsp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system},
                  {"role":"user","content":prompt}],
        temperature=0.2,
    )
    return (rsp.choices[0].message.content or "").strip()

def ask_ollama(system, question, context, model_name="llama3.2:3b"):
    prompt = (
        f"{system}\n\n"
        "Use the context to answer the question strictly in numbered steps.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
        timeout=180,
    )
    r.raise_for_status()
    return (r.json() or {}).get("response", "").strip()

def is_forbidden_question(q: str) -> bool:
    return bool(FORBIDDEN_RE.search(q or ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="data/index")
    ap.add_argument("--chunks", default="data/corpus/processed_text/chunks.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--emb", default="all-MiniLM-L6-v2")
    ap.add_argument("--llm", choices=["ollama","openai","none"], default="none")
    ap.add_argument("--ollama_model", default="llama3.2:3b")
    ap.add_argument("--min_sim", type=float, default=0.20, help="min cosine similarity (approx via inner product) to accept context")
    args = ap.parse_args()

    index_dir, chunks_path = Path(args.index_dir), Path(args.chunks)
    try:
        index, meta = load_index(index_dir)
        chunks = load_chunks(chunks_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if len(meta) != len(chunks):
        print(f"[WARN] meta.jsonl ({len(meta)}) and chunks.jsonl ({len(chunks)}) differ in length. "
              "Ensure both were built from the same preprocessing run.")

    try:
        embed = SentenceTransformer(args.emb)
    except Exception as e:
        print(f"[ERROR] Failed to load embedding model '{args.emb}': {e}")
        sys.exit(1)

    print("Type a question (or 'exit'):")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        # 1) Out-of-scope filter (no LLM call)
        if is_forbidden_question(q):
            print("\n" + "=" * 80)
            print("Question:", q)
            print("\nAnswer:\n  I can’t help with prescriptions, dosages, medications, or diagnoses.")
            print("  For medical advice, please see a qualified clinician or call local health services.")
            print("=" * 80 + "\n")
            continue

        # 2) Retrieve nearest chunks
        qv = embed.encode([q], normalize_embeddings=True).astype(np.float32)
        D, I = index.search(qv, args.k)
        hit_ids = [i if i >= 0 else None for i in I[0].tolist()]

        # Low-confidence gating
        top_sim = float(D[0][0]) if D is not None and D.shape[1] > 0 else 0.0
        if top_sim < args.min_sim:
            print("\n" + "=" * 80)
            print("Question:", q)
            print("\nAnswer:\n  I’m not confident the manuals I have cover this. "
                  "Please rephrase with a first-aid topic such as CPR, choking, burns, or bleeding.")
            print("=" * 80 + "\n")
            continue

        context = build_context(hit_ids, meta, chunks, max_chars=7000, max_per_source=2)
        if not context.strip():
            print("\n" + "=" * 80)
            print("Question:", q)
            print("\nAnswer:\n  No relevant context retrieved. Try rephrasing your question.")
            print("=" * 80 + "\n")
            continue

        citations = build_citations(hit_ids, meta, chunks)

        # 3) Generate answer
        if args.llm == "ollama":
            answer = ask_ollama(SYSTEM_PROMPT, q, context, model_name=args.ollama_model)
        elif args.llm == "openai":
            answer = ask_openai(SYSTEM_PROMPT, q, context)
        else:
            answer = "[Retrieval-only preview]\n\n" + context

        # 4) Print nicely
        print("\n" + "=" * 80)
        print("Question:", q)
        print("\nAnswer:\n", textwrap.dedent(answer).strip())
        if citations and args.llm != "none":
            print("\nSources:", "; ".join(citations))
        if args.llm == "none":
            print("Disclaimer: This guidance does not replace professional medical advice. "
                  "Call emergency services for life-threatening situations.")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
