#!/usr/bin/env python3
"""
run_eval_first_aid.py  — Option B layout

Evaluates retrieval with hit@1 / hit@3 given queries.csv:
  id,question,expected_doc_id

Outputs:
  Evaluation_results/per_query_results.csv
  Evaluation_results/summary.json

Usage:
  python scripts/run_eval_first_aid.py \
    --index_dir data/index \
    --queries data/queries.csv \
    --emb all-MiniLM-L6-v2 \
    --k 5
"""
import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_index(index_dir: Path):
    idx_path = index_dir / "faiss.index"
    meta_path = index_dir / "meta.jsonl"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Index files not found in {index_dir}. Expected faiss.index and meta.jsonl")
    index = faiss.read_index(str(idx_path))
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))
    return index, meta


def load_queries(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Queries CSV not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            q = (row.get("question") or "").strip()
            if not q:
                continue
            rows.append({
                "id": (row.get("id") or "").strip(),
                "question": q,
                "expected_doc_id": (row.get("expected_doc_id") or "").strip().upper(),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="data/index")
    ap.add_argument("--queries", default="data/queries.csv")
    ap.add_argument("--emb", default="all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out_dir", default="Evaluation_results")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    queries_path = Path(args.queries)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load index + meta
    index, meta = load_index(index_dir)
    doc_ids = np.array([(m.get("doc_id") or "").upper() for m in meta])
    indexed_set = set(doc_ids.tolist())

    # Load and filter queries to only those present in the index (you have 3 PDFs)
    queries = load_queries(queries_path)
    filt_queries = [q for q in queries if q["expected_doc_id"] in indexed_set]
    skipped = len(queries) - len(filt_queries)
    if not filt_queries:
        raise RuntimeError("No queries match the current index doc_ids. Check expected_doc_id values.")

    # Encode all queries in a batch
    model = SentenceTransformer(args.emb)
    q_texts = [q["question"] for q in filt_queries]
    q_emb = model.encode(q_texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    # Search
    k = max(1, args.k)
    D, I = index.search(q_emb, k)

    # Per-query CSV
    per_rows = []
    top1 = top3 = 0
    for qi, q in enumerate(filt_queries):
        neighbors = I[qi].tolist()
        neighbor_doc_ids = [doc_ids[i] if 0 <= i < len(doc_ids) else "" for i in neighbors]
        exp = q["expected_doc_id"]

        hit1 = int(neighbor_doc_ids[:1] and neighbor_doc_ids[0] == exp)
        hit3 = int(exp in neighbor_doc_ids[:3])
        top1 += hit1
        top3 += hit3

        per_rows.append({
            "id": q["id"],
            "question": q["question"],
            "expected_doc_id": exp,
            "rank_1_doc_id": neighbor_doc_ids[0] if neighbor_doc_ids else "",
            "hit_at_1": hit1,
            "hit_at_3": hit3,
            "neighbors_doc_ids": "|".join(neighbor_doc_ids),
        })

    # Save per_query_results.csv
    per_path = out_dir / "per_query_results.csv"
    with open(per_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question",
                "expected_doc_id",
                "rank_1_doc_id",
                "hit_at_1",
                "hit_at_3",
                "neighbors_doc_ids",
            ],
        )
        w.writeheader()
        for r in per_rows:
            w.writerow(r)

    # Summary
    n = len(filt_queries)
    summary = {
        "n_total_queries": len(queries),
        "n_evaluated": n,
        "n_skipped_not_in_index": skipped,
        "top1_hit_pct": round(100.0 * top1 / n, 1) if n else 0.0,
        "top3_hit_pct": round(100.0 * top3 / n, 1) if n else 0.0,
    }
    sum_path = out_dir / "summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[summary]", summary)
    print(f"[save] {per_path}")
    print(f"[save] {sum_path}")


if __name__ == "__main__":
    main()
