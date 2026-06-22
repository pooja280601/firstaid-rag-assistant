#!/usr/bin/env python3
"""
build_index.py
Embed chunks.jsonl and build a FAISS index.
Usage:
  python build_index.py --chunks quantitative_eval/data/corpus/processed_text/chunks.jsonl \
                        --index_dir quantitative_eval/data/index \
                        --model all-MiniLM-L6-v2
"""
import argparse, json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks', required=True)
    ap.add_argument('--index_dir', required=True)
    ap.add_argument('--model', default='all-MiniLM-L6-v2')
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Load chunks
    texts, meta = [], []
    with open(args.chunks, 'r') as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj['text'])
            meta.append({k: obj[k] for k in obj if k != 'text'})

    # Embed
    model = SentenceTransformer(args.model)
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    # FAISS index (cosine via inner product on normalized vectors)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # Save index + meta
    faiss.write_index(index, str(index_dir / 'faiss.index'))
    with open(index_dir / 'meta.jsonl', 'w') as w:
        for m in meta:
            w.write(json.dumps(m) + "\n")

    print(f"Indexed {len(texts)} chunks to {index_dir}")

if __name__ == "__main__":
    main()
