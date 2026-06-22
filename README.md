# First Aid RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers first aid questions using
trusted Australian first aid manuals. Built with FAISS vector search, sentence
transformers, and a Streamlit UI — supports both local (Ollama) and cloud (OpenAI) LLMs.

## How it works

```
PDF documents
    → prep_first_aid.py     (extract + chunk text)
    → build_index.py        (embed chunks → FAISS index)
    → app.py                (retrieve relevant chunks → LLM → cited answer)
    → run_eval_first_aid.py (quantitative retrieval evaluation: hit@1, hit@3)
```

**Retrieval:** Queries are embedded with `all-MiniLM-L6-v2` and matched against a FAISS
flat inner-product index (cosine similarity on normalised vectors). Top-K chunks are
assembled into a context window, capped per source to avoid over-weighting any single document.

**Safety guardrails:**
- Regex filter blocks out-of-scope questions (prescriptions, medications, diagnoses)
- Similarity gate rejects low-confidence retrievals rather than hallucinating an answer
- System prompt restricts the LLM to cited source material only; forces 000 referral for life-threatening situations
- Page-level citations traced back to source PDFs

**Evaluation:** `run_eval_first_aid.py` reports hit@1 and hit@3 retrieval accuracy on a
test query set. Current results: **60% hit@1, 60% hit@3** on 5 queries (`Evaluation_results/summary.json`).

## Source documents

The index was built from three publicly available Australian first aid documents.
Two of the PDFs (Red Cross and St John) are copyrighted and are not included in this
repository — download them from the official sources below and place them in
`data/corpus/raw_pdfs/` before re-indexing.

| Document | Publisher | License | Download |
|---|---|---|---|
| Essential First Aid Guide | Australian Red Cross | © Australian Red Cross — free public guide | [redcross.org.au](https://www.redcross.org.au/globalassets/cms/first-aid/first-aid-pdfs/first-aid-essentials/red-cross-essential-first-aid-guide-english.pdf) |
| First Aid Quick Reference Guide | St John Ambulance Australia | © St John Ambulance Australia — personal use only; redistribution requires permission | [shop.stjohn.org.au](https://shop.stjohn.org.au/products/first-aid-quick-reference) |
| First Aid in the Workplace (Code of Practice) | Safe Work Australia | [CC BY-NC 3.0 AU](https://creativecommons.org/licenses/by-nc/3.0/au/) | [safeworkaustralia.gov.au](https://www.safeworkaustralia.gov.au/doc/model-code-practice-first-aid-workplace) |

The Safe Work Australia PDF is included in this repo under its Creative Commons
Attribution-Noncommercial 3.0 Australia licence. The pre-built `chunks.jsonl`, FAISS index,
and metadata cover all three documents so the app runs immediately without re-indexing.

## Tech stack

Python · FAISS · sentence-transformers · Streamlit · OpenAI API (optional) · Ollama (optional)

## Setup

```bash
# 1. Install dependencies
pip install streamlit faiss-cpu sentence-transformers openai requests pypdf numpy

# 2. Run the app (uses pre-built index — no re-indexing needed)
streamlit run app.py

# 3. In the sidebar, choose Answer mode:
#    - Ollama (local): run `ollama serve` and pull a model, e.g. `ollama pull llama3.2:3b`
#    - OpenAI: set OPENAI_API_KEY environment variable
#    - Retrieval only: shows raw retrieved chunks, no LLM needed
```

## Re-indexing from scratch (optional)

If you add new PDFs or want to rebuild the index:

```bash
# 1. Place PDFs in data/corpus/raw_pdfs/
# 2. Update data/corpus/metadata.csv with file_name, doc_id, title for each PDF

# 3. Chunk the PDFs
python scripts/prep_first_aid.py \
  --in data/corpus/raw_pdfs \
  --out data/corpus/processed_text \
  --meta data/corpus/metadata.csv \
  --chunk_words 700 --overlap 120

# 4. Build the FAISS index
python scripts/build_index.py \
  --chunks data/corpus/processed_text/chunks.jsonl \
  --index_dir data/index \
  --model all-MiniLM-L6-v2

# 5. Run the retrieval evaluation
python scripts/run_eval_first_aid.py \
  --index_dir data/index \
  --queries data/queries.csv \
  --emb all-MiniLM-L6-v2 \
  --k 5
```

## CLI demo (no Streamlit)

```bash
python scripts/demo_first_aid_v2.py --llm ollama --ollama_model llama3.2:3b
```

## Disclaimer

This tool is for educational and portfolio purposes. It does not constitute medical advice.
Always call emergency services (000 in Australia) for life-threatening situations.
