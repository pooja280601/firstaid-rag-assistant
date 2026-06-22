# app.py
import os, json, re, requests
from pathlib import Path
import numpy as np
import streamlit as st
import faiss
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

FORBIDDEN_RE = re.compile(r"(prescrib(e|ing|er|ption)|antibiotic(s)?|dose|dosage|mg|diagnos(e|is|ing)|medication(s)?|drug(s)?|rx)", re.I)
PAGE_RE = re.compile(r"<<PAGE\s+(\d+)>>")

# ---------- loaders ----------
@st.cache_resource
def load_index(index_dir: str):
    p = Path(index_dir)
    index = faiss.read_index(str(p / "faiss.index"))
    meta = [json.loads(x) for x in (p / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
    return index, meta

@st.cache_resource
def load_chunks(chunks_path: str):
    return [json.loads(x) for x in Path(chunks_path).read_text(encoding="utf-8").splitlines()]

@st.cache_resource
def load_embed(model_name: str):
    return SentenceTransformer(model_name)

# ---------- retrieval helpers ----------
def first_page(text: str):
    m = PAGE_RE.search(text or "")
    return int(m.group(1)) if m else None

def build_context(hit_ids, meta, chunks, max_chars=7000, max_per_source=2):
    parts, per_source, total = [], {}, 0
    for i in hit_ids:
        if i < 0 or i >= len(chunks) or i >= len(meta):
            continue
        m, c = meta[i], chunks[i]
        key = (m.get("title"), m.get("source_file"))
        if per_source.get(key, 0) >= max_per_source:
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        piece = f"[Source: {m.get('title')} - {m.get('source_file')}]\n{text}"
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        per_source[key] = per_source.get(key, 0) + 1
        total += len(piece)
    return "\n\n-----\n\n".join(parts)

def build_citations(hit_ids, meta, chunks):
    by_source = {}
    for i in hit_ids:
        if i < 0 or i >= len(chunks) or i >= len(meta):
            continue
        m, c = meta[i], chunks[i]
        key = (m.get("title") or "Unknown", m.get("source_file") or "?")
        pg = first_page(c.get("text") or "")
        if key not in by_source or (pg is not None and (by_source[key] is None or pg < by_source[key])):
            by_source[key] = pg
    cites = []
    for (title, src), pg in by_source.items():
        label = f"{title}" + (f" - Page {pg}" if pg else "") + f" ({src})"
        cites.append(label)
    return cites

# ---------- LLM calls ----------
def ask_openai(system, question, context):
    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY not set."
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

# ---------- UI ----------
st.set_page_config(page_title="First Aid Assistant", layout="centered")

# Header with small logo + title (no background image)
logo_path = Path("assets/logo.png")
col_logo, col_title = st.columns([1, 8], vertical_alignment="center")
with col_logo:
    if logo_path.exists():
        st.image(str(logo_path), width=64)
with col_title:
    st.title("First Aid Assistant")

with st.sidebar:
    st.subheader("Settings")
    index_dir = st.text_input("Index directory", "data/index")
    chunks_path = st.text_input("Chunks path", "data/corpus/processed_text/chunks.jsonl")
    embed_model = st.text_input("Embedding model", "all-MiniLM-L6-v2")
    k = st.slider("Top-K retrieval", 3, 15, 10)
    min_sim = st.slider("Min similarity gate", 0.0, 1.0, 0.20, 0.01)
    llm_mode = st.selectbox("Answer mode", ["Ollama", "OpenAI", "Retrieval only"], index=0)
    ollama_model = st.text_input("Ollama model", "llama3.2:3b")
    show_ctx = st.checkbox("Show retrieved context", value=False)
    st.caption("For OpenAI, set OPENAI_API_KEY. For Ollama, run 'ollama serve' and pull a model.")

# Load resources (cached)
index, meta = load_index(index_dir)
chunks = load_chunks(chunks_path)
embed = load_embed(embed_model)

# Form with Submit button
with st.form("qa_form", clear_on_submit=False):
    q = st.text_input("Ask a first-aid question (e.g., 'How to give CPR?')", key="question")
    submitted = st.form_submit_button("Submit")

if submitted and q:
    # Guardrail for out-of-scope asks
    if FORBIDDEN_RE.search(q):
        st.warning("I can’t help with prescriptions, dosages, medications, or diagnoses. For medical advice, contact a clinician or local health service.")
    else:
        # Retrieve
        qv = embed.encode([q], normalize_embeddings=True).astype(np.float32)
        D, I = index.search(qv, k)
        top_sim = float(D[0][0]) if D is not None and D.shape[1] > 0 else 0.0
        if top_sim < min_sim:
            st.info("I’m not confident the manuals cover this. Please rephrase with a first-aid topic such as CPR, choking, burns, or bleeding.")
        else:
            hit_ids = [i for i in I[0].tolist() if i >= 0]
            ctx = build_context(hit_ids, meta, chunks, max_per_source=2)
            cites = build_citations(hit_ids, meta, chunks)

            # Answer
            if llm_mode == "Ollama":
                answer = ask_ollama(SYSTEM_PROMPT, q, ctx, model_name=ollama_model)
                st.subheader("Answer")
                st.markdown(answer)
            elif llm_mode == "OpenAI":
                answer = ask_openai(SYSTEM_PROMPT, q, ctx)
                st.subheader("Answer")
                st.markdown(answer)
            else:
                st.subheader("Guidance (retrieved)")
                st.code(ctx, language="markdown")

            # Optional: show raw context
            if show_ctx and llm_mode != "Retrieval only":
                with st.expander("Show retrieved context"):
                    st.code(ctx, language="markdown")

            # Sources
            if cites:
                st.subheader("Sources")
                for c in cites:
                    st.write("• " + c)

            st.caption("This guidance does not replace professional medical advice. Call emergency services for life-threatening situations.")
