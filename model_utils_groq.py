import os
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import faiss
from openai import OpenAI  # GROQ ONLY
import re

# Cache version to force reindex when chunking changes
CACHE_VERSION = "2"

# -------------------------------
# GROQ CONFIG - NO GEMINI SHIT
# -------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"  # 🏆 BEST FOR HSE

# -------------------------------
# Step 1: Initialize models
# -------------------------------
print("🔹 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Optional: cross-encoder for reranking
try:
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("✅ Cross-encoder loaded.")
except Exception as e:
    cross_encoder = None
    print("⚠️ No cross-encoder.")


# -------------------------------
# GROQ HELPER - PURE & SIMPLE
# -------------------------------
def generate_with_groq(prompt: str):
    """GROQ ONLY - FAST AS FUCK FOR HSE"""
    try:
        key = os.environ.get('groq_api')
        if not key:
            return "SET GROQ_API_KEY YOU DUMBASS!"

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=key,
        )

        # Build messages once for reuse in both attempts
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are an EXPERT Occupational Safety & Health (OSH/HSE) assistant. "
                    "Answer ONLY using the provided Sources. Interpret the user's request from an HSE perspective. "
                    "Formalize the answer in professional HSE format without altering its factual meaning. "
                    "Begin with a brief one-line title that clearly states the topic. "
                    "Respond with concise bullet points (no inline citations). "
                    "End with a short closing sentence that reinforces the scope or compliance. "
                    "Do not include any 'References:' lines in the answer; sources are shown separately. "
                    "If the answer is not in the Sources, reply exactly 'I don't know.' "
                    "Never invent or speculate. Keep structure tidy; no preambles or conclusions aside from the title and closing sentence."
                )
            },
            {"role": "user", "content": prompt}
        ]

        # Try with seed for determinism; fallback if the provider rejects it
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                presence_penalty=0,
                frequency_penalty=0,
                max_tokens=3072,
                seed=42,
            )
        except TypeError:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                presence_penalty=0,
                frequency_penalty=0,
                max_tokens=3072,
            )

        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"⚠️ GROQ ERROR: {e}")
        return "GROQ FAILED - CHECK YOUR KEY!"


# -------------------------------
# Step 2: Chunking & Books (UNCHANGED)
# -------------------------------
def chunk_text(text, max_chars=1200, overlap=200):
    if not text: return []
    text = " ".join(text.split())
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n: break
        start = max(0, end - overlap)
    return chunks

BASE_DIR = Path(__file__).resolve().parent
BOOKS_PATH = str(BASE_DIR / "books")
CACHE_DIR = BASE_DIR / "cache"
DOCS_JSON = CACHE_DIR / "documents.json"
META_JSON = CACHE_DIR / "metadata.json"
FP_JSON = CACHE_DIR / "fingerprint.json"
INDEX_PATH = CACHE_DIR / "faiss.index"

def compute_fingerprint(books_path: str):
    fp = {"cache_version": CACHE_VERSION}
    for book_file in os.listdir(books_path):
        path = os.path.join(books_path, book_file)
        if os.path.isfile(path) and (book_file.endswith(".pdf") or book_file.endswith(".txt")):
            fp[book_file] = {"size": os.path.getsize(path), "mtime": os.path.getmtime(path)}
    return fp

def load_cache_if_valid(expected_fp):
    if not (DOCS_JSON.exists() and META_JSON.exists() and FP_JSON.exists() and INDEX_PATH.exists()):
        raise FileNotFoundError("Cache files missing")
    with open(FP_JSON, "r", encoding="utf-8") as f: cached_fp = json.load(f)
    if cached_fp != expected_fp: raise ValueError("Books changed")
    with open(DOCS_JSON, "r", encoding="utf-8") as f: docs = json.load(f)
    with open(META_JSON, "r", encoding="utf-8") as f: meta = json.load(f)
    idx = faiss.read_index(str(INDEX_PATH))
    return docs, meta, idx

def save_cache(docs, meta, fp, idx):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DOCS_JSON, "w", encoding="utf-8") as f: json.dump(docs, f, ensure_ascii=False)
    with open(META_JSON, "w", encoding="utf-8") as f: json.dump(meta, f, ensure_ascii=False)
    with open(FP_JSON, "w", encoding="utf-8") as f: json.dump(fp, f, ensure_ascii=False)
    faiss.write_index(idx, str(INDEX_PATH))

books_fp = compute_fingerprint(BOOKS_PATH)
try:
    documents, metadata, index = load_cache_if_valid(books_fp)
    print(f"✅ Loaded {index.ntotal} entries.")
except Exception as e:
    print("⚠️ Reprocessing books.")
    documents, metadata = [], []
    for book_file in os.listdir(BOOKS_PATH):
        path = os.path.join(BOOKS_PATH, book_file)
        if book_file.endswith(".txt"):
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        documents.append(line.strip())
                        metadata.append((book_file, f"Line {i+1}"))
        elif book_file.endswith(".pdf"):
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        for ci, chunk in enumerate(chunk_text(text)):
                            documents.append(chunk)
                            metadata.append((book_file, f"Page {page_num+1}, Chunk {ci+1}"))

    print(f"✅ Loaded {len(documents)} chunks.")
    embeddings = embedding_model.encode(documents, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"✅ FAISS built with {index.ntotal} entries.")
    save_cache(documents, metadata, books_fp, index)


# -------------------------------
# Step 3: Chatbot (GROQ ONLY)
# -------------------------------
def compress_references(refs):
    seen, out = set(), []
    for book, loc in refs:
        page = loc.split(",")[0].strip()
        key = (book, page)
        if key not in seen:
            seen.add(key)
            out.append(f"{book} ({page})")
    return out

def is_osh_query(text: str) -> bool:
    t = (text or "").lower()
    keywords = [
        "osh", "occupational safety", "occupational health", "hse", "workplace safety",
        "osh statistics", "national statistics", "data collection", "monitoring", "compliance",
        "regulation", "legislation", "employer", "worker", "hazard", "risk", "incident",
        "accident", "inspection"
    ]
    return any(k in t for k in keywords)

def get_book_answer(user_query, top_k=50, final_k=5):
    if not user_query or not documents: return "No info.", []
    is_osh = is_osh_query(user_query)
    if not is_osh:
        return "I don't know.", []
    query_text = expand_query(user_query)
    query_emb = embedding_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(query_emb, top_k)
    candidate_idxs = [idx for idx in indices[0] if idx < len(documents)]
    if not candidate_idxs: return "I don't know.", []
    # Gate irrelevant questions: if similarity is low and query isn't OSH-related, or top doc lacks key tokens, return I don't know
    try:
        best_sim = float(np.max(distances[0])) if len(distances[0]) else 0.0
    except Exception:
        best_sim = 0.0
    SIM_THRESHOLD = 0.55
    top_doc = documents[candidate_idxs[0]].lower() if candidate_idxs else ""
    tokens = re.findall(r"[a-zA-Z]+", user_query.lower())
    key_tokens = [t for t in tokens if len(t) > 3]
    has_overlap = any(t in top_doc for t in key_tokens)
    if not is_osh and (best_sim < SIM_THRESHOLD or not has_overlap):
        return "I don't know.", []

    if cross_encoder:
        pairs = [(query_text, documents[i]) for i in candidate_idxs]
        scores = list(cross_encoder.predict(pairs))
        top_idxs = [candidate_idxs[i] for i in sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)][:final_k]
    else:
        top_idxs = candidate_idxs[:final_k]

    retrieved_chunks = [documents[i] for i in top_idxs]
    refs = [metadata[i] for i in top_idxs]
    context = "\n\n".join([f"[Source {i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks)])

    prompt = f"""
You are answering strictly from the Sources below.
- Use professional HSE tone and formal formatting.
- Maintain original meaning; do not alter semantics.
- Begin with a short one-line title summarizing the question.
- Put the title on its own line above the bullets.
- Respond in concise bullet points; limit to 5–7 bullets.
- Each bullet must reflect a single explicit requirement.
- Focus ONLY on explicit requirements for OSH data collection and national statistics.
- Exclude generic OSH governance items unless they directly specify data collection/statistics obligations.
- Avoid excessive density or repetition; streamline points into shorter sentences for readability without losing detail.
- Do not include quoted phrases, parentheses, or inline citations inside bullets.
- Use hyphen-leading bullets ('- ') and put each bullet on a new line; do not use '*' for bullets.
- Place the final closing sentence on its own line at the end.
- Do not include any 'References' line; sources are handled by UI.
- Do not use any external information.
- If the Sources do not contain the answer, reply exactly: I don't know.

Sources:
{context}

Question: {user_query}

Answer:
"""

    
    def strip_reference_lines(text: str) -> str:
        if not text:
            return text
        lines = text.splitlines()
        kept = [ln for ln in lines if not re.match(r"\s*References\s*:\s*", ln, flags=re.IGNORECASE)]
        return "\n".join(kept).strip()
    response = generate_with_groq(prompt)  # GROQ ONLY!
    clean_response = strip_reference_lines(response)
    references = compress_references(refs)
    if clean_response.strip().lower().startswith("i don't know"):
        references = []
    return clean_response, references


# -------------------------------
# RUN IT
# -------------------------------
if __name__ == "__main__":
    print(f"✅ HSE BOT - GROQ ONLY! Model: {GROQ_MODEL}")
    print("Set GROQ_API_KEY and type 'exit' to quit.\n")
    
    while True:
        q = input("👤 You: ")
        if q.lower() in ["exit", "quit"]: break
        ans, refs = get_book_answer(q)
        print(f"\n🤖 Bot: {ans}")
        if refs: print("📚 References:", ", ".join(refs))
        print("-" * 50)

def expand_query(user_query: str):
    boosters = [
        "OSH data collection", "national OSH statistics", "statistical indicators",
        "data reporting", "data aggregation", "monitoring", "registry", "database",
        "shall", "must", "require", "policy requires"
    ]
    if not is_osh_query(user_query):
        return user_query
    return f"{user_query}\nKey terms: " + ", ".join(boosters)