
import os
import fitz
import numpy as np
import faiss
import shutil
import platform
import subprocess
import tempfile
import time
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import streamlit as st


def find_ollama_path():
    ollama_path = shutil.which("ollama")
    if ollama_path:
        return ollama_path
    if platform.system() == "Windows":
        possible_paths = [
            os.path.expanduser(r"~\\AppData\\Local\\Programs\\Ollama\\ollama.exe"),
            r"C:\\Program Files\\Ollama\\ollama.exe",
            r"C:\\Program Files (x86)\\Ollama\\ollama.exe"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
    return None

OLLAMA_EXEC = find_ollama_path()
if not OLLAMA_EXEC:
    raise FileNotFoundError("❌ Ollama not found. Install from https://ollama.ai/")


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def extract_text_from_pdf(file_path):
    pdf = fitz.open(file_path)
    chunks, metadata = [], []
    for page_num, page in enumerate(pdf):
        text = page.get_text().strip()
        if not text:
            continue
        chunked = chunk_text(text)
        for chunk in chunked:
            chunks.append(chunk)
            metadata.append({
                "page_number": page_num + 1,
                "source": os.path.basename(file_path)
            })
    return chunks, metadata


class HybridRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.metadata = []
        self.bm25 = None

    def build_index(self, all_chunks, all_metadata):
        if not all_chunks:
            raise ValueError("No text chunks to index.")
        self.chunks = all_chunks
        self.metadata = all_metadata
        embeddings = self.embedder.encode(all_chunks).astype('float32')

        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        
        tokenized_chunks = [c.lower().split() for c in all_chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query, top_k=5):
        if not self.index or not self.bm25:
            return []
        query_vec = self.embedder.encode([query]).astype('float32')
        _, semantic_idx = self.index.search(query_vec, top_k)

        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_idx = np.argsort(bm25_scores)[::-1][:top_k]

        final_indices = list(dict.fromkeys(list(semantic_idx[0]) + list(bm25_idx)))
        return [(self.chunks[i], self.metadata[i]) for i in final_indices[:top_k]]


def ollama_chat(prompt, model_name="llama3local"):
    result = subprocess.run(
        [OLLAMA_EXEC, "run", model_name],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8", errors="replace").strip()

def generate_answer(query, retriever, top_k=5):
    results = retriever.search(query, top_k=top_k)
    if not results:
        return "No relevant information found in documents."
    context_text = "\n\n".join([f"[Page {m['page_number']} - {m['source']}]\n{c}" for c, m in results])
    prompt = f"""
You are a helpful AI assistant.
Answer based ONLY on the context below.

Context:
{context_text}

Question:
{query}
"""
    return ollama_chat(prompt)

def summarize_pdf(retriever):
    if not retriever.chunks:
        return "No document content available for summarization."
    all_text = "\n\n".join(retriever.chunks)
    prompt = f"Summarize the following document in bullet points:\n\n{all_text}"
    return ollama_chat(prompt)


def evaluate_retrieval(retriever, test_queries, ground_truth, top_k=5):
    precisions, recalls = [], []
    for i, query in enumerate(test_queries):
        results = retriever.search(query, top_k=top_k)
        retrieved_indices = [retriever.chunks.index(c) for c, _ in results if c in retriever.chunks]
        relevant = ground_truth[i]

        true_positives = len(set(retrieved_indices) & relevant)
        precision = true_positives / top_k if top_k else 0
        recall = true_positives / len(relevant) if relevant else 0

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    return avg_precision, avg_recall


st.set_page_config(
    page_title="InsightForge – Local Hybrid AI for PDF Mastery",
    page_icon="📄",
    layout="wide"
)


st.markdown("""
    <style>
    /* Remove unwanted padding but keep title bar */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }
    header[data-testid="stHeader"] {
        background: white;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    /* Background image */
    .stApp {
        background-image: url("https://collegeinfogeek.com/wp-content/uploads/2018/11/Essential-Books.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-color: rgba(255, 255, 255, 0.85);
        pointer-events: none;
        z-index: -1;
    }
    /* Card styling */
    div[data-testid="stVerticalBlock"] > div {
        background: rgba(255, 255, 255, 0.88);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.15);
    }
    /* Titles */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        color: #003366;
    }
    /* Buttons */
    .stButton>button {
        background-color: #004080;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1em;
        font-weight: 500;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0059b3;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e6f0ff;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #004080;
        color: white !important;
    }
    /* Hide Streamlit menu & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Custom footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 8px;
        background-color: rgba(0, 51, 102, 0.9);
        color: white;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="footer">Made with ❤️ by Tiasha Bhattacharyya</div>', unsafe_allow_html=True)


st.markdown("""
<h1>📄 InsightForge – Local Hybrid AI for PDF Mastery</h1>
<p style="font-size:18px; color:#004080; font-weight:500;">
Fuse semantic intelligence with keyword precision for next-gen document search, summarization, and evaluation – all running locally.
</p>
""", unsafe_allow_html=True)


uploaded_files = st.file_uploader("📂 Upload one or more PDFs", type="pdf", accept_multiple_files=True)
retriever = HybridRetriever()

if uploaded_files:
    all_chunks, all_metadata = [], []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded in uploaded_files:
            file_path = os.path.join(tmp_dir, uploaded.name)
            with open(file_path, "wb") as f:
                f.write(uploaded.getbuffer())
            chunks, metadata = extract_text_from_pdf(file_path)
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)

    if not all_chunks:
        st.error("❌ No text extracted from uploaded PDFs. Please upload valid, readable PDFs.")
    else:
        try:
            retriever.build_index(all_chunks, all_metadata)
            st.success("✅ PDFs indexed successfully!")
        except Exception as e:
            st.error(f"Indexing failed: {e}")

        tab1, tab2, tab3 = st.tabs(["Ask Questions", "Summarize Document", "Evaluate Retrieval"])

        with tab1:
            query = st.text_input("Enter your question:")
            if query:
                with st.spinner("Thinking..."):
                    start = time.time()
                    answer = generate_answer(query, retriever)
                    duration = time.time() - start
                st.markdown(f"### Answer (took {duration:.2f}s):")
                st.write(answer)

        with tab2:
            if st.button("Generate Summary"):
                with st.spinner("Summarizing..."):
                    summary = summarize_pdf(retriever)
                st.markdown("### Summary:")
                st.write(summary)

        with tab3:
            st.markdown("### Retrieval Evaluation")
            st.info("Define test queries and ground truth chunk indices here for evaluation.")
            example_queries = ["What is the method described in chapter 2?", "Summarize key points from introduction."]
            example_ground_truth = [{0, 1}, {2, 3}]

            if st.button("Run Evaluation"):
                p, r = evaluate_retrieval(retriever, example_queries, example_ground_truth)
                st.write(f"Precision@5: {p:.2f}")
                st.write(f"Recall@5: {r:.2f}")
else:
    st.info("📌 Upload PDF files to get started.")
