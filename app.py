import streamlit as st
import tempfile
import os
import re

from ingestion.loader import load_documents
from chunking.chunker import chunk_text
from embeddings.embedder import TextEmbedder
from vector_store.faiss_index import FaissVectorStore
from generation.answer_generator import AnswerGenerator
from utils.confidence import compute_confidence
from reranker.cross_encoder import CrossEncoderReranker


# =========================================================
# HELPER: KEYWORD OVERLAP (ANTI SEMANTIC-DRIFT)
# =========================================================
def keyword_overlap(question, retrieved_chunks, min_overlap=1):
    question_keywords = set(
        re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
    )
    if not question_keywords:
        return False

    combined_text = " ".join(
        chunk["text"].lower() for chunk in retrieved_chunks
    )

    overlap = [kw for kw in question_keywords if kw in combined_text]
    return len(overlap) >= min_overlap


# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Document Intelligence System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. GLASSMORPHISM UI
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #EDEDED;
}

.stApp {
    background: radial-gradient(circle at 15% 50%, rgba(0,180,216,0.12), #0A0A0C 60%),
                radial-gradient(circle at 85% 30%, rgba(100,50,255,0.10), #0A0A0C 60%);
    background-attachment: fixed;
}

.hero {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 32px;
    text-align: center;
    margin-bottom: 28px;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #ffffff, #00B4D8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: #B0B0B0;
    font-size: 1.05rem;
}

.answer-box {
    background: rgba(255,255,255,0.06);
    border-left: 4px solid #00B4D8;
    padding: 18px;
    border-radius: 14px;
    margin-top: 10px;
}

.confidence {
    font-size: 0.85rem;
    color: #00B4D8;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. HERO HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">Document Intelligence System</div>
    <div class="hero-sub">
        Document-grounded Question Answering using RAG  
        <br/>No hallucination • Evidence-backed • Explainable
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 4. INIT LLM
# =========================================================
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ GROQ_API_KEY not set.")
    st.stop()

generator = AnswerGenerator(api_key)

# =========================================================
# 5. SESSION STATE
# =========================================================
for key in ["vector_store", "embedder", "reranker", "processed_file", "messages"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "messages" else []

# =========================================================
# 6. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom: 10px;">
            <h3 style="color: #FFFFFF; margin-bottom: 2px;">
                📄 Document Intelligence
            </h3>
            <span style="color: #9AA0A6; font-size: 0.8rem;">
                Strict document-grounded RAG
            </span>
        </div>

        <hr style="border: 0.5px solid rgba(255,255,255,0.12); margin: 10px 0;">

        <h4 style="color: #FFFFFF; margin-bottom: 6px;">
           ℹ️ Instructions
        </h4>

        <div style="
            color: #B0B0B0;
            font-size: 0.8rem;
            line-height: 1.45;
        ">
            Upload a PDF<br>
            Ask document-specific questions<br>
            Answers are generated only from the document<br>
            Unsupported queries are rejected<br>
            Evidence and confidence are shown
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 7. FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader("📤 Upload a PDF to initialize analysis", type=["pdf"])

if uploaded_file:
    if st.session_state.processed_file != uploaded_file.name:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        docs = load_documents(pdf_path)

        all_chunks, sources = [], []
        for doc in docs:
            for c in chunk_text(doc["text"]):
                all_chunks.append(c)
                sources.append(doc["source"])

        with st.spinner("Building document index..."):
            embedder = TextEmbedder()
            embeddings = embedder.embed_texts(all_chunks)

            vector_store = FaissVectorStore(embeddings.shape[1])
            vector_store.add(embeddings, all_chunks, sources)

        st.session_state.embedder = embedder
        st.session_state.vector_store = vector_store
        st.session_state.reranker = CrossEncoderReranker()
        st.session_state.processed_file = uploaded_file.name

        os.remove(pdf_path)

    st.success(f"Active Document: {uploaded_file.name}")

    # =====================================================
    # 8. CHAT
    # =====================================================
    st.markdown("### 💬 Ask Questions")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask a question about the document"):

        if len(question.strip()) < 4:
            st.warning("Please ask a meaningful document-related question.")
            st.stop()

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document…"):

                query_embedding = st.session_state.embedder.embed_query(question)

                faiss_results = st.session_state.vector_store.search(
                    query_embedding, top_k=20
                )

                reranked_results = st.session_state.reranker.rerank(
                    question, faiss_results, top_k=5
                )

                confidence = compute_confidence(reranked_results)
                avg_conf = confidence["average"]
                max_conf = confidence["max"]

                keyword_ok = keyword_overlap(question, reranked_results)

                # =================================================
                # 🔒 FINAL HARD GATE (NO EXCEPTIONS)
                # =================================================
                if (
                    not reranked_results
                    or max_conf < 0.45
                    or avg_conf < 0.30
                    or not keyword_ok
                ):
                    answer = (
                        "The uploaded document does not contain information "
                        "relevant to this question."
                    )
                    evidence = ["No supporting content found in the document."]
                    citations = []
                else:
                    answer, evidence, citations = generator.generate(
                        question, reranked_results
                    )

                st.markdown(
                    f"<div class='answer-box'>{answer}</div>",
                    unsafe_allow_html=True
                )

                if evidence:
                    with st.expander("🧾 Supporting Evidence from Document"):
                        for e in evidence:
                            st.markdown(f"- {e}")

                if citations:
                    with st.expander("📚 Source Citations"):
                        for c in citations:
                            st.markdown(f"- {c}")

                st.markdown(
                    f"<div class='confidence'>📊 Confidence Score: {avg_conf:.2f}</div>",
                    unsafe_allow_html=True
                )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("Upload a PDF to begin document analysis.")



