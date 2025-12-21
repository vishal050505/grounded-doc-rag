import streamlit as st
import tempfile
import os

from ingestion.loader import load_documents
from chunking.chunker import chunk_text
from embeddings.embedder import TextEmbedder
from vector_store.faiss_index import FaissVectorStore
from generation.answer_generator import AnswerGenerator
from utils.confidence import compute_confidence
from reranker.cross_encoder import CrossEncoderReranker


# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. UI STYLES
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #EDEDED;
}

.stApp {
    background: radial-gradient(circle at 15% 50%, rgba(0,180,216,0.1), #0A0A0C 60%),
                radial-gradient(circle at 85% 30%, rgba(100,50,255,0.08), #0A0A0C 60%);
    background-attachment: fixed;
}

.hero {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin-bottom: 30px;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
}

.hero-sub {
    color: #A0A0A0;
}

.answer-box {
    background: rgba(255,255,255,0.05);
    border-left: 4px solid #00B4D8;
    padding: 16px;
    border-radius: 12px;
}

.confidence {
    font-size: 0.85rem;
    color: #00B4D8;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">Document Intelligence System</div>
    <div class="hero-sub">
        Strict document-grounded question answering using RAG
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 4. INIT LLM
# =========================================================
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not set.")
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
    st.markdown("### 📄 Document Intelligence System")
    st.caption("Document-grounded RAG • No hallucination")

# =========================================================
# 7. FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

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

        embedder = TextEmbedder()
        embeddings = embedder.embed_texts(all_chunks)

        vector_store = FaissVectorStore(embeddings.shape[1])
        vector_store.add(embeddings, all_chunks, sources)

        st.session_state.embedder = embedder
        st.session_state.vector_store = vector_store
        st.session_state.processed_file = uploaded_file.name
        st.session_state.reranker = CrossEncoderReranker()

        os.remove(pdf_path)

    st.success(f"Active Document: {uploaded_file.name}")

    # =====================================================
    # 8. CHAT
    # =====================================================
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask a question about the document"):

        # 🚫 Block garbage / nonsense input
        if len(question.strip()) < 4:
            st.warning("Please ask a meaningful document-related question.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})

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

                # =================================================
                # 🔒 FINAL HARD GATE (NO EXCEPTIONS)
                # =================================================
                if (
                    not reranked_results
                    or max_conf < 0.45
                    or avg_conf < 0.30
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

                st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

                if evidence:
                    with st.expander("🧾 Evidence from Document"):
                        for e in evidence:
                            st.markdown(f"- {e}")

                if citations:
                    with st.expander("📚 Source"):
                        for c in citations:
                            st.markdown(f"- {c}")

                st.markdown(
                    f"<div class='confidence'>Confidence Score: {avg_conf:.2f}</div>",
                    unsafe_allow_html=True
                )

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Upload a PDF to begin.")
