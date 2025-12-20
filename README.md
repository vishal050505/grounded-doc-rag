📄 Document Intelligence System (RAG)

A document-grounded question answering system that allows users to upload a PDF and ask questions.
Answers are generated strictly from the document using a Retrieval-Augmented Generation (RAG) pipeline with hallucination prevention, evidence, and confidence scoring.

🔑 Key Features

📄 PDF-based Question Answering

🧠 Semantic Retrieval using Sentence Embeddings

🔎 FAISS Vector Search + Re-ranking

🚫 Hallucination Prevention (retrieval gate)

🧾 Evidence-aware Answers

📊 Confidence Score

🎨 Clean Streamlit UI

🧠 How It Works
PDF Upload
 → Text Extraction
 → Chunking
 → Embeddings (MiniLM)
 → FAISS Vector Search
 → Re-ranking
 → LLM Answer Generation


The LLM is only called when the document supports the query

If information is missing, the system refuses to answer

🏗️ Tech Stack

UI: Streamlit

Embeddings: HuggingFace Sentence Transformers (MiniLM)

Vector DB: FAISS

Re-ranking: Cross-Encoder

LLM: LLaMA-3.1-8B (Groq API)

🚀 Why This Project

Unlike typical “Chat with PDF” demos, this system:

Implements RAG manually (no LangChain dependency)

Prevents hallucinations instead of masking them

Provides evidence and confidence for every answer

Prioritizes trust over aggressive answering

🎤 One-Line Summary

A RAG-based document intelligence system that answers questions strictly from uploaded PDFs with evidence and hallucination prevention.

▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py