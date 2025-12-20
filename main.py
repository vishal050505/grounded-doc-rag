import os

from ingestion.loader import load_documents
from chunking.chunker import chunk_text
from embeddings.embedder import TextEmbedder
from vector_store.faiss_index import FaissVectorStore
from generation.answer_generator import AnswerGenerator
from utils.confidence import compute_confidence


# MODULE 1 & 2 

# Load documents
docs = load_documents("data/docs")

# Chunk documents
all_chunks = []
sources = []

for doc in docs:
    chunks = chunk_text(doc["text"])
    for chunk in chunks:
        all_chunks.append(chunk)
        sources.append(doc["source"])

print(f"Total chunks created: {len(all_chunks)}")

# Create embeddings
embedder = TextEmbedder()
embeddings = embedder.embed_texts(all_chunks)

# Build FAISS index
embedding_dim = embeddings.shape[1]
vector_store = FaissVectorStore(embedding_dim)
vector_store.add(embeddings, all_chunks, sources)

print("FAISS index built successfully.")


# MODULE 3 (GROQ) 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
generator = AnswerGenerator(GROQ_API_KEY)

print("\n🔍 Ask questions from your documents (type 'exit' to quit)\n")

while True:
    query = input("❓ Question: ").strip()

    if query.lower() == "exit":
        print("👋 Exiting Q&A. Goodbye!")
        break

    # Retrieve
    query_embedding = embedder.embed_texts([query])
    results = vector_store.search(query_embedding, top_k=5)

    # Generate answer
    answer, citations = generator.generate(query, results)
    confidence = compute_confidence(results)

    print("\n📘 ANSWER:\n")
    print(answer)

    print("\n📚 CITATIONS:")
    for c in citations:
        print("-", c)

    print("\n📊 CONFIDENCE SCORE:", confidence)

    if confidence < 0.3:
        print("⚠️ Low confidence: Answer may be incomplete.")

    print("\n" + "-" * 60 + "\n")
