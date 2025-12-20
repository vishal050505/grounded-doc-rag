import faiss

class FaissVectorStore:
    def __init__(self, embedding_dim):
        # Inner Product is correct for normalized embeddings (BGE)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.texts = []
        self.sources = []

    def add(self, embeddings, texts, sources):
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.sources.extend(sources)

    def search(self, query_embedding, top_k=5):
        scores, indices = self.index.search(query_embedding, top_k)
        results = []

        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.texts[idx],
                "source": self.sources[idx],
                "score": float(score)
            })

        return results
