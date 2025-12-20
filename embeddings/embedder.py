from sentence_transformers import SentenceTransformer
import numpy as np

class TextEmbedder:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # IMPORTANT for BGE
        )
        return embeddings

    def embed_query(self, query):
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding
