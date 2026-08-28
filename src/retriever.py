import numpy as np

from vector_store import VectorStore
from embedder import GeminiEmbedder


class Retriever:

    def __init__(self, storage_path="data/vector_store"):
        self.store = VectorStore(storage_path)
        self.embedder = GeminiEmbedder()

        self.vectors, self.metadata = self.store.load()

    def search(self, query, top_k=2):

        # Create embedding for user question
        query_vector = self.embedder.embed(query)

        query_vector = np.array(query_vector)

        # Normalize vectors
        document_vectors = self.vectors / np.linalg.norm(
            self.vectors,
            axis=1,
            keepdims=True
        )

        query_vector = query_vector / np.linalg.norm(query_vector)

        # Cosine similarity
        similarities = document_vectors @ query_vector

        # Get highest scoring documents
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for index in top_indices:

            results.append({
                "text": self.metadata[index]["text"],
                "source": self.metadata[index]["source"],
                "chunk_id": self.metadata[index]["chunk_id"],
                "score": float(similarities[index])
            })

        return results