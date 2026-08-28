from pathlib import Path

from chunking import chunk_text
from embedder import GeminiEmbedder
from vector_store import VectorStore


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Document path
DOCUMENT_PATH = PROJECT_ROOT / "data" / "company_policy.txt"

print("Reading document...")

with open(DOCUMENT_PATH, "r", encoding="utf-8") as file:
    text = file.read()


# 1. Chunk document
chunks = chunk_text(text)

print(f"Created {len(chunks)} chunks.")


# 2. Create embeddings
embedder = GeminiEmbedder()

vectors = []
metadata = []


for index, chunk in enumerate(chunks):

    print(f"Creating embedding {index + 1}/{len(chunks)}...")

    vector = embedder.embed(chunk)

    vectors.append(vector)

    metadata.append({
        "text": chunk,
        "source": "company_policy.txt",
        "chunk_id": index
    })


# 3. Store vectors
store = VectorStore(
    storage_path=PROJECT_ROOT / "data" / "vector_store"
)

store.save(vectors, metadata)


print("\nIndex built successfully!")
print("Number of vectors:", len(vectors))
print("Vector dimensions:", len(vectors[0]))