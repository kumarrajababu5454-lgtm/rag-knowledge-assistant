from pathlib import Path

from retriever import Retriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent

retriever = Retriever(
    storage_path=PROJECT_ROOT / "data" / "vector_store"
)


query = "How many paid leave days do employees receive?"

results = retriever.search(query, top_k=2)


print("\nQuery:")
print(query)

print("\nRetrieved results:")

for result in results:

    print("\n----------------------------")

    print("Score:", result["score"])

    print("Source:", result["source"])

    print("Chunk ID:", result["chunk_id"])

    print("Text:")
    print(result["text"])