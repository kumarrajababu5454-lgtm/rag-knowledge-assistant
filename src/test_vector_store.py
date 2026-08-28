from vector_store import VectorStore

import numpy as np


# Example embeddings
vectors = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
]

metadata = [
    {
        "text": "Employees receive 18 paid leave days per year.",
        "source": "company_policy.txt",
        "chunk_id": 1,
    },
    {
        "text": "The manager must approve planned leave requests.",
        "source": "company_policy.txt",
        "chunk_id": 2,
    },
]


store = VectorStore()

store.save(vectors, metadata)

loaded_vectors, loaded_metadata = store.load()

print("Vectors loaded successfully!")
print("Vector shape:", loaded_vectors.shape)

print("\nMetadata:")
for item in loaded_metadata:
    print(item)