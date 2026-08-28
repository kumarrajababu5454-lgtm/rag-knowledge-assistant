import json
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, storage_path="data/vector_store"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.vectors_file = self.storage_path / "vectors.npy"
        self.metadata_file = self.storage_path / "metadata.json"

    def save(self, vectors, metadata):
        """
        Save embedding vectors and their corresponding metadata.
        """

        np.save(self.vectors_file, np.array(vectors))

        with open(self.metadata_file, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

    def load(self):
        """
        Load vectors and metadata from disk.
        """

        vectors = np.load(self.vectors_file)

        with open(self.metadata_file, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        return vectors, metadata