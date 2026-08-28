from pathlib import Path
from chunking import chunk_text

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Build the document path
DOCUMENT_PATH = PROJECT_ROOT / "data" / "company_policy.txt"
print("Looking for document at:")
print(DOCUMENT_PATH)

# Read document
with open(DOCUMENT_PATH, "r", encoding="utf-8") as file:
    text = file.read()

# Create chunks
chunks = chunk_text(text)

print("\nNumber of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)