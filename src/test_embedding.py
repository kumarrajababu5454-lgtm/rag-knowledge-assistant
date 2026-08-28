from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

text = "Employees receive 18 paid leave days per year."

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text
)

embedding = response.embeddings[0].values

print("Embedding created successfully!")
print("Number of dimensions:", len(embedding))
print("First 10 values:", embedding[:10])