from pathlib import Path

from google import genai
from dotenv import load_dotenv
import os

from retriever import Retriever


load_dotenv()


class RAGPipeline:

    def __init__(self):

        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        self.model = "gemini-3.6-flash"

        PROJECT_ROOT = Path(__file__).resolve().parent.parent

        self.retriever = Retriever(
            storage_path=PROJECT_ROOT / "data" / "vector_store"
        )

    def answer(self, question, top_k=3):

        # 1. Retrieve relevant documents
        results = self.retriever.search(
            question,
            top_k=top_k
        )

        # 2. Build context
        context_parts = []

        for result in results:

            context_parts.append(
                f"Source: {result['source']}\n"
                f"Content: {result['text']}"
            )

        context = "\n\n".join(context_parts)

        # 3. Create RAG prompt
        prompt = f"""
You are a helpful company policy assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say that you do not have enough information.

Do not invent or assume information.

Context:
{context}

User Question:
{question}

Answer:
"""

        # 4. Generate answer
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return {
            "answer": response.text,
            "sources": results
        }