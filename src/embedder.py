from google import genai
from dotenv import load_dotenv
import os


load_dotenv()


class GeminiEmbedder:

    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        self.model = "gemini-embedding-001"

    def embed(self, text):
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )

        return response.embeddings[0].values