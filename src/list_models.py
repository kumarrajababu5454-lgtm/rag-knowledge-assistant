from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("\nGemini models supporting generateContent:\n")

for model in client.models.list():
    if hasattr(model, "supported_actions"):
        if "generateContent" in model.supported_actions:
            print(model.name)