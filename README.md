# 📚 RAG Knowledge Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** application built with **Python, Gemini, embeddings, semantic search, NumPy, and Streamlit**.

The application allows users to ask questions about company documents. Instead of asking the language model to answer from its general knowledge, the application first searches the company's knowledge base, retrieves the most relevant information, and then gives that information to Gemini as context.

## 🚀 Live Application

**Live Demo:**
https://rag-knowledge-assistant-usxanajvtckochz4opqqj2.streamlit.app/

---

# 🎯 What This Project Does

The application follows this workflow:

```text
User Question
      ↓
Streamlit Frontend
      ↓
RAG Pipeline
      ↓
Convert Question → Embedding
      ↓
Semantic Similarity Search
      ↓
Retrieve Relevant Document Chunks
      ↓
Build Context
      ↓
Send Context + Question to Gemini
      ↓
Generate Grounded Answer
      ↓
Display Answer + Sources
```

The important difference between a normal LLM application and this application is:

```text
Normal LLM

User Question
      ↓
Gemini
      ↓
Answer
```

versus:

```text
RAG Application

User Question
      ↓
Search Knowledge Base
      ↓
Retrieve Relevant Information
      ↓
Gemini + Retrieved Context
      ↓
Grounded Answer
```

---

# 🧠 What is RAG?

RAG stands for:

**Retrieval-Augmented Generation**

It combines two major processes:

### 1. Retrieval

Find relevant information from an external knowledge source.

### 2. Generation

Use a foundation model to generate a natural-language answer using the retrieved information.

Therefore:

```text
RAG = Retrieval + Generation
```

This is useful when the information required by an AI application is:

* private
* company-specific
* frequently changing
* too large to place directly into a prompt
* not part of the model's original training knowledge

---

# 🏗️ Project Architecture

```text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │ Streamlit UI    │
                  │ frontend/app.py │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  RAGPipeline    │
                  │    src/rag.py   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Retriever     │
                  │ src/retriever.py│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Vector Store    │
                  │ vectors.npy     │
                  │ metadata.json   │
                  └────────┬────────┘
                           │
                           ▼
                  Relevant Chunks
                           │
                           ▼
                  ┌─────────────────┐
                  │ Gemini Model    │
                  │ Foundation Model│
                  └────────┬────────┘
                           │
                           ▼
                    Final Answer
                           │
                           ▼
                  Sources + Answer
```

---

# 📁 Project Structure

```text
rag-knowledge-assistant/
│
├── data/
│   ├── company_policy.txt
│   │
│   └── vector_store/
│       ├── vectors.npy
│       └── metadata.json
│
├── frontend/
│   └── app.py
│
├── src/
│   ├── build_index.py
│   ├── chunking.py
│   ├── embedder.py
│   ├── list_models.py
│   ├── rag.py
│   ├── retriever.py
│   ├── vector_store.py
│   │
│   ├── test_chunking.py
│   ├── test_embedding.py
│   ├── test_gemini.py
│   ├── test_rag.py
│   ├── test_retrieval.py
│   └── test_vector_store.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 📂 Folder-by-Folder Explanation

## `data/`

This folder contains the application's knowledge.

The original document is:

```text
data/company_policy.txt
```

This is the external knowledge source used by the RAG system.

Example:

```text
Employees receive 18 paid leave days per year.

Employees can request sick leave according to company policy.

The manager must approve planned leave.
```

The important idea is:

```text
Company Document
       ↓
RAG Knowledge Base
```

---

# 📄 `data/company_policy.txt`

This is the source document.

The RAG system does not directly send the entire document to Gemini every time.

Instead:

```text
Document
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Store
   ↓
Retrieval
```

This makes semantic search possible.

---

# 📁 `data/vector_store/`

This directory contains the generated vector index.

It contains:

```text
vectors.npy
metadata.json
```

---

# 🧮 `vectors.npy`

This file stores the numerical embedding vectors.

For example, text:

```text
Employees receive 18 paid leave days per year.
```

is converted into a numerical vector.

In this project, Gemini's embedding model generated vectors with:

```text
3072 dimensions
```

Conceptually:

```text
Text
 ↓
Embedding Model
 ↓
[0.021, -0.183, 0.442, ...]
 ↓
3072-dimensional vector
```

These vectors allow semantic similarity search.

---

# 🗂️ `metadata.json`

The vector alone does not tell us what text it represents.

Therefore metadata is stored alongside the vectors.

Conceptually:

```json
{
    "text": "Employees receive 18 paid leave days per year.",
    "source": "company_policy.txt"
}
```

The metadata allows the application to know:

* which text was retrieved
* which document it came from
* what source should be displayed to the user

---

# 📁 `src/`

The `src` directory contains the application's backend logic.

Even though we are not using FastAPI, the RAG logic is still separated from the Streamlit frontend.

This is an important software-engineering practice.

```text
frontend/
    UI

src/
    AI / RAG logic
```

---

# 🧩 `src/chunking.py`

This module is responsible for splitting a large document into smaller pieces.

Why?

A large document should not always be treated as one giant block.

Instead:

```text
Large Document
      ↓
Chunk 1
Chunk 2
Chunk 3
Chunk 4
...
```

A simple implementation can look like:

```python
def chunk_text(text):
    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if paragraph:
            chunks.append(paragraph)

    return chunks
```

The actual chunking strategy can be improved later using:

* token-based chunking
* overlapping chunks
* sentence-aware chunking
* recursive character splitting

The important RAG concept is:

```text
Document → Smaller Retrievable Units
```

---

# 🧠 `src/embedder.py`

This module communicates with Gemini's embedding model.

The project uses:

```text
gemini-embedding-001
```

Conceptually:

```python
from google import genai

client = genai.Client(
    api_key=API_KEY
)

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text
)
```

The purpose is:

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

The same embedding process is used for:

```text
Document chunks
```

and later:

```text
User question
```

This is essential because both need to exist in the same vector space for similarity search.

---

# 🗄️ `src/vector_store.py`

This module handles storage and loading of vectors and their metadata.

For this project, a lightweight local vector store was intentionally used instead of a large dedicated vector database.

The basic idea is:

```text
Embedding vectors
      ↓
NumPy
      ↓
vectors.npy
```

and:

```text
Chunk information
      ↓
JSON
      ↓
metadata.json
```

This approach is useful for:

* learning
* prototypes
* small datasets
* lightweight projects
* machines with limited resources

For production-scale applications, alternatives could include:

* FAISS
* Chroma
* Qdrant
* Pinecone
* Weaviate
* Milvus

---

# 🔎 `src/retriever.py`

The retriever is responsible for finding the most relevant chunks.

The process is:

```text
User Question
      ↓
Question Embedding
      ↓
Compare With Stored Vectors
      ↓
Similarity Scores
      ↓
Sort By Relevance
      ↓
Top K Results
```

For example:

```text
Question:

How many paid leave days do employees receive?
```

The retriever may return:

```text
Result 1
Score: 0.91

Employees receive 18 paid leave days per year.
```

The application then sends the highest-ranking results to Gemini.

---

# 🏗️ `src/build_index.py`

This script builds the RAG index.

Its job is:

```text
Read Document
     ↓
Chunk Document
     ↓
Generate Embeddings
     ↓
Store Vectors
     ↓
Store Metadata
```

Conceptually:

```python
from pathlib import Path

document_path = Path("data/company_policy.txt")

text = document_path.read_text(
    encoding="utf-8"
)

chunks = chunk_text(text)

vectors = []

for chunk in chunks:
    vector = embed(chunk)
    vectors.append(vector)

save_vectors(vectors)
save_metadata(chunks)
```

After indexing, the project contains:

```text
data/vector_store/
├── vectors.npy
└── metadata.json
```

In this project, the completed index contained:

```text
Number of vectors: 8
Vector dimensions: 3072
```

---

# 🤖 `src/rag.py`

This is the main RAG orchestration layer.

The `RAGPipeline` class connects:

```text
Retriever
+
Gemini
=
RAG Application
```

The current implementation follows this structure:

```python
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

        results = self.retriever.search(
            question,
            top_k=top_k
        )

        context_parts = []

        for result in results:

            context_parts.append(
                f"Source: {result['source']}\n"
                f"Content: {result['text']}"
            )

        context = "\n\n".join(context_parts)

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

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return {
            "answer": response.text,
            "sources": results
        }
```

---

# 🔥 Understanding `RAGPipeline.answer()`

The most important part of the project is this function.

## Step 1 — Receive the question

```python
def answer(self, question, top_k=3):
```

Example:

```text
How many paid leave days do employees receive?
```

---

## Step 2 — Retrieve relevant information

```python
results = self.retriever.search(
    question,
    top_k=top_k
)
```

This performs semantic search.

`top_k=3` means:

```text
Return the 3 most relevant chunks.
```

---

## Step 3 — Build context

The retrieved chunks are converted into a single context string.

```python
context_parts = []

for result in results:

    context_parts.append(
        f"Source: {result['source']}\n"
        f"Content: {result['text']}"
    )
```

Then:

```python
context = "\n\n".join(context_parts)
```

The result looks conceptually like:

```text
Source: company_policy.txt
Content: Employees receive 18 paid leave days per year.

Source: company_policy.txt
Content: The manager must approve planned leave.
```

---

# 📝 RAG Prompt

The retrieved context is placed inside the prompt.

```text
You are a helpful company policy assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say that you do not have enough information.

Do not invent or assume information.

Context:
[Retrieved Information]

User Question:
[User Question]

Answer:
```

This is an important grounding technique.

The model is explicitly instructed:

```text
Use the retrieved information.
Do not invent information.
```

---

# 🛡️ Grounded Generation

One of the tests performed in this project was asking about information that did not exist in the document.

For example:

```text
What is the company's maternity leave policy?
```

Instead of inventing an answer, the system responds with:

```text
I do not have enough information.
```

This demonstrates a basic form of hallucination control.

Important:

This does **not** mean the system can never hallucinate.

It means the prompt and retrieval design encourage the model to stay grounded in the supplied context.

---

# 🖥️ `frontend/app.py`

This file is the Streamlit user interface.

The frontend is intentionally kept separate from the RAG logic.

The application:

1. Creates the Streamlit page.
2. Loads the RAG pipeline.
3. Accepts the user's question.
4. Calls `rag.answer()`.
5. Displays the answer.
6. Displays the retrieved sources.

Core structure:

```python
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.append(str(SRC_PATH))

from rag import RAGPipeline
```

The project root is calculated so that the application can correctly locate:

```text
src/
data/
```

---

# ⚡ Loading the RAG Pipeline

The application uses Streamlit resource caching:

```python
@st.cache_resource
def load_rag():
    return RAGPipeline()
```

Then:

```python
rag = load_rag()
```

This prevents the application from unnecessarily recreating the RAG pipeline on every interaction.

---

# 🧑‍💻 User Input

The application creates a text input:

```python
question = st.text_input(
    "Ask a question:",
    placeholder="Example: How many paid leave days do employees receive?"
)
```

Then the user clicks:

```text
Ask
```

---

# 🔄 Calling the RAG Pipeline

When the user clicks the button:

```python
if st.button("Ask", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            result = rag.answer(question)
```

The frontend does not perform the retrieval itself.

It simply calls:

```python
rag.answer(question)
```

This is good separation of responsibilities.

---

# 💬 Displaying the Answer

The result returned by `RAGPipeline` contains:

```python
{
    "answer": response.text,
    "sources": results
}
```

The frontend displays:

```python
st.subheader("Answer")

st.write(result["answer"])
```

---

# 📖 Displaying Sources

The application also displays the retrieved source chunks:

```python
st.subheader("Sources")

for source in result["sources"]:

    with st.expander(
        f"📄 {source['source']} — Score: {source['score']:.3f}"
    ):

        st.write(source["text"])
```

Therefore the user can see:

```text
Answer
  ↓
Sources
  ↓
Document
  ↓
Retrieved text
  ↓
Similarity score
```

This improves transparency.

---

# 🧪 Testing Files

The project also contains several test scripts.

```text
src/
├── test_chunking.py
├── test_embedding.py
├── test_gemini.py
├── test_rag.py
├── test_retrieval.py
└── test_vector_store.py
```

These were created to test individual pieces of the RAG pipeline.

---

# 🧪 `test_chunking.py`

Tests whether document splitting works correctly.

Conceptually:

```python
text = """
Paragraph one.

Paragraph two.

Paragraph three.
"""

chunks = chunk_text(text)

print(chunks)
```

The goal is to verify:

```text
Document
 ↓
Correct chunks
```

---

# 🧪 `test_embedding.py`

Tests the embedding model.

The goal is to verify that:

```text
Text
 ↓
Gemini Embedding Model
 ↓
Vector
```

works correctly.

The project confirmed the embedding dimension:

```text
3072
```

---

# 🧪 `test_gemini.py`

Tests communication with Gemini.

The purpose is to verify:

```text
Python
 ↓
Gemini API
 ↓
Model Response
```

before integrating the model into the complete RAG pipeline.

---

# 🧪 `test_retrieval.py`

Tests semantic retrieval.

Example:

```text
Question
 ↓
Embedding
 ↓
Similarity Search
 ↓
Relevant Chunk
```

This ensures that the correct document information is being retrieved.

---

# 🧪 `test_vector_store.py`

Tests saving and loading:

```text
vectors.npy
metadata.json
```

This verifies that the vector store is functioning correctly.

---

# 🧪 `test_rag.py`

Tests the complete RAG flow:

```text
Question
 ↓
Retriever
 ↓
Context
 ↓
Gemini
 ↓
Answer
```

This is an integration-style test for the core application.

---

# 📋 `src/list_models.py`

This utility was used during development to inspect the models available through the Gemini API.

This is useful when:

* checking model names
* checking available capabilities
* debugging API configuration
* selecting an appropriate foundation model

---

# 🔐 `.env`

The `.env` file contains the Gemini API key locally.

Example:

```text
GEMINI_API_KEY=your_api_key_here
```

## ⚠️ NEVER commit `.env`

The API key is a secret.

It must not be uploaded to GitHub.

---

# 🛡️ `.gitignore`

The project uses:

```text
.env
__pycache__/
*.pyc
```

This prevents:

```text
.env
```

from being committed to GitHub.

This is extremely important for API-based AI applications.

---

# 📦 `requirements.txt`

The project requires the packages used by the application.

Example:

```text
streamlit
google-genai
python-dotenv
numpy
```

These dependencies allow Streamlit Cloud to install the required Python libraries during deployment.

---

# 🛠️ Complete Setup From Scratch

Anyone can reproduce the project using the following process.

---

## Step 1 — Install Python

Install Python 3.10+.

Verify:

```powershell
python --version
```

Example:

```text
Python 3.13.x
```

---

# Step 2 — Create the project

```powershell
mkdir rag-knowledge-assistant

cd rag-knowledge-assistant
```

Open VS Code:

```powershell
code .
```

---

# Step 3 — Create a virtual environment

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see:

```text
(.venv)
```

in the terminal.

---

# Step 4 — Install dependencies

```powershell
pip install streamlit google-genai python-dotenv numpy
```

Then create:

```text
requirements.txt
```

with:

```text
streamlit
google-genai
python-dotenv
numpy
```

---

# Step 5 — Create the project folders

```powershell
mkdir data
mkdir data\vector_store
mkdir frontend
mkdir src
```

---

# Step 6 — Create the document

Create:

```text
data/company_policy.txt
```

Add company knowledge such as:

```text
Employees receive 18 paid leave days per year.

Employees can request sick leave according to company policy.

The manager must approve planned leave.
```

---

# Step 7 — Create the `.env` file

Create:

```text
.env
```

Add:

```text
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Never commit this file.

---

# Step 8 — Create the `.gitignore`

Create:

```text
.gitignore
```

Add:

```text
.env
__pycache__/
*.pyc
```

---

# Step 9 — Build the document processing pipeline

The application performs:

```text
Read Document
      ↓
Chunk Document
      ↓
Generate Embeddings
      ↓
Save Vectors
      ↓
Save Metadata
```

Run the indexing script:

```powershell
python src/build_index.py
```

After successful indexing:

```text
data/
└── vector_store/
    ├── vectors.npy
    └── metadata.json
```

---

# Step 10 — Test embeddings

Run:

```powershell
python src/test_embedding.py
```

The embedding system should successfully produce vectors.

---

# Step 11 — Test retrieval

Run:

```powershell
python src/test_retrieval.py
```

Example question:

```text
How many paid leave days do employees receive?
```

The retriever should find:

```text
Employees receive 18 paid leave days per year.
```

---

# Step 12 — Test the RAG pipeline

Run:

```powershell
python src/test_rag.py
```

The pipeline should produce a grounded answer.

---

# Step 13 — Run Streamlit locally

Run:

```powershell
streamlit run frontend/app.py
```

Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open it in your browser.

---

# 🔬 How a Query Works Internally

Suppose the user asks:

```text
How many paid leave days do employees receive?
```

### Step 1

Streamlit receives:

```text
question
```

### Step 2

The RAG pipeline sends the question to the retriever.

### Step 3

The question is converted into an embedding:

```text
Question
 ↓
3072-dimensional vector
```

### Step 4

The vector is compared with stored document vectors.

### Step 5

The most relevant chunks are selected.

For example:

```text
Employees receive 18 paid leave days per year.
```

### Step 6

The retrieved information is placed into the prompt.

### Step 7

Gemini receives:

```text
Context
+
Question
```

### Step 8

Gemini generates:

```text
Employees receive 18 paid leave days per year.
```

### Step 9

The application displays:

```text
Answer

Employees receive 18 paid leave days per year.

Sources

company_policy.txt
```

---

# ❌ What Happens When Information Is Missing?

Suppose the user asks:

```text
What is the company's maternity leave policy?
```

But the document doesn't contain that information.

The application retrieves the available context and instructs Gemini:

```text
Answer using ONLY the provided context.

If the answer cannot be found,
say that you do not have enough information.

Do not invent or assume information.
```

Therefore the expected behavior is:

```text
I do not have enough information.
```

This is called **grounded generation**.

---

# ☁️ Deployment

The application is deployed using Streamlit Community Cloud.

Deployment architecture:

```text
GitHub Repository
       ↓
Streamlit Cloud
       ↓
requirements.txt
       ↓
frontend/app.py
       ↓
RAG Pipeline
       ↓
Gemini API
```

The deployed application's entry point is:

```text
frontend/app.py
```

---

# 🔐 Streamlit Secrets

The local `.env` file is not uploaded to GitHub.

Instead, the deployed application receives the API key through Streamlit Secrets.

Add:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Do not publish the actual key.

---

# 🚀 Live Deployment

The current deployed application is:

https://rag-knowledge-assistant-usxanajvtckochz4opqqj2.streamlit.app/

---

# 🔄 Git and GitHub Workflow

After creating or modifying files:

```powershell
git status
```

Add changes:

```powershell
git add .
```

Commit:

```powershell
git commit -m "Update RAG application"
```

Push:

```powershell
git push origin master
```

Verify:

```powershell
git status
```

Expected:

```text
Your branch is up to date with 'origin/master'.

nothing to commit, working tree clean
```

---

# 🌐 Deployment Update Workflow

Once the Streamlit application is connected to GitHub, the normal workflow becomes:

```text
Change Code
    ↓
Test Locally
    ↓
git add .
    ↓
git commit
    ↓
git push
    ↓
GitHub
    ↓
Streamlit Cloud detects change
    ↓
Application redeploys
```

This is the basic CI/CD-style workflow used by many deployed applications.

---

# 🧠 Key AI Engineering Concepts Demonstrated

This project demonstrates:

| Concept               | Implemented |
| --------------------- | ----------- |
| Foundation Models     | ✅           |
| Gemini API            | ✅           |
| Prompt Engineering    | ✅           |
| Document Ingestion    | ✅           |
| Document Chunking     | ✅           |
| Embeddings            | ✅           |
| Semantic Search       | ✅           |
| Vector Storage        | ✅           |
| Retrieval             | ✅           |
| Context Construction  | ✅           |
| RAG                   | ✅           |
| Grounded Generation   | ✅           |
| Source Display        | ✅           |
| Similarity Scores     | ✅           |
| Streamlit UI          | ✅           |
| Environment Variables | ✅           |
| Git                   | ✅           |
| GitHub                | ✅           |
| Cloud Deployment      | ✅           |

---

# 🧩 Why We Did Not Use FastAPI

This project intentionally does not use FastAPI.

The architecture is:

```text
Streamlit
   ↓
RAG Pipeline
   ↓
Retriever
   ↓
Gemini
```

For this small portfolio application, adding FastAPI would introduce another service without providing a major benefit.

FastAPI becomes more useful when:

```text
Frontend
    ↓
HTTP API
    ↓
Backend
    ↓
RAG System
```

is required.

For this project, the simpler architecture is sufficient.

---

# 💡 Why We Used a Lightweight Vector Store

A dedicated vector database was not necessary for this project.

Instead:

```text
NumPy
+
JSON
```

were used.

This keeps the project:

* lightweight
* easy to understand
* easy to reproduce
* inexpensive
* suitable for small knowledge bases

For a production system with millions of documents, a dedicated vector database would be more appropriate.

---

# ⚠️ Current Limitations

This project is intentionally a learning/portfolio-scale RAG application.

Current limitations include:

* single primary knowledge document
* lightweight local vector store
* basic chunking strategy
* no authentication
* no user accounts
* no conversation persistence
* no advanced RAG evaluation framework
* no reranking model
* no production observability
* no FastAPI backend
* no large-scale vector database

These are opportunities for future versions.

---

# 🚀 Future Improvements

Possible Version 2 improvements:

```text
Multiple Documents
        ↓
PDF / DOCX / TXT ingestion
        ↓
Better Chunking
        ↓
Vector Database
        ↓
Reranking
        ↓
Conversation Memory
        ↓
RAG Evaluation
        ↓
Observability
```

Possible future features:

### 1. Multiple document support

```text
data/documents/
├── company_policy.txt
├── employee_handbook.txt
├── benefits.txt
└── leave_policy.txt
```

### 2. PDF ingestion

Allow the application to read PDF company documents.

### 3. Chat history

Instead of individual questions:

```text
User
 ↓
Question
 ↓
Answer
 ↓
Follow-up Question
 ↓
Context-aware Answer
```

### 4. Better evaluation

Evaluate:

* retrieval precision
* retrieval recall
* answer correctness
* groundedness
* hallucination rate
* latency

### 5. Production vector database

Replace the lightweight local vector store with:

```text
FAISS
Chroma
Qdrant
Pinecone
Weaviate
Milvus
```

depending on the requirements.

---

# 🎓 What I Learned From Building This Project

This project demonstrates the complete basic lifecycle of an AI application:

```text
1. Define the problem
        ↓
2. Collect knowledge
        ↓
3. Process documents
        ↓
4. Chunk documents
        ↓
5. Generate embeddings
        ↓
6. Store vectors
        ↓
7. Retrieve relevant information
        ↓
8. Build RAG context
        ↓
9. Call foundation model
        ↓
10. Ground the answer
        ↓
11. Build frontend
        ↓
12. Test
        ↓
13. Git/GitHub
        ↓
14. Deploy
```

This is the fundamental workflow behind many real-world RAG applications.

---

# 🏆 Final Architecture

```text
                         ┌──────────────────────┐
                         │      USER            │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Streamlit        │
                         │    frontend/app.py   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    RAGPipeline       │
                         │      src/rag.py      │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                ┌────────────────┐    ┌─────────────────┐
                │   Retriever    │    │  Gemini Model   │
                │retriever.py    │    │ Foundation Model│
                └───────┬────────┘    └────────┬────────┘
                        │                      │
                        ▼                      │
                ┌────────────────┐             │
                │ Vector Store   │             │
                │ vectors.npy    │             │
                │ metadata.json  │             │
                └───────┬────────┘             │
                        │                      │
                        ▼                      │
                ┌────────────────┐             │
                │ Company Policy │             │
                │ company_policy │             │
                │     .txt       │             │
                └────────────────┘             │
                                               │
                        Retrieved Context ──────┘
                                               │
                                               ▼
                                      ┌────────────────┐
                                      │ Grounded Answer│
                                      └───────┬────────┘
                                              │
                                              ▼
                                      ┌────────────────┐
                                      │ Answer + Source│
                                      └────────────────┘
```

---

# ⭐ Project Status

**Status: Completed — Version 1**

The application currently supports:

```text
✅ Document ingestion
✅ Chunking
✅ Gemini embeddings
✅ 3072-dimensional vectors
✅ Local vector storage
✅ Semantic retrieval
✅ Top-K retrieval
✅ RAG prompting
✅ Gemini generation
✅ Grounded answers
✅ "I don't know" behavior
✅ Source display
✅ Similarity scores
✅ Streamlit frontend
✅ Git/GitHub
✅ Cloud deployment
```

## 🔗 Live Demo

https://rag-knowledge-assistant-usxanajvtckochz4opqqj2.streamlit.app/

---

# 👨‍💻 Project Summary

This project was built to understand how modern AI applications use foundation models together with external knowledge.

The key lesson is:

```text
LLM alone
   ↓
Can generate answers

RAG
   ↓
Searches external knowledge
   ↓
Retrieves relevant information
   ↓
Provides context to LLM
   ↓
Generates a grounded answer
```

The project therefore demonstrates not just **how to call an LLM API**, but how to build a complete AI application around a foundation model.

---

# 📌 One-Line Portfolio Description

> Built and deployed an end-to-end Retrieval-Augmented Generation (RAG) knowledge assistant using Gemini embeddings, semantic vector search, grounded generation, source retrieval, and Streamlit.
