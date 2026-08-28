import streamlit as st
from pathlib import Path
import sys

# Add src directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"

sys.path.append(str(SRC_PATH))

from rag import RAGPipeline


# Page configuration
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="centered"
)


# Title
st.title("📚 RAG Knowledge Assistant")

st.write(
    "Ask questions about the information contained in the company documents."
)


# Initialize RAG pipeline
@st.cache_resource
def load_rag():
    return RAGPipeline()


rag = load_rag()


# User question
question = st.text_input(
    "Ask a question:",
    placeholder="Example: How many paid leave days do employees receive?"
)


# Ask button
if st.button("Ask", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching documents and generating answer..."):

            result = rag.answer(question)


        # Answer
        st.subheader("Answer")

        st.write(result["answer"])


        # Sources
        st.subheader("Sources")

        for source in result["sources"]:

            with st.expander(
                f"📄 {source['source']} — Score: {source['score']:.3f}"
            ):

                st.write(source["text"])