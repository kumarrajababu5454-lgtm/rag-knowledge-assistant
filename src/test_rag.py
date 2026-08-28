from rag import RAGPipeline


rag = RAGPipeline()


question = "What is the company's maternity leave policy?"

result = rag.answer(question)


print("\n==============================")
print("RAG ANSWER")
print("==============================")

print(result["answer"])


print("\n==============================")
print("SOURCES")
print("==============================")


for source in result["sources"]:

    print("\n----------------------------")

    print("Score:", source["score"])

    print("Source:", source["source"])

    print("Chunk ID:", source["chunk_id"])

    print("Text:", source["text"])