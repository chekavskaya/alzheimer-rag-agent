from rag.retrieval import Retriever
from rag.generator import Generator
from rag.pipeline import RAGPipeline
import os

if __name__ == "__main__":
    retriever = Retriever(
        data_path="data/pubmed_chunks.csv",
        text_column="text",
        metadata_columns=["pmid", "title"]
    )

    generator = Generator(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="gpt-4o-mini"
    )

    rag = RAGPipeline(
        retriever=retriever,
        generator=generator,
        top_k=3
    )

    question = "What are potential therapeutic targets for Alzheimer's disease?"

    result = rag.answer(question)

    print("\nQUESTION:")
    print(result["question"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for src in result["sources"]:
        print(f"[{src['id']}] PMID {src['pmid']} — {src['title']}")


