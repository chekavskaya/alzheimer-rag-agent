class RAGPipeline:
    def __init__(self, retriever, generator, top_k=5):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def answer(self, question: str):
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            top_k=self.top_k
        )

        contexts_str = ""
        sources = []

        for i, chunk in enumerate(retrieved_chunks, start=1):
            text = chunk["text"]
            meta = chunk.get("metadata", {})

            contexts_str += f"[{i}] {text}\n\n"

            sources.append({
                "id": i,
                "pmid": int(meta.get("pmid")) if meta.get("pmid") else None,
                "title": meta.get("title", "")
            })

        answer = self.generator.generate(
            question=question,
            context=contexts_str
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }
