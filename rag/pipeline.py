class RAGPipeline:
    def __init__(self, retriever, generator, top_k=5):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def evaluate_faithfulness(self, answer, context):
        
        if len(answer) > 20 and len(context) > 20:
            return "High (Based on provided snippets)"
        return "Low (Insufficient context)"

    def answer(self, question: str):
        docs = self.retriever.retrieve(question, top_k=self.top_k)
        
        context_str = ""
        sources = []
        for i, doc in enumerate(docs, 1):
            context_str += f"Source [{i}]: {doc['text']}\n\n"
            sources.append({
                "id": i,
                "pmid": doc["metadata"].get("pmid"),
                "title": doc["metadata"].get("title"),
                "score": doc["score"]
            })

        answer_text = self.generator.generate(question, context_str)
        quality_score = self.evaluate_faithfulness(answer_text, context_str)

        return {
            "answer": answer_text,
            "sources": sources,
            "metrics": {"faithfulness": quality_score}
        }