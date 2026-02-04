from rag.retrieval import Retriever

retriever = Retriever(
    data_path="data/pubmed_chunks.csv",
    text_column="text",
    metadata_columns=["pmid", "title"]
)

results = retriever.retrieve("potential therapeutic targets for Alzheimer's disease", top_k=3)

for r in results:
    print("SCORE:", r["score"])
    print("TEXT:", r["text"][:300])  
    meta = r.get("metadata", {})
    meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items()) if meta else "N/A"
    print("META:", meta_str)
    print("-" * 50)
