import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class Retriever:
    def __init__(
        self,
        data_path: str,
        text_column: str = "text",
        metadata_columns: List[str] = None,
        
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.data_path = data_path
        self.text_column = text_column
        self.metadata_columns = metadata_columns or ["pmid", "title"]
        self.model = SentenceTransformer(embedding_model_name)
        
        self._load_and_prepare()

    def _load_and_prepare(self):
        self.df = pd.read_csv(self.data_path)
        
        self.df[self.text_column] = self.df[self.text_column].fillna("Empty snippet")
        texts = self.df[self.text_column].astype(str).tolist()
        
       
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
       
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            row = self.df.iloc[idx]
            results.append({
                "text": row[self.text_column],
                "score": float(score),
                "metadata": {col: row[col] for col in self.metadata_columns if col in row}
            })
        return results