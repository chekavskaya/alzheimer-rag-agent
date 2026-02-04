import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class Retriever:
 
    def __init__(
        self,
        data_path: str,
        text_column: str = "chunk_text",
        metadata_columns: List[str] | None = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.data_path = data_path
        self.text_column = text_column
        self.metadata_columns = metadata_columns or []
        self.embedding_model_name = embedding_model_name

        self.model = SentenceTransformer(self.embedding_model_name)
        self.index = None
        self.embeddings = None
        self.df = None

        self._load_data()
        self._build_index()

    def _load_data(self):
        
        self.df = pd.read_csv(self.data_path)

        
        if self.text_column not in self.df.columns:
            if "text" in self.df.columns:
                print("Using 'text' column instead of 'chunk_text'")
                self.text_column = "text"
            else:
                raise ValueError(
                    f"Text column not found. Available columns: {list(self.df.columns)}"
                )

        self.texts = self.df[self.text_column].astype(str).tolist()

    def _build_index(self):
        
        print("Computing embeddings...")
        self.embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        print(f"FAISS index built with {self.index.ntotal} vectors")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            row = self.df.iloc[idx]

            # Корректное извлечение метаданных
            metadata = {}
            for col in self.metadata_columns:
                if col in self.df.columns:
                    metadata[col] = row[col]

            results.append(
                {
                    "text": row[self.text_column],
                    "score": float(score),
                    "metadata": metadata,
                }
            )

        return results
