import os
from typing import List, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import yaml

with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
MODEL_NAME = config['bert']['model']


class SongBiEncoderSearcher:
    """
    Bi-encoder semantic search over a large song dataset.

    - Computes & saves a document embedding matrix (NumPy).
    - Optionally builds a FAISS index for fast top-k search.
    - Provides:
        * full_scores(query): np.ndarray of shape (N,)
          where scores[i] is the similarity to document i.
    """

    def __init__(
        self,
    ):
        self.model_name = MODEL_NAME
        self.model = SentenceTransformer(self.model_name)
        print(f"Model running on device: {self.model.device}")
        self.doc_embs: Optional[np.ndarray] = None  # (N, d)




    def build_from_csv(
        self,
        csv_path: str,
        text_cols: Optional[List[str]] = None,
        chunksize: int = 2048,
        batch_size: int = 64,
        sep: str = ",",
        encoding: str = "utf-8",
    ):
        if text_cols is None:
            text_cols = ["title", "artist", "lyrics"]

        print(f"Building embeddings from CSV: {csv_path}")

        # 1) Count total rows (lightweight, just metadata)
        total_rows = 0
        for df in pd.read_csv(csv_path, chunksize=chunksize, sep=sep, encoding=encoding):
            total_rows += len(df)
        print(f"  Total rows: {total_rows}")

        # 2) Get embedding dimension from a tiny probe
        probe_df = next(pd.read_csv(csv_path, chunksize=1, sep=sep, encoding=encoding))
        missing = [c for c in text_cols if c not in probe_df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        probe_text = probe_df[text_cols].fillna("").astype(str).apply(" - ".join, axis=1).tolist()
        probe_emb = self.model.encode(
            probe_text,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")
        dim = probe_emb.shape[1]

        # 3) Preallocate big embedding matrix
        doc_embs = np.empty((total_rows, dim), dtype="float32")

        # 4) Second pass: fill doc_embs chunk by chunk
        write_pos = 0
        current_index = 0
        for df in pd.read_csv(csv_path, chunksize=chunksize, sep=sep, encoding=encoding):
            print(f"  Processing rows {current_index} to {current_index + len(df) - 1}...")
            current_index += len(df)

            missing = [c for c in text_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in CSV: {missing}")

            text_df = df[text_cols].fillna("").astype(str)
            texts = text_df.apply(" - ".join, axis=1).tolist()

            embs = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype("float32")

            n = len(embs)
            doc_embs[write_pos : write_pos + n] = embs
            write_pos += n

        # 5) Normalize for cosine similarity (in-place)
        faiss.normalize_L2(doc_embs)

        self.doc_embs = doc_embs
        
        
        
    def save_cache(self, cache_dir: str):
        print(f"Saving cache to {cache_dir}...")
        os.makedirs(cache_dir, exist_ok=True)
        np.save(os.path.join(cache_dir, "bert_doc_embeddings.npy"), self.doc_embs)

    def load_cache(self, cache_dir: str):
        path = os.path.join(cache_dir, "bert_doc_embeddings.npy")
        if not os.path.exists(path):
            return False
        print(f"Loading cache from {cache_dir}...")
        self.doc_embs = np.load(path)
        print(f"{self.doc_embs.shape[0]} embeddings loaded from cache.")
        return True
            
            

        
        
        
        

    # ------------------------------------------------------------------
    # 1) Full scores: returns np.array of shape (N,)
    # ------------------------------------------------------------------
    def full_scores(self, query: str) -> np.ndarray:
        """
        Return cosine similarity scores for *every* document.

        Output:
            scores: np.ndarray of shape (N,)
                    scores[i] = similarity between query and document i
                               (same order as embeddings / song_ids)
        """
        # self._ensure_embs_loaded()

        # Encode query
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)

        # Matrix multiply: (1, d) @ (N, d)^T -> (1, N)
        scores = (q_emb @ self.doc_embs.T)[0]  # shape (N,)

        return scores
    
    
    
if __name__ == "__main__":
    
    # Paths
    data_path = "sample_data/processed/genius-clean-with-title-artist-10000.csv"
    
    # Example usage
    searcher = SongBiEncoderSearcher()
    searcher.build_from_csv(data_path)
    searcher.save_cache("cache/bert")
    
    query = "love and heartbreak"
    scores = searcher.full_scores(query)
    top5_indices = np.argsort(scores)[-5:][::-1]
    print(f"\n--- Top 5 Songs for query: '{query}' ---")
    for i in top5_indices:
        print(f"Score: {scores[i]:.4f} -> Index: {i}")