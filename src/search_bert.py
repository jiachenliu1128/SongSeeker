import os
from typing import List, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import yaml


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
        model_name: str = "all-MiniLM-L6-v2",
        emb_path: str = "./cache/bert_song_embeddings.npy",
        id_path: str = "./cache/bert_song_ids.npy"
    ):
        self.model_name = model_name
        self.emb_path = emb_path
        self.id_path = id_path

        self.model = SentenceTransformer(model_name)
        print(f"Model running on device: {self.model.device}")

        # Lazy-loaded
        self.doc_embs: Optional[np.ndarray] = None  # (N, d)
        self.song_ids: Optional[np.ndarray] = None  # (N,)
        self.index: Optional[faiss.Index] = None





    # ------------------------------------------------------------------
    # Build embeddings (and optionally FAISS index) from CSV
    # ------------------------------------------------------------------
    def build_from_csv(
        self,
        csv_path: str,
        text_cols: Optional[List[str]] = None,
        chunksize: int = 2048,
        batch_size: int = 64,
        sep: str = ",",
        encoding: str = "utf-8",
    ):
        """
        Build embeddings from CSV.
        If id_col is None, use row index as document ID.
        """
        if text_cols is None:
            text_cols = ["title", "artist", "lyrics"]

        all_embs = []
        all_ids = []

        current_index = 0  # running row index across chunks

        print(f"Building embeddings from CSV: {csv_path}")
        for df in pd.read_csv(csv_path, chunksize=chunksize, sep=sep, encoding=encoding):
            print(f"  Processing rows {current_index} to {current_index + len(df) - 1}...")
            # Check text columns
            missing = [c for c in text_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in CSV: {missing}")

            # Text for embedding
            text_df = df[text_cols].fillna("").astype(str)
            texts = text_df.apply(" - ".join, axis=1).tolist()
            
            # IDs
            ids = list(range(current_index, current_index + len(df)))
            current_index += len(df)

            # Encode lyrics/title/artist
            embs = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype("float32")

            all_embs.append(embs)
            all_ids.extend(ids)

        # Finish stacking
        doc_embs = np.vstack(all_embs).astype("float32")
        song_ids = np.array(all_ids)

        # Normalize for cosine similarity
        faiss.normalize_L2(doc_embs)

        # Save to disk
        np.save(self.emb_path, doc_embs)
        np.save(self.id_path, song_ids)

        # Store in memory
        self.doc_embs = doc_embs
        self.song_ids = song_ids
            
            
            
            

    # ------------------------------------------------------------------
    # Loading stuff back
    # ------------------------------------------------------------------
    def _ensure_embs_loaded(self):
        if self.doc_embs is None:
            if not os.path.exists(self.emb_path):
                raise FileNotFoundError(f"Embedding file not found: {self.emb_path}")
            self.doc_embs = np.load(self.emb_path)
        if self.song_ids is None:
            if not os.path.exists(self.id_path):
                raise FileNotFoundError(f"ID file not found: {self.id_path}")
            self.song_ids = np.load(self.id_path)
        
        
        
        

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
        self._ensure_embs_loaded()

        # Encode query
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)

        # Matrix multiply: (1, d) @ (N, d)^T -> (1, N)
        scores = (q_emb @ self.doc_embs.T)[0]  # shape (N,)

        return scores
    
    
    
if __name__ == "__main__":
    # Load configuration
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    data_path = config['data']['processed']
    
    # Example usage
    searcher = SongBiEncoderSearcher()
    searcher.build_from_csv(data_path)
    
    query = "love and heartbreak"
    scores = searcher.full_scores(query)
    top5_indices = np.argsort(scores)[-5:][::-1]
    print(f"\n--- Top 5 Songs for query: '{query}' ---")
    for i in top5_indices:
        print(f"Score: {scores[i]:.4f} -> Index: {i}")