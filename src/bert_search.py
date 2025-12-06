import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss


class SongBiEncoderSearcher:
    """
    Bi-encoder semantic search over a large song dataset.

    Features:
    - Load a SentenceTransformer model.
    - Build embeddings from a large CSV in chunks.
    - Build & save/ load a FAISS index (cosine similarity).
    - Query with natural language and get top-k most relevant songs.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "./cache/song_index.faiss",
        id_path: str = "./cache/song_ids.npy",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: SentenceTransformer model name.
            index_path: Path to FAISS index file.
            id_path: Path to NumPy file storing song IDs in same order as index.
            device: Optional device for SentenceTransformer ("cpu", "cuda", etc.).
        """
        self.model_name = model_name
        self.index_path = index_path
        self.id_path = id_path

        self.model = SentenceTransformer(model_name, device=device)
        self.index: Optional[faiss.Index] = None
        self.song_ids: Optional[np.ndarray] = None




    # --------------------------
    # Building the index
    # --------------------------
    def build_index_from_csv(
        self,
        csv_path: str,
        id_col: str = "id",
        text_cols: Optional[List[str]] = None,
        chunksize: int = 2048,
        batch_size: int = 64,
        sep: str = ",",
        encoding: str = "utf-8",
    ):
        """
        Read a large CSV in chunks, compute embeddings, and build a FAISS index.

        Args:
            csv_path: Path to the CSV file.
            id_col: Column name containing a unique identifier for each song.
            text_cols: List of columns to concatenate as text input to the model.
                       If None, defaults to ["title", "artist", "lyrics"].
            chunksize: Number of rows per pandas chunk (controls RAM usage).
            batch_size: Batch size for SentenceTransformer encoding.
            sep: CSV separator.
            encoding: CSV encoding.
        """
        if text_cols is None:
            text_cols = ["title", "artist", "lyrics"]

        all_embs = []
        all_ids = []

        # 1. Loop over CSV chunks
        for df in pd.read_csv(csv_path, chunksize=chunksize, sep=sep, encoding=encoding):
            # Ensure required columns exist
            missing = [c for c in [id_col] + text_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in CSV: {missing}")

            # Build text to embed by concatenating selected columns
            text_df = df[text_cols].fillna("").astype(str)
            texts = text_df.apply(" - ".join, axis=1).tolist()
            ids = df[id_col].tolist()

            # 2. Encode chunk
            embs = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            ).astype("float32")

            all_embs.append(embs)
            all_ids.extend(ids)

        # 3. Stack all embeddings into one big matrix (N, d)
        all_embs = np.vstack(all_embs)   # float32
        all_ids = np.array(all_ids)

        # 4. Normalize for cosine similarity
        faiss.normalize_L2(all_embs)

        # 5. Build FAISS index (FlatIP = inner product)
        d = all_embs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(all_embs)

        # Save index and IDs
        faiss.write_index(index, self.index_path)
        np.save(self.id_path, all_ids)

        # Keep in memory for immediate use
        self.index = index
        self.song_ids = all_ids

    # --------------------------
    # Loading an existing index
    # --------------------------
    def load_index(self):
        """
        Load an existing FAISS index and song ID mapping from disk.
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        if not os.path.exists(self.id_path):
            raise FileNotFoundError(f"ID file not found: {self.id_path}")

        self.index = faiss.read_index(self.index_path)
        self.song_ids = np.load(self.id_path)

    # --------------------------
    # Querying
    # --------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
    ):
        """
        Encode a query and return top_k most similar songs.

        Returns:
            A list of dicts: [{"song_id": ..., "score": ...}, ...]
        """
        if self.index is None or self.song_ids is None:
            raise RuntimeError(
                "Index not loaded. Call build_index_from_csv(...) once, or load_index()."
            )

        # 1. Encode query
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)

        # 2. Search FAISS index
        scores, idxs = self.index.search(q_emb, top_k)
        scores = scores[0]
        idxs = idxs[0]

        # 3. Map back to IDs
        results = []
        for score, idx in zip(scores, idxs):
            if idx == -1:
                continue
            results.append(
                {
                    "song_id": int(self.song_ids[idx]),
                    "score": float(score),  # cosine similarity in [-1, 1]
                }
            )
        return results