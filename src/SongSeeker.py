#!/usr/bin/env python
"""
Rank documents for a free-text query using the learned ranking model
(LogisticRegression trained in learn_to_rank.py).

Assumes:
- BM25, W2V, and BERT search modules are available as:
    import search_bm25, search_w2v, search_bert
- The logistic regression model and scaler are saved as:
    models/logreg_model.pkl
    models/scaler.pkl
- The same CSV (or at least same corpus) used for training is available.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib

from . import search_bm25
from . import search_w2v
from . import search_bert



class LearnedRanker:
    def __init__(self, csv_path: str, model_dir: str = "models"):
        """
        Initialize BM25 / W2V / BERT searchers and load the trained LTR model.

        Parameters
        ----------
        csv_path : str
            Path to the labeled (or at least same corpus) CSV file.
        model_dir : str
            Directory containing logreg_model.pkl and scaler.pkl.
        """
        self.csv_path = csv_path
        self.model_dir = model_dir

        print(f"[Init] Loading corpus from: {csv_path}")
        df = pd.read_csv(csv_path)
        self.df = df

        # --- Choose the same text column logic as in learn_to_rank.prepare_features ---
        text_col = "lyrics"
        if "lyrics" not in df.columns:
            if "text" in df.columns:
                text_col = "text"
            else:
                text_col = df.select_dtypes(include=["object"]).columns[-1]

        print(f"[Init] Using text column: '{text_col}'")
        self.text_col = text_col
        self.documents = df[text_col].fillna("").tolist()

        # --- Initialize BM25 ---
        print("[Init] Building BM25 cache...")
        self.bm25 = search_bm25.TextRetrieval()
        if not self.bm25.load_cache("cache/bm25"):
            self.bm25.processed_docs = self.bm25.preprocess_docs(self.documents)
            self.bm25.build_vocabulary()
            self.bm25.build_doc_term_matrix()
            self.bm25.save_cache("cache/bm25")
        else:
            print("[Init] BM25 cache loaded.")

        # --- Initialize W2V ---
        print("[Init] Building Word2Vec cache (cosine mode only)...")
        self.w2v = search_w2v.TextRetrieval()
        self.w2v.dataset = pd.DataFrame({2: self.documents})
        if not self.w2v.load_cache("cache/w2v", expect_full_mats=False):
            self.w2v.load_embeddings()
            self.w2v.build_doc_W2V_cache(max_doc_tokens=200, keep_full_mats=False)
            self.w2v.save_cache("cache/w2v")
        else:
            print("[Init] W2V cache loaded.")

        # --- Initialize BERT ---
        print("[Init] Initializing BERT bi-encoder...")
        self.bert = search_bert.SongBiEncoderSearcher()
        if not self.bert.load_cache("cache/bert"):
            self.bert.build_from_csv(csv_path, text_cols=[self.text_col])
            self.bert.save_cache("cache/bert")
        else:
            print("[Init] BERT cache loaded.")

        # --- Load trained logistic regression model & scaler ---
        model_path = os.path.join(model_dir, "logreg_model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        print(f"[Init] Loading model from: {model_path}")
        print(f"[Init] Loading scaler from: {scaler_path}")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        print("[Init] LearnedRanker ready.")

    # ------------------------------------------------------------------
    # internal helper: build features for a *single* query across corpus
    # ------------------------------------------------------------------
    def _scores_for_query(self, query: str) -> np.ndarray:
        """
        Compute [BM25, W2V, BERT] feature matrix for this query over all docs.

        Returns
        -------
        X : np.ndarray of shape (n_docs, 3)
            Each row is [s_bm25_i, s_w2v_i, s_bert_i].
        """
        n_docs = len(self.documents)
        print(f"[Query] Computing BM25/W2V/BERT scores for {n_docs} docs...")

        # BM25
        s_bm25 = self.bm25.execute_search_BM25(query)  # shape (n_docs,)

        # W2V cosine
        s_w2v = self.w2v.execute_search_W2V(query, mode="cosine")  # shape (n_docs,)

        # BERT scores: mirror logic from learn_to_rank.prepare_features
        if hasattr(self.bert, "full_scores"):
            s_bert = self.bert.full_scores(query)
        elif hasattr(self.bert, "execute_search"):
            s_bert = self.bert.execute_search(query)
        else:
            s_bert = np.zeros(n_docs, dtype=np.float32)

        # Make sure lengths match (handle any padding/trunc as in training)
        if len(s_bm25) != n_docs:
            raise RuntimeError(f"BM25 scores length mismatch: {len(s_bm25)} vs {n_docs}")
        if len(s_w2v) != n_docs:
            raise RuntimeError(f"W2V scores length mismatch: {len(s_w2v)} vs {n_docs}")
        if len(s_bert) != n_docs:
            print(
                f"[Warning] BERT scores length mismatch ({len(s_bert)} vs {n_docs}). "
                "Padding/Truncating with np.resize (same as training)."
            )
            s_bert = np.resize(s_bert, n_docs)

        # Feature matrix
        X = np.column_stack([s_bm25, s_w2v, s_bert]).astype(np.float32)
        return X

    # ------------------------------------------------------------------
    def rank(self, query: str, top_k: int = 10):
        """
        Rank all documents for the given query using the learned LTR model.

        Parameters
        ----------
        query : str
            User query string.
        top_k : int
            Number of top docs to return.

        Returns
        -------
        result_df : pd.DataFrame
            DataFrame with columns:
            - 'rank' : 1-based rank
            - 'score': model score (p(y=1))
            - plus a few meta columns from the original CSV
        """
        print(f"\n[Rank] Query: {query!r}")
        X = self._scores_for_query(query)

        # Apply the same scaling as during training
        X_scaled = self.scaler.transform(X)

        # Use predicted probability of the "relevant" class as ranking score
        scores = self.model.predict_proba(X_scaled)[:, 1]  # shape (n_docs,)

        # Sort documents by descending score
        order = np.argsort(-scores)
        top_k = min(top_k, len(order))
        top_idx = order[:top_k]

        # Build a result DataFrame with some meta info
        meta_cols = []
        for c in ["title", "artist", "lyrics", "text"]:
            if c in self.df.columns:
                meta_cols.append(c)

        result = self.df.iloc[top_idx][meta_cols].copy()
        result.insert(0, "score", scores[top_idx])
        result.insert(0, "rank", np.arange(1, top_k + 1))

        return result.reset_index(drop=True)


# ----------------------------------------------------------------------
# Simple CLI interface
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Rank documents using learned LTR model (BM25 + W2V + BERT)."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top documents to display.",
    )
    args = parser.parse_args()

    data_path = "sample_data/processed/genius-clean-with-title-artist-10000.csv"
    ranker = LearnedRanker(data_path)

    print("\nType a query to get ranked results (Ctrl+C or empty line to exit).")
    while True:
        try:
            q = input("\nQuery> ").strip()
            if not q:
                print("Empty query, exiting.")
                break

            result = ranker.rank(q, top_k=args.top_k)
            # pretty-print top results
            with pd.option_context("display.max_colwidth", 80):
                print(result)

        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()