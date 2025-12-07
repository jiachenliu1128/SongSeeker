# -*- coding: utf-8 -*-
"""
Word2Vec-based text retrieval over Genius lyrics dataset.
ALSO includes a utility to convert GloVe to Word2Vec format.

Default usage (search):
    python w2v_genius.py

Conversion usage:
    python w2v_genius.py convert glove.6B.100d.txt glove.6B.100d.word2vec.txt
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import yaml
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


# Read config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)
DATA_PATH = config['data']['processed']
W2V_MODEL = config['w2v']['model']


# =========================
# Config: embeddings source
# =========================
# If you have a local file, set this path. Examples:
#   - GoogleNews-vectors-negative300.bin (binary=True)
#   - glove.6B.100d.word2vec.txt (binary=False)  # after conversion
LOCAL_VEC_PATH = None       # e.g., "embeddings/GoogleNews-vectors-negative300.bin"
LOCAL_VEC_BINARY = True
ONLINE_MODEL_NAME = W2V_MODEL   # used only if LOCAL_VEC_PATH is None

# =========================
# Helper functions
# =========================
def ensure_stopwords():
    """Make sure NLTK stopwords are available."""
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

def clean_text(s: str, punctuations: str, sw: set) -> str:
    """
    lower, remove HTML, remove numbers, remove punctuation,
    collapse whitespaces, remove stopwords
    """
    s = str(s).strip().lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = s.translate(str.maketrans({c: " " for c in punctuations}))
    s = re.sub(r"\s+", " ", s).strip()
    toks = [w for w in s.split() if w and w not in sw]
    return " ".join(toks)

def sigmoid(x):
    # Numerical stability
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))

# =========================
# Retrieval class
# =========================
class TextRetrieval():
    # Preprocessing
    punctuations = ""
    stop_words = set()

    # Dataset
    dataset = None      # pd.DataFrame with column index 2 for cleaned text
    meta = None         # dict with "title", "artist" lists for pretty print

    # Embeddings
    kv = None           # gensim KeyedVectors
    dim = None

    # Caches
    docs_tokens = None  # list[list[str]]
    docs_vecs = None    # list[np.ndarray], shape=(m_i, dim)
    docs_mean = None    # np.ndarray, shape=(N, dim)

    # Scoring
    alpha = 1.0         # scaling for avg-LL

    def __init__(self):
        ensure_stopwords()
        # include common ASCII + curly quotes/dashes
        self.punctuations = "\"\\,<>./?@#$%^&*_~/!()-[]{};:’'`“”–—"
        self.stop_words = set(stopwords.words("english"))
        self.dataset = None
        self.meta = None
        self.alpha = 1.0

    # ---------- Data IO ----------
    def read_and_preprocess_genius(self, csv_path: str):
        """
        Read local genius-clean-with-title-artist-5000.csv and build:
        - dataset: with column index 2 as cleaned text (to be compatible with prior code)
        - meta:    title/artist lists for reporting
        """
        df = pd.read_csv(csv_path)

        # tolerant column autodetection
        lower_cols = {c.lower(): c for c in df.columns}
        title_col  = lower_cols.get("title")
        artist_col = lower_cols.get("artist")
        # lyrics/text/content: try common names; fallback to the last column
        lyrics_col = (lower_cols.get("lyrics")
                        or lower_cols.get("text")
                        or lower_cols.get("content")
                        or list(df.columns)[-1])

        # Build concatenated raw text: [title] + [artist] + [lyrics]
        pieces = []
        if title_col  is not None:  pieces.append(df[title_col].astype(str))
        if artist_col is not None:  pieces.append(df[artist_col].astype(str))
        pieces.append(df[lyrics_col].astype(str))

        df["raw_concat"] = ""
        for p in pieces:
            df["raw_concat"] = (df["raw_concat"] + " " + p.fillna("").astype(str)).str.strip()

        # Cleaned text
        punct = self.punctuations
        sw = self.stop_words
        df["text"] = df["raw_concat"].apply(lambda t: clean_text(t, punct, sw))

        # Keep a simple (0,1,2) frame to reuse downstream logic
        self.dataset = pd.DataFrame({0: 0, 1: 0, 2: df["text"]})

        # Keep meta for pretty print
        self.meta = {
            "title": df[title_col].astype(str).tolist() if title_col else [None]*len(df),
            "artist": df[artist_col].astype(str).tolist() if artist_col else [None]*len(df)
        }
        print(f"[genius] loaded rows={len(df)}; cleaned text ready.")

    # ---------- Embeddings ----------
    def load_embeddings(self):
        """
        Prefer local vectors; fallback to gensim.downloader online model.
        """
        try:
            if LOCAL_VEC_PATH:
                print(f"[W2V] loading local vectors: {LOCAL_VEC_PATH} (binary={LOCAL_VEC_BINARY})")
                self.kv = KeyedVectors.load_word2vec_format(LOCAL_VEC_PATH, binary=LOCAL_VEC_BINARY)
            else:
                import gensim.downloader as api
                print(f"[W2V] loading online model via gensim.downloader: {ONLINE_MODEL_NAME}")
                self.kv = api.load(ONLINE_MODEL_NAME)
        except Exception as e:
            print(f"[W2V] failed to load embeddings: {e}", file=sys.stderr)
            raise
        self.dim = int(getattr(self.kv, "vector_size", self.kv.vector_size))
        print(f"[W2V] dim={self.dim}, |vocab|≈{len(self.kv.key_to_index):,}")

    # ---------- Vectorization ----------
    def text2W2VMatrix(self, text):
        """
        Convert text (or list of tokens) to a (m, dim) matrix with in-vocab words only.
        Returns empty (0,dim) if none in vocab.
        """
        if isinstance(text, list):
            tokens = text
        else:
            tokens = str(text).split()

        in_vocab = [w for w in tokens if w in self.kv.key_to_index]
        if not in_vocab:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = np.vstack([self.kv.get_vector(w) for w in in_vocab]).astype(np.float32)
        return mat

    def build_doc_W2V_cache(self, max_doc_tokens=200):
        """
        Cache doc matrices and mean vectors.
        Fix duplicate append bug: do NOT append into self.docs_tokens again in loop.
        """
        docs = self.dataset[2].tolist()
        self.docs_tokens = [str(d).split() for d in docs]
        self.docs_vecs = []
        self.docs_mean = np.zeros((len(docs), self.dim), dtype=np.float32)

        for i, tokens in enumerate(self.docs_tokens):
            in_vocab = [w for w in tokens if w in self.kv.key_to_index]
            if max_doc_tokens and len(in_vocab) > max_doc_tokens:
                in_vocab = in_vocab[:max_doc_tokens]

            if not in_vocab:
                self.docs_vecs.append(np.zeros((0, self.dim), dtype=np.float32))
            else:
                mat = np.vstack([self.kv.get_vector(w) for w in in_vocab]).astype(np.float32)
                self.docs_vecs.append(mat)
                self.docs_mean[i] = mat.mean(axis=0)

        non_empty = sum(1 for m in self.docs_vecs if m.shape[0] > 0)
        print(f"[cache] docs={len(docs)}, dim={self.dim}, non_empty={non_empty}")

    # ---------- Scoring ----------
    def _clean_query(self, q: str) -> str:
        return clean_text(q, self.punctuations, self.stop_words)

    def w2v_avgll_score(self, query: str, doc_idx: int) -> float:
        """
        Average log-likelihood over pairwise dot products with sigmoid.
        """
        q_clean = self._clean_query(query)
        Q = self.text2W2VMatrix(q_clean)
        if Q.shape[0] == 0:
            return 0.0
        D = self.docs_vecs[doc_idx]
        if D.shape[0] == 0:
            return -1e10
        dot_matrix = np.matmul(Q, D.T)
        prob_matrix = sigmoid(self.alpha * dot_matrix)
        log_prob_matrix = np.log(prob_matrix + 1e-12)
        return float(log_prob_matrix.mean())

    def w2v_cosine_scores_batch(self, query: str) -> np.ndarray:
        """
        Vectorized cosine similarity for all docs.
        """
        q_clean = self._clean_query(query)
        Q = self.text2W2VMatrix(q_clean)
        if Q.shape[0] == 0:
            return np.zeros(self.dataset.shape[0], dtype=np.float32)

        q_mean = Q.mean(axis=0)
        q_norm = np.linalg.norm(q_mean) + 1e-12

        D = self.docs_mean  # (N, dim)
        d_norms = np.linalg.norm(D, axis=1) + 1e-12
        dots = D @ q_mean  # (N,)
        sims = dots / (d_norms * q_norm)
        return sims.astype(np.float32)

    def execute_search_W2V(self, query: str, mode: str = "avg_ll") -> np.ndarray:
        """
        mode in {"avg_ll", "cosine"}
        """
        n = self.dataset.shape[0]
        if mode == "cosine":
            return self.w2v_cosine_scores_batch(query)
        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            scores[i] = self.w2v_avgll_score(query, i)
        return scores

# ---------- Pretty printing ----------
def print_top_bottom_with_meta(scores: np.ndarray, meta: dict, k: int = 5):
    idx_desc = np.argsort(-scores)
    idx_asc = np.argsort(scores)

    def row(i):
        title = meta.get("title")[i] if meta and meta.get("title") else None
        artist = meta.get("artist")[i] if meta and meta.get("artist") else None
        tag = ""
        if title and title.strip() and title != "None":
            tag += f"{title}"
        if artist and artist.strip() and artist != "None":
            tag += f" — {artist}"
        if not tag:
            tag = f"doc_{i}"
        return f"{tag} | score={scores[i]:.6f}"

    print("Top-5:")
    for i in idx_desc[:k]:
        print(" ", row(i))
    print("Bottom-5:")
    for i in idx_asc[:k]:
        print(" ", row(i))

# ---------- Main ----------

def main_convert():
    """Wrapper for GloVe conversion utility."""
    if len(sys.argv) != 4: # script.py convert <in> <out>
        print("Usage: python w2v_genius.py convert <glove_input.txt> <word2vec_output.txt>")
        sys.exit(1)
    glove_input_file = sys.argv[2]
    word2vec_output_file = sys.argv[3]
    try:
        glove2word2vec(glove_input_file, word2vec_output_file)
        print(f"[ok] converted {glove_input_file} -> {word2vec_output_file}")
    except Exception as e:
        print(f"[error] conversion failed: {e}", file=sys.stderr)
        sys.exit(1)

def main_search():
    """Main logic for running text retrieval."""
    # Instantiate
    tr = TextRetrieval()

    # 1) Load and clean local Genius CSV
    tr.read_and_preprocess_genius(DATA_PATH)
    print(f"[info] num_docs = {tr.dataset.shape[0]}")

    # 2) Load word embeddings
    tr.load_embeddings()
    tr.alpha = 1.0  # try 1.0~3.0 for avg-LL sharpness

    # 3) Build per-document cache
    tr.build_doc_W2V_cache(max_doc_tokens=200)

    # 4) Run some demo queries (lyrics-y)
    queries = [
        "love heartbreak",
        "party dance floor",
        "rain city night lonely"
    ]

    print("#########\nResults for W2V (avg_log_likelihood)")
    for q in queries:
        print("QUERY:", q)
        scores = tr.execute_search_W2V(q, mode="avg_ll")
        print_top_bottom_with_meta(scores, tr.meta)
        print()

    print("#########\nResults for W2V (cosine baseline)")
    for q in queries:
        print("QUERY:", q)
        scores = tr.execute_search_W2V(q, mode="cosine")
        print_top_bottom_with_meta(scores, tr.meta)
        print()

    print("[done] w2v_genius finished.")


if __name__ == "__main__":
    # Check if the user wants to run the 'convert' utility
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        main_convert()
    else:
        # Otherwise, run the default 'search' main
        main_search()