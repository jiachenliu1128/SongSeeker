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
import pickle
from nltk.corpus import stopwords
import yaml
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


# Read config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)
W2V_MODEL = config['w2v']['model']

# Data paths
DATA_PATH = "data/processed/clean-with-title-artist-1000000.csv"


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

    def build_doc_W2V_cache(self, max_doc_tokens=200, keep_full_mats: bool = False):
        """Cache per-document mean vectors, and optionally full matrices.

        Parameters
        ----------
        max_doc_tokens : int
            Maximum number of in-vocabulary tokens per document to use when
            computing the cache. If > 0, documents are truncated to at most
            this many tokens.
        keep_full_mats : bool
            If True, store the full (m_i, dim) matrix for each document in
            ``self.docs_vecs`` so that avg-LL scoring can be used.
            If False (recommended for large corpora), only ``self.docs_mean``
            is populated and ``self.docs_vecs`` is set to None to save memory.
        """
        docs = self.dataset[2].tolist()
        self.docs_tokens = [str(d).split() for d in docs]

        # Initialize caches
        self.docs_mean = np.zeros((len(docs), self.dim), dtype=np.float32)
        self.docs_vecs = [] if keep_full_mats else None

        for i, tokens in enumerate(self.docs_tokens):
            print(f"W2V processing doc {i+1}/{len(docs)}", end='\r')
            in_vocab = [w for w in tokens if w in self.kv.key_to_index]
            if max_doc_tokens and len(in_vocab) > max_doc_tokens:
                in_vocab = in_vocab[:max_doc_tokens]

            if not in_vocab:
                # docs_mean row stays zeros; optionally store an empty matrix
                if keep_full_mats:
                    self.docs_vecs.append(np.zeros((0, self.dim), dtype=np.float32))
            else:
                mat = np.vstack([self.kv.get_vector(w) for w in in_vocab]).astype(np.float32)
                if keep_full_mats:
                    self.docs_vecs.append(mat)
                self.docs_mean[i] = mat.mean(axis=0)

        if keep_full_mats:
            non_empty = sum(1 for m in self.docs_vecs if m.shape[0] > 0)
            print(f"[cache] docs={len(docs)}, dim={self.dim}, non_empty={non_empty}")
        else:
            print(f"[cache] docs={len(docs)}, dim={self.dim}, full mats disabled (keep_full_mats=False)")

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
        """Compute retrieval scores for all documents.

        Parameters
        ----------
        query : str
            Raw query string.
        mode : {"avg_ll", "cosine"}
            Scoring mode. "cosine" uses only ``self.docs_mean`` (memory-light).
            "avg_ll" additionally requires ``self.docs_vecs`` to be populated
            (set ``keep_full_mats=True`` when building the cache).
        """
        n = self.dataset.shape[0]
        if mode == "cosine":
            return self.w2v_cosine_scores_batch(query)

        if self.docs_vecs is None:
            raise RuntimeError(
                "avg_ll mode requested but docs_vecs is None. "
                "Rebuild the cache with keep_full_mats=True, or use mode='cosine'."
            )

        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            scores[i] = self.w2v_avgll_score(query, i)
        return scores
    
    
    def save_cache(self, cache_dir: str):
        print(f"[W2V] Saving cache to {cache_dir}...")
        os.makedirs(cache_dir, exist_ok=True)
        # mean vectors
        np.save(os.path.join(cache_dir, "w2v_docs_mean.npy"), self.docs_mean)
        # tokens (optional but often useful)
        with open(os.path.join(cache_dir, "w2v_docs_tokens.pkl"), "wb") as f:
            pickle.dump(self.docs_tokens, f)
        # full matrices (only if you used keep_full_mats=True)
        if self.docs_vecs is not None:
            with open(os.path.join(cache_dir, "w2v_docs_vecs.pkl"), "wb") as f:
                pickle.dump(self.docs_vecs, f)

    def load_cache(self, cache_dir: str, expect_full_mats: bool = False):
        mean_path = os.path.join(cache_dir, "w2v_docs_mean.npy")
        tokens_path = os.path.join(cache_dir, "w2v_docs_tokens.pkl")
        vecs_path = os.path.join(cache_dir, "w2v_docs_vecs.pkl")

        # Check existence
        if not (os.path.exists(mean_path) and os.path.exists(tokens_path)):
            return False

        # Load mean vectors and tokens
        self.docs_mean = np.load(mean_path)
        with open(tokens_path, "rb") as f:
            self.docs_tokens = pickle.load(f)

        # Load full matrices if expected
        if expect_full_mats:
            if not os.path.exists(vecs_path):
                return False
            with open(vecs_path, "rb") as f:
                self.docs_vecs = pickle.load(f)
        else:
            self.docs_vecs = None

        return True
    

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
    # For large corpora, avoid storing full per-document matrices to save memory.
    # If you really need avg-LL scoring, call with keep_full_mats=True instead.
    tr.build_doc_W2V_cache(max_doc_tokens=200, keep_full_mats=False)
    tr.save_cache("cache/w2v")

    # 4) Run some demo queries (lyrics-y)
    queries = [
        "love heartbreak",
        "party dance floor",
        "rain city night lonely"
    ]

    # print("#########\nResults for W2V (avg_log_likelihood)")
    # for q in queries:
    #     print("QUERY:", q)
    #     scores = tr.execute_search_W2V(q, mode="avg_ll")
    #     print_top_bottom_with_meta(scores, tr.meta)
    #     print()

    print("#########\nResults for W2V (cosine baseline)")
    for q in queries:
        print("QUERY:", q)
        scores = tr.execute_search_W2V(q, mode="cosine")
        print_top_bottom_with_meta(scores, tr.meta)
        print()

    print("[done] w2v_genius finished.")


# Adapter Class for Evaluate System (Required for Pipeline)
class W2VSearcher(TextRetrieval):
    def __init__(self, dataframe):
        """Initialize adapter with pipeline data."""
        super().__init__()

        # Prepare Data
        df = dataframe.copy()
        if 'lyrics' not in df.columns:
            print("[W2V] Warning: Missing 'lyrics' column.")
            df['lyrics'] = ""

        punct = self.punctuations
        sw = self.stop_words
        df[2] = df['lyrics'].astype(str).apply(lambda t: clean_text(t, punct, sw))

        self.dataset = df

        # Init Model & Cache
        self.load_embeddings()
        # Ensure we build the cache so searching works immediately
        self.build_doc_W2V_cache(max_doc_tokens=200, keep_full_mats=False)

    def search(self, query):
        """Return batch cosine similarity scores."""
        # This calls the method from the parent TextRetrieval class
        return self.w2v_cosine_scores_batch(query)

if __name__ == "__main__":
    # Check if run as a script directly
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        main_convert()
    elif len(sys.argv) > 1 and sys.argv[1] == "adapter_test":
        print("W2VSearcher adapter is defined.")
    else:
        main_search()