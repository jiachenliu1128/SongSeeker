import numpy as np
import pandas as pd
from collections import Counter
import math
from nltk.corpus import stopwords
import nltk
import pickle
import scipy.sparse as sp
import os

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
manual_stopwords = set(stopwords.words('english'))




# Text Retrieval class implementing BM25
class TextRetrieval():
    punctuations = ""
    stop_words = set()
    vocab = []
    dataset = None
    processed_docs = []
    doc_term_matrix = None

    def __init__(self):
        self.punctuations = "\'\"\\,<>./?@#$%^&*_~/!()-[]{};:"
        self.stop_words = manual_stopwords

    def preprocess_docs(self, docs):
        processed = []
        doc_count = 0
        for doc in docs:
            print(f"Processing document {doc_count+1}/{len(docs)}", end='\r')
            doc_count += 1
            if isinstance(doc, str):
                doc = doc.strip().lower()
                doc = ''.join(c for c in doc if c not in self.punctuations)
                tokens = doc.split()
                tokens = [word for word in tokens if word not in self.stop_words]
                processed.append(tokens)
            else:
                processed.append([])
        return processed

    def build_vocabulary(self):
        print("Building vocabulary...                        ", end='\r')
        all_words = [word for doc in self.processed_docs for word in doc]
        self.vocab = sorted(list(set(all_words)))
        # Fast term -> index lookup
        self.vocab_index = {term: i for i, term in enumerate(self.vocab)}

    def build_doc_term_matrix(self):
        """
        Build a sparse document-term matrix.

        Old version was O(num_docs * vocab_size) because it iterated over the
        full vocabulary for every document. This version is O(total_tokens)
        and stores counts in a CSR sparse matrix.
        """
        rows = []
        cols = []
        data = []

        for doc_idx, doc in enumerate(self.processed_docs):
            print(
                f"Building doc-term matrix for document {doc_idx+1}/{len(self.processed_docs)}",
                end='\r',
            )
            doc_counts = Counter(doc)
            for term, count in doc_counts.items():
                idx = self.vocab_index.get(term)
                if idx is None:
                    continue
                rows.append(doc_idx)
                cols.append(idx)
                data.append(count)

        n_docs = len(self.processed_docs)
        n_terms = len(self.vocab)
        self.doc_term_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_docs, n_terms), dtype=np.int32
        )

    def execute_search_BM25(self, query, k1=1.5, b=0.75):
        """
        Compute BM25 scores for a query using the sparse doc-term matrix.
        """
        query_tokens = self.preprocess_docs([query])[0]

        # doc_lengths: (n_docs,) dense vector
        doc_lengths = np.asarray(self.doc_term_matrix.sum(axis=1)).ravel()
        avg_doc_length = float(doc_lengths.mean()) if len(doc_lengths) > 0 else 0.0
        num_docs = self.doc_term_matrix.shape[0]
        scores = np.zeros(num_docs, dtype=np.float64)

        for term in query_tokens:
            term_idx = self.vocab_index.get(term)
            if term_idx is None:
                continue

            # tf is a sparse column; convert to dense 1D
            tf_col = self.doc_term_matrix[:, term_idx].toarray().ravel()
            df = np.count_nonzero(tf_col)
            if df == 0:
                continue

            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf_col * (k1 + 1.0)
            denominator = tf_col + k1 * (1.0 - b + b * (doc_lengths / (avg_doc_length + 1e-9)))
            scores += idf * (numerator / (denominator + 1e-9))

        return scores
    
    def save_cache(self, cache_dir: str):
        print(f"Saving cache to {cache_dir}...                                ")
        os.makedirs(cache_dir, exist_ok=True)
        # tokens, vocab, doc lengths, etc.
        with open(os.path.join(cache_dir, "bm25_meta.pkl"), "wb") as f:
            pickle.dump(
                {
                    "processed_docs": self.processed_docs,
                    "vocab": self.vocab,
                    "vocab_index": self.vocab_index,
                },
                f,
            )
        sp.save_npz(
            os.path.join(cache_dir, "bm25_doc_term.npz"),
            self.doc_term_matrix,
        )

    def load_cache(self, cache_dir: str):
        meta_path = os.path.join(cache_dir, "bm25_meta.pkl")
        dtm_path = os.path.join(cache_dir, "bm25_doc_term.npz")
        if not (os.path.exists(meta_path) and os.path.exists(dtm_path)):
            return False  # cache miss

        print(f"Cache found. Loading from {cache_dir}...                        ")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
            print(f"{len(meta)} items loaded from cache.")
            print(f"Document count: {len(meta['processed_docs'])}")
        self.processed_docs = meta["processed_docs"]
        self.vocab = meta["vocab"]
        self.vocab_index = meta["vocab_index"]

        self.doc_term_matrix = sp.load_npz(dtm_path)
        return True  # cache hit




def search_songs(query, tr, dataset):
    relevance_docs = tr.execute_search_BM25(query)
    top5_indices = np.argsort(relevance_docs)[-5:][::-1]

    print(f"\n--- Top 5 Songs for query: '{query}' ---")
    for i in top5_indices:
        print(f"Score: {relevance_docs[i]:.2f} -> Title: {dataset.iloc[i]['title']}")





if __name__ == "__main__":
    try:
        # data_path
        data_path = "sample_data/processed/genius-clean-with-title-artist-10000.csv"
        
        # Load dataset
        dataset = pd.read_csv(data_path)
        tr = TextRetrieval()

        print("Preprocessing the dataset...")
        tr.processed_docs = tr.preprocess_docs(dataset['lyrics'])

        print("Building vocabulary and document-term matrix...")
        tr.build_vocabulary()
        tr.build_doc_term_matrix()
        tr.save_cache("cache/bm25")
        print("Setup complete. You can now start searching.")

    except FileNotFoundError:
        print(f"\n[ERROR] The file '{data_path}' was not found.")
        print("Please make sure your CSV file is in the right directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
