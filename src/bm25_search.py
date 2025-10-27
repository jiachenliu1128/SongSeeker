import numpy as np
import pandas as pd
import re
from collections import Counter
import math
from nltk.corpus import stopwords
import nltk
import yaml

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

manual_stopwords = set(stopwords.words('english'))

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
        for doc in docs:
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
        all_words = [word for doc in self.processed_docs for word in doc]
        self.vocab = sorted(list(set(all_words)))

    def build_doc_term_matrix(self):
        matrix = []
        for doc in self.processed_docs:
            doc_counts = Counter(doc)
            term_counts = [doc_counts.get(word, 0) for word in self.vocab]
            matrix.append(term_counts)
        self.doc_term_matrix = np.array(matrix)

    def execute_search_BM25(self, query, k1=1.5, b=0.75):
        query_tokens = self.preprocess_docs([query])[0]
        doc_lengths = self.doc_term_matrix.sum(axis=1)
        avg_doc_length = np.mean(doc_lengths)
        num_docs = len(self.doc_term_matrix)
        scores = np.zeros(num_docs)

        for term in query_tokens:
            if term in self.vocab:
                term_idx = self.vocab.index(term)
                df = np.count_nonzero(self.doc_term_matrix[:, term_idx])
                idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
                tf = self.doc_term_matrix[:, term_idx]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_lengths / avg_doc_length))
                scores += idf * (numerator / denominator)
        return scores

def search_songs(query, tr, dataset):
    relevance_docs = tr.execute_search_BM25(query)
    top5_indices = np.argsort(relevance_docs)[-5:][::-1]

    print(f"\n--- Top 5 Songs for query: '{query}' ---")
    for i in top5_indices:
        print(f"Score: {relevance_docs[i]:.2f} -> Title: {dataset.iloc[i]['title']}")


if __name__ == "__main__":
    try:
        # Load configuration
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        data_path = config['data']['processed']

        dataset = pd.read_csv(data_path)

        tr = TextRetrieval()

        print("Preprocessing the dataset...")
        tr.processed_docs = tr.preprocess_docs(dataset['lyrics'])

        print("Building vocabulary and document-term matrix...")
        tr.build_vocabulary()
        tr.build_doc_term_matrix()
        print("Setup complete. You can now start searching.")

        while True:
            query = input("\nEnter your search query (or type 'quit' to exit): ")
            if query.lower() == 'quit':
                print("Exiting...")
                break
            search_songs(query, tr, dataset)

    except FileNotFoundError:
        print(f"\n[ERROR] The file '{data_path}' was not found.")
        print("Please make sure your CSV file is in the right directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
