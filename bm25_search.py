import numpy as np
import pandas as pd
import re
from collections import Counter
import math

manual_stopwords = set(['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])

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

try:
    dataset = pd.read_csv('clean-with-title-artist-all.csv')

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
    print("\n[ERROR] The file 'clean-with-title-artist-all.csv' was not found.")
    print("Please make sure the script and the CSV file are in the same directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
