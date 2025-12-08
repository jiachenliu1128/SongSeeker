import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add path to import sibling modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from search_bm25 import TextRetrieval
    from search_bert import SongBiEncoderSearcher
    from search_w2v import W2VSearcher
except ImportError as e:
    print(f"Module import failed: {e}")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_DATA_PATH = os.path.join(BASE_DIR, "sample_data", "processed", "Labeled_genius-clean-with-title-artist-5000.csv")

# Test Queries
QUERY_MAP = {
    'q1': "love and heartbreak",
    'q2': "party and dance",
    'q3': "lonely rain night",
    'q4': "quiet fog morning",
    'q5': "zero-gravity space exploration",
    'q6': "dancing in the club with friends until the sun comes up and forgetting all problems",
    'q7': "driving down the highway with windows down feeling free and wild",
    'q8': "hanging out with best friends making memories that will last forever and laughing together",
    'q9': "standing up against the world and fighting for what is right despite the odds",
    'q10': "looking into your eyes and realizing you are the only one I want to spend my life with"
}

# --- Utils ---
def min_max_normalize(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return np.zeros_like(series)
    return (series - min_val) / (max_val - min_val + 1e-9)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_metrics(predicted_indices, true_indices, k=5):
    if not true_indices:
        return 0.0
    top_k = predicted_indices[:k]
    hits = sum([1 for idx in top_k if idx in true_indices])
    return hits / k

def load_test_subset(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None, None, None

    # 1. Load FULL dataset
    print(f"Loading full dataset from: {csv_path}")
    df_full = pd.read_csv(csv_path, encoding='utf-8', lineterminator='\n')
    
    # 2. Slice the LAST 1000 rows (Test Set)
    df_test = df_full.iloc[4000:].reset_index(drop=True)
    print(f"Splitting data: Using last {len(df_test)} rows for EVALUATION.")

    # 3. Clean and Prep
    df_test.columns = df_test.columns.str.strip()
    df_test['lyrics'] = df_test['lyrics'].fillna("")

    # 4. Save temp file
    temp_test_path = csv_path.replace(".csv", "_temp_test_subset.csv")
    df_test.to_csv(temp_test_path, index=False)

    # 5. Extract Ground Truth
    ground_truth = {}
    for i in range(1, 11):
        key = f'q{i}'
        if key in df_test.columns:
            mask = (df_test[key] == 1) | (df_test[key] == '1')
            ground_truth[key] = set(df_test[mask].index)
        else:
            ground_truth[key] = set()

    return df_test, ground_truth, temp_test_path

# --- Main Pipeline ---
class SongSeekerPipeline:
    def __init__(self, dataframe, csv_path):
        self.df = dataframe.copy()
        
        # --- Try loading trained weights ---
        model_path = os.path.join(BASE_DIR, "models", "logreg_model.pkl")
        if os.path.exists(model_path):
            print(f"Loading trained model from {model_path}...")
            clf = joblib.load(model_path)
            self.weights = {
                'w1': clf.coef_[0][0], # BM25
                'w2': clf.coef_[0][1], # W2V
                'w3': clf.coef_[0][2], # BERT
                'b': clf.intercept_[0]
            }
            print(f"Using learned weights: {self.weights}")
        else:
            print("No trained model found. Using default weights.")
            self.weights = {'w1': 0.3, 'w2': 0.2, 'w3': 0.5, 'b': -2.0}

        # 1. Init BM25
        print("Initializing BM25 on Test Set...")
        self.bm25 = TextRetrieval()
        self.bm25.processed_docs = self.bm25.preprocess_docs(self.df['lyrics'].tolist())
        self.bm25.build_vocabulary()
        self.bm25.build_doc_term_matrix()

        # 2. Init BERT
        print("Initializing BERT on Test Set...")
        # === 修复点：移除不支持的参数 ===
        # 旧版 search_bert.py 不支持 emb_path/id_path 参数，所以我们不传
        self.bert = SongBiEncoderSearcher() 
        
        # 直接从 CSV 构建（这会把向量存在内存 self.doc_embs 中，不需要存盘也能跑）
        self.bert.build_from_csv(csv_path)
        
        # 3. Init W2V
        print("Initializing Word2Vec on Test Set...")
        try:
            self.w2v = W2VSearcher(self.df)
        except NameError:
            print("W2V module missing, skipping.")
            self.w2v = None

    def search(self, query):
        res = self.df.copy()

        # Get scores
        try: res['s_bm25'] = self.bm25.execute_search_BM25(query)
        except: res['s_bm25'] = 0.0

        try: res['s_bert'] = self.bert.full_scores(query)
        except: res['s_bert'] = 0.0

        if self.w2v:
            try:
                w2v_out = self.w2v.search(query)
                res['s_w2v'] = w2v_out['score'] if isinstance(w2v_out, pd.DataFrame) else w2v_out
            except: res['s_w2v'] = 0.0
        else:
            res['s_w2v'] = 0.0

        res.fillna(0.0, inplace=True)

        # Normalize
        res['n_bm25'] = min_max_normalize(res['s_bm25'])
        res['n_bert'] = min_max_normalize(res['s_bert'])
        res['n_w2v'] = min_max_normalize(res['s_w2v'])

        # Calculate Hybrid Score
        w1, w2, w3, b = self.weights['w1'], self.weights['w2'], self.weights['w3'], self.weights['b']
        z = (w1 * res['n_bm25'] + w3 * res['n_bert'] + w2 * res['n_w2v'] + b)
        res['final_score'] = sigmoid(z)

        return res.sort_values('final_score', ascending=False)

if __name__ == "__main__":
    df_test, gt_dict, temp_csv_path = load_test_subset(FULL_DATA_PATH)

    if df_test is not None:
        pipeline = SongSeekerPipeline(df_test, csv_path=temp_csv_path)
        precisions = []

        print("\n=== Running Evaluation on Test Set (1000 songs) ===")
        for q_id, query in QUERY_MAP.items():
            print(f"\nProcessing {q_id}...")
            
            try:
                results = pipeline.search(query)
                true_ids = gt_dict.get(q_id, set())

                if not true_ids:
                    print("  -> No ground truth in test set, skipping.")
                    continue

                # Metrics
                pred_indices = results.index[:5].tolist()
                p_at_5 = calculate_metrics(pred_indices, true_ids, k=5)
                precisions.append(p_at_5)

                top_track = results.iloc[0]
                print(f"  Query: {query}")
                print(f"  Top Result: {top_track['title']} (Score: {top_track['final_score']:.4f})")
                print(f"  Precision@5: {p_at_5:.2f}")

            except Exception as e:
                print(f"  Error processing {q_id}: {e}")

        if precisions:
            avg_map = sum(precisions) / len(precisions)
            print(f"\n=== Final Results (Test Set) ===")
            print(f"MAP (Hybrid): {avg_map:.4f}")
            
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)