import pandas as pd
import numpy as np
import os
import sys
import joblib
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Ensure sibling imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from . import search_bm25
    from . import search_w2v
    from . import search_bert
except ImportError as e:
    print(f"Error importing search modules: {e}")

QUERIES = {
    "q1": "love and heartbreak",
    "q2": "party and dance",
    "q3": "lonely rain night",
    "q4": "quiet fog morning",
    "q5": "zero-gravity space exploration",
    "q6": "dancing in the club with friends until the sun comes up and forgetting all problems",
    "q7": "driving down the highway with windows down feeling free and wild",
    "q8": "hanging out with best friends making memories that will last forever and laughing together",
    "q9": "standing up against the world and fighting for what is right despite the odds",
    "q10": "looking into your eyes and realizing you are the only one I want to spend my life with"
}

# --- Feature Extraction (Compatible with Temp Files) ---
def prepare_features(labeled_data_path):
    print(f"--- 1. Data Loading (Training Subset) ---")
    print(f"Reading file: {labeled_data_path}")
    
    # Read CSV (handle potential line terminators)
    df = pd.read_csv(labeled_data_path, encoding='utf-8', lineterminator='\n')
    df.columns = df.columns.str.strip()
    
    text_col = 'lyrics'
    if 'lyrics' not in df.columns:
        text_col = 'text' if 'text' in df.columns else df.select_dtypes(include=['object']).columns[-1]
    
    print(f"Using text column: '{text_col}'")
    documents = df[text_col].fillna("").tolist()

    print("\n--- 2. Initializing BM25 ---")
    bm25 = search_bm25.TextRetrieval()
    bm25.processed_docs = bm25.preprocess_docs(documents)
    bm25.build_vocabulary()
    bm25.build_doc_term_matrix()

    print("\n--- 3. Initializing Word2Vec ---")
    # Adapter check: Try to instantiate the correct class
    if hasattr(search_w2v, 'W2VSearcher'):
        w2v = search_w2v.W2VSearcher(df) 
    else:
        # Fallback
        w2v = search_w2v.TextRetrieval()
        w2v.dataset = pd.DataFrame({2: documents}) 
        w2v.load_embeddings()
        w2v.build_doc_W2V_cache()

    print("\n--- 4. Initializing BERT ---")
    bert = search_bert.SongBiEncoderSearcher()
    # Force build from the specific training CSV subset to align indices
    bert.build_from_csv(labeled_data_path)

    print("\n--- 5. Generating Training Features ---")
    feature_list = []
    label_list = []

    for q_key, query_text in QUERIES.items():
        if q_key not in df.columns:
            print(f"  [Skip] Missing label column '{q_key}'")
            continue
            
        print(f"  Processing Query: {q_key}")
        
        # 1. BM25 Score
        s_bm25 = bm25.execute_search_BM25(query_text)
        
        # 2. W2V Score
        if hasattr(w2v, 'search'):
            s_w2v = w2v.search(query_text)
        else:
            s_w2v = w2v.execute_search_W2V(query_text, mode="cosine")
        
        # 3. BERT Score
        if hasattr(bert, 'full_scores'):
            s_bert = bert.full_scores(query_text)
        else:
            s_bert = np.zeros(len(documents))

        # Handle Data Types (ensure numpy array)
        if isinstance(s_w2v, pd.DataFrame) or isinstance(s_w2v, pd.Series):
             s_w2v = s_w2v.values if hasattr(s_w2v, 'values') else np.array(s_w2v)

        current_labels = df[q_key].values
        
        if len(s_bert) != len(documents):
             s_bert = np.resize(s_bert, len(documents))

        for i in range(len(documents)):
            if not np.isnan(current_labels[i]):
                features = [s_bm25[i], s_w2v[i], s_bert[i]]
                feature_list.append(features)
                label_list.append(int(current_labels[i]))

    X = np.array(feature_list)
    y = np.array(label_list)
    
    print(f"\nFeature preparation complete. Total training samples: {len(y)}")
    return X, y

# --- Statistical Helpers (Original Rigorous Version) ---
def calculate_pvalues_and_se(model, X):
    p = model.predict_proba(X)[:, 1]
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    V = p * (1 - p)
    H = np.dot(X_design.T * V, X_design)
    
    try:
        cov_matrix = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        print("[Warning] Hessian matrix is singular. Cannot calculate SE or p-values.")
        n_features_plus_one = X_design.shape[1]
        return [np.nan] * n_features_plus_one, [np.nan] * n_features_plus_one
    
    std_errors = np.sqrt(np.diag(cov_matrix))
    params = np.insert(model.coef_.flatten(), 0, model.intercept_)
    z_scores = params / (std_errors + 1e-10)
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    
    return p_values, std_errors

# --- Training Logic (Original Rigorous Version) ---
def train_logreg(X, y):
    print(f"\n--- 6. Training Logistic Regression Model (with Optimization) ---")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Internal validation split (on the 4000 samples)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(y_train)}, Test set size: {len(y_test)}")

    print("Running GridSearchCV to find best hyperparameters...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'class_weight': ['balanced', None]
    }
    
    base_clf = LogisticRegression(solver='lbfgs', max_iter=2000)
    grid_search = GridSearchCV(base_clf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    clf = grid_search.best_estimator_
    print(f"\nBest Parameters found: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

    print("Calculating statistical significance (p-values) and Standard Errors...")
    p_values, std_errors = calculate_pvalues_and_se(clf, X_train)
    
    p_bias, se_bias = p_values[0], std_errors[0]
    p_w1, se_w1 = (p_values[1], std_errors[1]) if len(p_values) > 1 else (np.nan, np.nan)
    p_w2, se_w2 = (p_values[2], std_errors[2]) if len(p_values) > 2 else (np.nan, np.nan)
    p_w3, se_w3 = (p_values[3], std_errors[3]) if len(p_values) > 3 else (np.nan, np.nan)
    
    y_pred = clf.predict(X_test)

    print("\n--- Model Evaluation (Test Set) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score (Binary): {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n--- Feature Weights & Statistical Significance ---")
    print(f"{'Feature':<12} | {'Weight (Beta)':<15} | {'Std. Error':<15} | {'P-Value':<15} | {'Sig.'}")
    print("-" * 72)
    
    def get_sig_star(p):
        if np.isnan(p): return "N/A"
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return ""

    print(f"{'Bias':<12} | {clf.intercept_[0]:<15.4f} | {se_bias:<15.4f} | {p_bias:<15.10f} | {get_sig_star(p_bias)}")
    print(f"{'w1 (BM25)':<12} | {clf.coef_[0][0]:<15.4f} | {se_w1:<15.4f} | {p_w1:<15.10f} | {get_sig_star(p_w1)}")
    print(f"{'w2 (W2V)':<12} | {clf.coef_[0][1]:<15.4f} | {se_w2:<15.4f} | {p_w2:<15.10f} | {get_sig_star(p_w2)}")
    print(f"{'w3 (BERT)':<12} | {clf.coef_[0][2]:<15.4f} | {se_w3:<15.4f} | {p_w3:<15.10f} | {get_sig_star(p_w3)}")
    print("-" * 72)
    print("Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05")

    return clf, scaler

# --- Main Execution (Data Splitting Logic) ---
if __name__ == "__main__":
    try:
        # 1. Locate Full Data
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        PROCESSED_DIR = os.path.join(BASE_DIR, "sample_data", "processed")
        FULL_FILE = "Labeled_genius-clean-with-title-artist-5000.csv"
        FULL_PATH = os.path.join(PROCESSED_DIR, FULL_FILE)

        if not os.path.exists(FULL_PATH):
            print(f"[Error] Data file not found at {FULL_PATH}")
            sys.exit(1)

        # 2. Slice First 4000 Rows
        print(f"Loading full dataset from: {FULL_PATH}")
        df_full = pd.read_csv(FULL_PATH, encoding='utf-8', lineterminator='\n')
        
        # SLICING LOGIC: Use first 4000 rows for training
        df_train = df_full.iloc[:4000].copy()
        print(f"Splitting data: Using first {len(df_train)} rows for TRAINING.")
        
        # 3. Save as Temp File (Critical for BERT/W2V index alignment)
        temp_train_path = os.path.join(PROCESSED_DIR, "temp_train_subset.csv")
        df_train.to_csv(temp_train_path, index=False)
        print(f"Saved temp training file to: {temp_train_path}")

        # 4. Run Pipeline
        X, y = prepare_features(temp_train_path)
        
        if len(y) > 0:
            model, scaler = train_logreg(X, y)
            
            # 5. Save Model
            target_dir = os.path.join(BASE_DIR, "models")
            os.makedirs(target_dir, exist_ok=True)
            
            joblib.dump(model, os.path.join(target_dir, "logreg_model.pkl"))
            joblib.dump(scaler, os.path.join(target_dir, "scaler.pkl"))
            
            print(f"\nModel and scaler saved to '{target_dir}'")
            
            # Cleanup
            if os.path.exists(temp_train_path):
                os.remove(temp_train_path)
                print("Temp file cleaned up.")
        else:
            print("Error: No training samples found.")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()