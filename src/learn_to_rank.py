import pandas as pd
import numpy as np
import yaml
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import search_bm25
import search_w2v
import search_bert

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

def prepare_features(labeled_data_path, config):
    print(f"--- 1. Data Loading ---")
    print(f"Reading file: {labeled_data_path}")
    df = pd.read_csv(labeled_data_path)
    
    text_col = 'lyrics'
    if 'lyrics' not in df.columns:
        if 'text' in df.columns:
            text_col = 'text'
        else:
            text_col = df.select_dtypes(include=['object']).columns[-1]
    
    print(f"Using text column: '{text_col}'")
    documents = df[text_col].fillna("").tolist()

    print("\n--- 2. Initializing BM25 ---")
    # Using the search_bm25 module
    bm25 = search_bm25.TextRetrieval()
    bm25.processed_docs = bm25.preprocess_docs(documents)
    bm25.build_vocabulary()
    bm25.build_doc_term_matrix()

    print("\n--- 3. Initializing Word2Vec ---")
    # Using the search_w2v module
    w2v = search_w2v.TextRetrieval()
    w2v.dataset = pd.DataFrame({2: documents})
    w2v.load_embeddings()
    w2v.build_doc_W2V_cache()

    print("\n--- 4. Initializing BERT ---")
    # Using the search_bert module
    bert = search_bert.SongBiEncoderSearcher()
    bert.build_from_csv(labeled_data_path)

    print("\n--- 5. Generating Training Features (for all queries) ---")
    feature_list = []
    label_list = []

    for q_key, query_text in QUERIES.items():
        if q_key not in df.columns:
            print(f"  [Skip] Missing label column '{q_key}' in the dataset")
            continue
            
        print(f"  Processing Query: {q_key} ('{query_text[:30]}...')")
        
        s_bm25 = bm25.execute_search_BM25(query_text)
        
        # [OPTIMIZATION 1] Core optimization: Enforce 'cosine' mode for W2V
        # The calling logic must be the new one, otherwise the result won't be good
        s_w2v = w2v.execute_search_W2V(query_text, mode="cosine")
        
        s_bert = bert.full_scores(query_text)

        current_labels = df[q_key].values
        
        assert len(s_bm25) == len(documents)
        assert len(s_w2v) == len(documents)
        assert len(s_bert) == len(documents)

        for i in range(len(documents)):
            if not np.isnan(current_labels[i]):
                # Feature concatenation: BM25 + W2V + BERT
                features = [s_bm25[i], s_w2v[i], s_bert[i]]
                feature_list.append(features)
                label_list.append(int(current_labels[i]))

    X = np.array(feature_list)
    y = np.array(label_list)
    
    print(f"\nFeature preparation complete. Total samples: {len(y)}")
    return X, y

def train_logreg(X, y):
    print(f"\n--- 6. Training Logistic Regression Model (with Optimization) ---")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(y_train)}, Test set size: {len(y_test)}")

    # [OPTIMIZATION 2] Core optimization: Use GridSearchCV for automatic hyperparameter tuning
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

    y_pred = clf.predict(X_test)

    print("\n--- Model Evaluation (Test Set) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score (Binary): {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n--- Feature Weights (Feature Importances) ---")
    print(f"w1 (BM25): {clf.coef_[0][0]:.4f}")
    print(f"w2 (W2V) : {clf.coef_[0][1]:.4f}")
    print(f"w3 (BERT): {clf.coef_[0][2]:.4f}")
    print(f"Bias     : {clf.intercept_[0]:.4f}")
    
    return clf, scaler

if __name__ == "__main__":
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        data_dir = os.path.dirname(config['data']['processed'])
        
        # [OPTIMIZATION 3] Pin the target file to prevent reading the wrong one
        target_filename = "Labeled_genius-clean-with-title-artist-5000.csv"
        labeled_path = os.path.join(data_dir, target_filename)
        
        if not os.path.exists(labeled_path):
            print(f"[Warning] Targeted file '{target_filename}' not found at {labeled_path}.")
            # Compatibility fallback
            possible_files = [f for f in os.listdir(data_dir) if "Labeled" in f and f.endswith(".csv")]
            if possible_files:
                labeled_path = os.path.join(data_dir, possible_files[0])
            else:
                labeled_path = config['data']['processed']
        
        print(f"Target Data Path: {labeled_path}")

        X, y = prepare_features(labeled_path, config)
        
        if len(y) > 0:
            model, scaler = train_logreg(X, y)
            
            joblib.dump(model, "logreg_model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            print("\nModel and scaler saved to 'logreg_model.pkl' and 'scaler.pkl'")
        else:
            print("Error: No valid training samples generated.")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Script execution failed: {e}")
        import traceback
        traceback.print_exc()