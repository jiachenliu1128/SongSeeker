import pandas as pd
import numpy as np
import yaml
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

import bm25_search
import w2v_genius
import bert_search

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
    bm25 = bm25_search.TextRetrieval()
    bm25.processed_docs = bm25.preprocess_docs(documents)
    bm25.build_vocabulary()
    bm25.build_doc_term_matrix()

    print("\n--- 3. Initializing Word2Vec ---")
    w2v = w2v_genius.TextRetrieval()
    w2v.dataset = pd.DataFrame({2: documents})
    w2v.load_embeddings()
    w2v.build_doc_W2V_cache()

    print("\n--- 4. Initializing BERT ---")
    bert = bert_search.BERTSearch()
    bert.encode_corpus(documents)

    print("\n--- 5. Generating Training Features (for all queries) ---")
    feature_list = []
    label_list = []

    for q_key, query_text in QUERIES.items():
        if q_key not in df.columns:
            print(f"  [Skip] Missing label column '{q_key}' in the dataset")
            continue
            
        print(f"  Processing Query: {q_key} ('{query_text[:30]}...')")
        
        s_bm25 = bm25.execute_search_BM25(query_text)
        
        s_w2v = w2v.execute_search_W2V(query_text, mode="avg_ll")
        
        s_bert = bert.execute_search(query_text)

        current_labels = df[q_key].values
        
        assert len(s_bm25) == len(documents)
        assert len(s_w2v) == len(documents)
        assert len(s_bert) == len(documents)

        for i in range(len(documents)):
            if not np.isnan(current_labels[i]):
                features = [s_bm25[i], s_w2v[i], s_bert[i]]
                feature_list.append(features)
                label_list.append(int(current_labels[i]))

    X = np.array(feature_list)
    y = np.array(label_list)
    
    print(f"\nFeature preparation complete. Total samples: {len(y)}")
    return X, y

def train_logreg(X, y):
    print(f"\n--- 6. Training Logistic Regression Model ---")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(y_train)}, Test set size: {len(y_test)}")

    clf = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
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
        possible_files = [f for f in os.listdir(data_dir) if "Labeled" in f and f.endswith(".csv")]
        
        if possible_files:
            labeled_path = os.path.join(data_dir, possible_files[0])
        else:
            labeled_path = config['data']['processed']
            print("[Warning] Could not automatically find 'Labeled' file, using default file from config.")

        X, y = prepare_features(labeled_path, config)
        
        if len(y) > 0:
            model, scaler = train_logreg(X, y)
            
            joblib.dump(model, "logreg_model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            print("\nModel and scaler saved to 'logreg_model.pkl' and 'scaler.pkl'")
        else:
            print("Error: No valid training samples generated. Please check if the CSV file contains q1...q10 label columns.")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Script execution failed: {e}")
        import traceback
        traceback.print_exc()
