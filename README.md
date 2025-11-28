# SongSeeker
Team 15: Jiachen Liu (jl315), Joey Cai (qcai20), Tong Tong (tong37), Jiachen Liang (liang88)

# Queries
1. love and heartbreak
2. party and dance
3. lonely rain night 
4. quiet fog morning
5. zero-gravity space exploration
6. dancing in the club with friends until the sun comes up and forgetting all problems
7. driving down the highway with windows down feeling free and wild
8. hanging out with best friends making memories that will last forever and laughing together
9. standing up against the world and fighting for what is right despite the odds
10. looking into your eyes and realizing you are the only one I want to spend my life with


# Role division
1. BERT embeddings -> script to get BERT similarity score -> Jiachen Liu
2. Data labeling by LLM -> if taking subset, choose data randomly - Joey
3. Logistic regression model -> need input features from three models and labeled data to train parameters. -> Tong
4. Pipeline integration (output and input alignment for each part) & evaluation -> evaluation: query get top k results (rank of songs) and compare with labeled data to calculate Precision@k, Recall@k, and NDCG@kmeasures - Loen


query -> Logistic Regression -> Ranking of songs:
1. song 1 -> relevent 1
2. song 2 -> relevent 0
3. song 3 -> relevent 1




# Logistic Regression Script
sigma(z) = 1 / (1 + exp(-z)) -> (range: 0 to 1)
z = w1*x1 + w2*x2 + w3*x3 + b

sci-kit-learn

logreg_train()
Input: 
    for each (query, song) pair, we have three features:
    x1: BM25 score from bm25_search.py
    x2: Word2Vec similarity score
    x3: BERT similarity score
    z: labeled relevance (0 or 1)
Output: 
    learned weights w1, w2, w3, and bias b
    we need these to compute final relevance score for evaluation

logreg_predict()
Input:
    for each (query, song) pair, we have three features:
    x1: BM25 score from bm25_search.py
    x2: Word2Vec similarity score
    x3: BERT similarity score
    learned weights w1, w2, w3, and bias b from logreg_train()
Output:
    predicted relevance score (between 0 and 1) for each (query, song) pair



