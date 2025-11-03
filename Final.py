import numpy as np
import math

docs = [
    "apple launches new iphone with advanced camera",
    "samsung unveils latest galaxy smartphone",
    "microsoft introduces new ai features in office",
    "google announces ai powered search tools",
    "apple and samsung continue smartphone rivalry"
]

vocab = sorted(set(" ".join(docs).split()))
tf = np.array([[doc.split().count(w) for w in vocab] for doc in docs])
idf = np.log(len(docs) / np.count_nonzero(tf, axis=0))
tfidf = tf * idf

def kmeans(X, k=2, max_iter=100):
    # Randomly initialize centroids
    np.random.seed(42)
    indices = np.random.choice(len(X), k, replace=False)
    centroids = X[indices]

    for _ in range(max_iter):
        # Assign clusters
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Stop if centroids don't change
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids

labels, centroids = kmeans(tfidf, k=2)
for i, (doc, label) in enumerate(zip(docs, labels)):
    print(f"Cluster {label}: {doc}")

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 1️⃣ Sample Documents & Labels
# ---------------------------
docs = [
    "The football team won the match",
    "He scored a goal in the championship",
    "New AI technology is transforming industry",
    "Machine learning enables smart systems"
]
labels = np.array(["sports", "sports", "tech", "tech"])

test_docs = [
    "The player scored two goals",
    "AI improves computer systems"
]

# --------------------------------------------------
# 2️⃣ ROCCHIO CLASSIFIER (Centroid-based)
# --------------------------------------------------
def rocchio_train(X, y):
    classes = np.unique(y)
    centroids = {}
    for c in classes:
        centroids[c] = X[y == c].mean(axis=0)
    return centroids

def rocchio_predict(X, centroids):
    preds = []
    for x in X:
        sims = {c: cosine_similarity([x], [centroid])[0, 0] for c, centroid in centroids.items()}
        preds.append(max(sims, key=sims.get))
    return preds

print("=== ROCCHIO CLASSIFIER ===")
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(docs).toarray()
X_test_tfidf = tfidf_vectorizer.transform(test_docs).toarray()

rocchio_centroids = rocchio_train(X_tfidf, labels)
rocchio_preds = rocchio_predict(X_test_tfidf, rocchio_centroids)
for doc, pred in zip(test_docs, rocchio_preds):
    print(f"[{pred}] {doc}")

# --------------------------------------------------
# 3️⃣ MULTINOMIAL NAIVE BAYES (Word Frequency)
# --------------------------------------------------
def train_multinomial_nb(X, y, alpha=1.0):
    classes = np.unique(y)
    n_features = X.shape[1]
    class_priors = {}
    cond_probs = {}
    for c in classes:
        X_c = X[y == c]
        class_priors[c] = X_c.shape[0] / X.shape[0]
        word_counts = X_c.sum(axis=0)
        cond_probs[c] = (word_counts + alpha) / (word_counts.sum() + alpha * n_features)
    return class_priors, cond_probs

def predict_multinomial_nb(X, class_priors, cond_probs):
    preds = []
    for x in X:
        scores = {}
        for c in class_priors:
            log_prob = np.log(class_priors[c]) + np.sum(x * np.log(cond_probs[c]))
            scores[c] = log_prob
        preds.append(max(scores, key=scores.get))
    return preds

print("\n=== MULTINOMIAL NAIVE BAYES ===")
count_vectorizer = CountVectorizer(stop_words='english')
X_count = count_vectorizer.fit_transform(docs).toarray()
X_test_count = count_vectorizer.transform(test_docs).toarray()

priors_m, cond_probs_m = train_multinomial_nb(X_count, labels)
preds_m = predict_multinomial_nb(X_test_count, priors_m, cond_probs_m)
for doc, pred in zip(test_docs, preds_m):
    print(f"[{pred}] {doc}")

# --------------------------------------------------
# 4️⃣ MULTIVARIATE BERNOULLI NAIVE BAYES (Word Presence)
# --------------------------------------------------
def train_bernoulli_nb(X, y, alpha=1.0):
    classes = np.unique(y)
    n_features = X.shape[1]
    class_priors = {}
    cond_probs = {}
    for c in classes:
        X_c = X[y == c]
        class_priors[c] = X_c.shape[0] / X.shape[0]
        word_presence = X_c.sum(axis=0)
        cond_probs[c] = (word_presence + alpha) / (X_c.shape[0] + 2 * alpha)
    return class_priors, cond_probs

def predict_bernoulli_nb(X, class_priors, cond_probs):
    preds = []
    for x in X:
        scores = {}
        for c in class_priors:
            p = cond_probs[c]
            log_prob = np.log(class_priors[c]) + np.sum(x * np.log(p) + (1 - x) * np.log(1 - p))
            scores[c] = log_prob
        preds.append(max(scores, key=scores.get))
    return preds

print("\n=== BERNOULLI NAIVE BAYES ===")
X_bin = Binarizer().fit_transform(X_count)
X_test_bin = Binarizer().fit_transform(X_test_count)

priors_b, cond_probs_b = train_bernoulli_nb(X_bin, labels)
preds_b = predict_bernoulli_nb(X_test_bin, priors_b, cond_probs_b)
for doc, pred in zip(test_docs, preds_b):
    print(f"[{pred}] {doc}")


#content based
import numpy as np

# Example items (documents/movies)
items = [
    "Action movie with car chase and explosions",
    "Romantic comedy about love and friendship",
    "Thriller with mystery and suspense",
    "Action-packed superhero adventure",
    "Emotional romantic drama"
]

# Step 1: Manual tokenization
tokenized = [i.lower().split() for i in items]
vocab = sorted(set(word for doc in tokenized for word in doc))

# Step 2: Compute simple TF vectors (no IDF for simplicity)
def tf_vector(doc_tokens):
    return np.array([doc_tokens.count(w) / len(doc_tokens) for w in vocab])

tf_vectors = np.array([tf_vector(doc) for doc in tokenized])

# Step 3: Manual cosine similarity
def cosine_sim(a, b):
    num = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return num / denom if denom != 0 else 0

# Step 4: Recommend similar items to a given one
def recommend_content_based(index, top_n=2):
    sims = [(i, cosine_sim(tf_vectors[index], tf_vectors[i])) for i in range(len(items)) if i != index]
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return [items[i] for i, _ in sims[:top_n]]

print("Item:", items[0])
print("Recommended:", recommend_content_based(0))


#collaborative
import numpy as np

# Rows = users, Columns = items
# 0 means user hasn’t rated that item
ratings = np.array([
    [5, 3, 0, 1],   # User 1
    [4, 0, 0, 1],   # User 2
    [1, 1, 0, 5],   # User 3
    [0, 0, 5, 4],   # User 4
    [0, 1, 5, 4],   # User 5
])

#User-Based
def cosine_similarity_manual(a, b):
    num = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return num / denom if denom != 0 else 0

def predict_user_based(user_index, item_index):
    target_user = ratings[user_index]
    sims, weighted_sum, sim_sum = [], 0, 0

    for other_user in range(len(ratings)):
        if other_user != user_index and ratings[other_user][item_index] > 0:
            sim = cosine_similarity_manual(target_user, ratings[other_user])
            sims.append(sim)
            weighted_sum += sim * ratings[other_user][item_index]
            sim_sum += abs(sim)

    return weighted_sum / sim_sum if sim_sum != 0 else 0

# Example: predict what User 1 would rate Item 3
predicted_rating = predict_user_based(0, 2)
print("Predicted rating (User-based):", round(predicted_rating, 2))

#Item-Based
# Transpose matrix → columns = users, rows = items
ratings_T = ratings.T

def predict_item_based(user_index, item_index):
    target_item = ratings_T[item_index]
    sims, weighted_sum, sim_sum = [], 0, 0

    for other_item in range(len(ratings_T)):
        if other_item != item_index and ratings[user_index][other_item] > 0:
            sim = cosine_similarity_manual(target_item, ratings_T[other_item])
            sims.append(sim)
            weighted_sum += sim * ratings[user_index][other_item]
            sim_sum += abs(sim)

    return weighted_sum / sim_sum if sim_sum != 0 else 0

# Example: predict what User 1 would rate Item 3
predicted_rating_item = predict_item_based(0, 2)
print("Predicted rating (Item-based):", round(predicted_rating_item, 2))

