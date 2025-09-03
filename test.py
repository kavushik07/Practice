#-------------------------------
# Term-Document Boolean Model
# -------------------------------

# Sample documents
docs = [
    "information retrieval system",
    "database search query",
    "information system database",
    "web search engine",
    "query processing system"
]

# Step 1: Build Vocabulary
processed_docs = [doc.lower().split() for doc in docs]
vocab = sorted(set(term for doc in processed_docs for term in doc))

# Step 2: Build Term-Document Matrix (rows=terms, cols=docs)
term_doc_matrix = []
for term in vocab:
    row = []
    for tokens in processed_docs:
        row.append(1 if term in tokens else 0)
    term_doc_matrix.append(row)

# Function: Get vector of a term
def get_term_vector(term):
    term = term.lower()
    if term not in vocab:
        return [0] * len(docs)
    idx = vocab.index(term)
    return term_doc_matrix[idx]

# Function: Boolean Search
def boolean_search(query):
    query = query.lower().strip()
    
    # Single term
    if " and " not in query and " or " not in query and " not " not in query:
        return [i for i, val in enumerate(get_term_vector(query)) if val == 1]
    
    # AND
    if " and " in query:
        terms = [t.strip() for t in query.split(" and ")]
        result = get_term_vector(terms[0])
        for term in terms[1:]:
            vec = get_term_vector(term)
            result = [a & b for a, b in zip(result, vec)]
        return [i for i, val in enumerate(result) if val == 1]
    
    # OR
    if " or " in query:
        terms = [t.strip() for t in query.split(" or ")]
        result = get_term_vector(terms[0])
        for term in terms[1:]:
            vec = get_term_vector(term)
            result = [a | b for a, b in zip(result, vec)]
        return [i for i, val in enumerate(result) if val == 1]
    
    # NOT
    if " not " in query:
        pos, neg = [t.strip() for t in query.split(" not ")]
        pos_vec = get_term_vector(pos)
        neg_vec = get_term_vector(neg)
        neg_vec = [1 - x for x in neg_vec]
        result = [a & b for a, b in zip(pos_vec, neg_vec)]
        return [i for i, val in enumerate(result) if val == 1]
    
    return []

# -------------------------------
# Print Term-Document Matrix
# -------------------------------
print("Term-Document Matrix:")
print("Terms\\Docs", end="")
for i in range(len(docs)):
    print(f"\tD{i}", end="")
print()

for i, term in enumerate(vocab):
    print(f"{term:<10}", end="")
    for val in term_doc_matrix[i]:
        print(f"\t{val}", end="")
    print()

# -------------------------------
# Example Queries
# -------------------------------
print("\nSearch Results:")
print("'information':", boolean_search("information"))
print("'information and system':", boolean_search("information and system"))
print("'search or query':", boolean_search("search or query"))
print("'system not database':", boolean_search("system not database"))


"""
Binary Independence Model (BIM) - Simple Version (Non-OOP)
"""

import math

# -------------------------------
# Documents
# -------------------------------
docs = [
    "information retrieval system",
    "database search query", 
    "information system database",
    "web search engine",
    "query processing system"
]

# -------------------------------
# Preprocessing: Vocabulary + Binary Matrix
# -------------------------------
processed_docs = [set(doc.lower().split()) for doc in docs]
vocab = sorted(set(term for doc in processed_docs for term in doc))
N_d = len(docs)  # total number of documents

# Binary term-document matrix
binary_matrix = []
for tokens in processed_docs:
    row = [1 if term in tokens else 0 for term in vocab]
    binary_matrix.append(row)


# -------------------------------
# Phase I Estimates (No relevance info)
# -------------------------------
def phase1_estimate(query_terms):
    estimates = {}
    for term in query_terms:
        if term not in vocab:
            continue
        idx = vocab.index(term)

        d_k = sum(doc[idx] for doc in binary_matrix)  # doc freq
        p_k = 0.5
        q_k = (d_k + 0.5) / (N_d + 1)  # smoothed

        estimates[term] = {"d_k": d_k, "p_k": p_k, "q_k": q_k}
    return estimates


# -------------------------------
# Phase II Estimates (With relevance info)
# -------------------------------
def phase2_estimate(query_terms, relevant_docs):
    estimates = {}
    N_r = len(relevant_docs)

    for term in query_terms:
        if term not in vocab:
            continue
        idx = vocab.index(term)

        r_k = sum(binary_matrix[doc_id][idx] for doc_id in relevant_docs)
        d_k = sum(doc[idx] for doc in binary_matrix)

        p_k = (r_k + 0.5) / (N_r + 1)
        q_k = (d_k - r_k + 0.5) / (N_d - N_r + 1)

        estimates[term] = {"r_k": r_k, "d_k": d_k, "p_k": p_k, "q_k": q_k}
    return estimates


# -------------------------------
# RSV Calculation
# -------------------------------
def calculate_rsv(doc_id, query_terms, estimates):
    rsv = 0
    for term in query_terms:
        if term not in estimates:
            continue
        idx = vocab.index(term)
        p_k, q_k = estimates[term]["p_k"], estimates[term]["q_k"]

        if binary_matrix[doc_id][idx] == 1:
            if p_k > 0 and q_k > 0:
                rsv += math.log(p_k / q_k)
        else:
            if p_k < 1 and q_k < 1:
                rsv += math.log((1 - p_k) / (1 - q_k))
    return rsv


# -------------------------------
# Search Functions
# -------------------------------
def search_phase1(query, top_k=5):
    query_terms = query.lower().split()
    estimates = phase1_estimate(query_terms)
    scores = [(i, calculate_rsv(i, query_terms, estimates)) for i in range(N_d)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


def search_phase2(query, relevant_docs, top_k=5):
    query_terms = query.lower().split()
    estimates = phase2_estimate(query_terms, relevant_docs)
    scores = [(i, calculate_rsv(i, query_terms, estimates)) for i in range(N_d)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


# -------------------------------
# Example Run
# -------------------------------
query = "information system"
print("=== Phase I (No Relevance Info) ===")
res1 = search_phase1(query)
print("Results:", res1)

print("\n=== Phase II (With Relevance Feedback) ===")
relevant_docs = [0, 2]  # assume docs 0 and 2 are relevant
res2 = search_phase2(query, relevant_docs)
print("Results:", res2)


# -------------------------------
# Quick Formulas Reference
# -------------------------------
def bim_formulas():
    print("Phase I:")
    print("  p_k = 0.5")
    print("  q_k = (d_k + 0.5) / (N_d + 1)")
    print("\nPhase II:")
    print("  p_k = (r_k + 0.5) / (N_r + 1)")
    print("  q_k = (d_k - r_k + 0.5) / (N_d - N_r + 1)")
    print("\nRSV:")
    print("  If term in doc: log(p_k / q_k)")
    print("  Else: log((1-p_k) / (1-q_k))")

"""
Vector Space Model (Non-OOP Version)
"""

import math
from collections import Counter

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess(docs):
    processed = []
    vocab_set = set()
    
    for doc in docs:
        tokens = doc.lower().split()
        processed.append(tokens)
        vocab_set.update(tokens)
    
    vocab = sorted(vocab_set)
    
    # Build TF matrix
    tf_matrix = []
    for doc_tokens in processed:
        counts = Counter(doc_tokens)
        tf_row = [counts.get(term, 0) for term in vocab]
        tf_matrix.append(tf_row)
    
    return vocab, tf_matrix

# -------------------------------
# IDF calculation
# -------------------------------
def compute_idf(tf_matrix, vocab):
    N = len(tf_matrix)
    idf = []
    for term_idx in range(len(vocab)):
        df = sum(1 for doc in tf_matrix if doc[term_idx] > 0)
        idf.append(math.log(N / df) if df > 0 else 0)
    return idf

# -------------------------------
# TF-IDF matrix
# -------------------------------
def compute_tfidf(tf_matrix, idf):
    return [[tf * idf[i] for i, tf in enumerate(doc)]
            for doc in tf_matrix]

# -------------------------------
# Query vector
# -------------------------------
def query_to_vector(query, vocab, idf):
    query_tf = Counter(query.lower().split())
    return [query_tf.get(term, 0) * idf[i] for i, term in enumerate(vocab)]

# -------------------------------
# Similarity measures
# -------------------------------
def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(a * a for a in v2))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0

def jaccard_coefficient(v1, v2):
    b1 = [1 if x > 0 else 0 for x in v1]
    b2 = [1 if x > 0 else 0 for x in v2]
    intersection = sum(a & b for a, b in zip(b1, b2))
    union = sum(a | b for a, b in zip(b1, b2))
    return intersection / union if union > 0 else 0

def dice_coefficient(v1, v2):
    b1 = [1 if x > 0 else 0 for x in v1]
    b2 = [1 if x > 0 else 0 for x in v2]
    intersection = sum(a & b for a, b in zip(b1, b2))
    total = sum(b1) + sum(b2)
    return (2 * intersection) / total if total > 0 else 0

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

# -------------------------------
# Search function
# -------------------------------
def search(query, docs, vocab, tfidf_matrix, idf, similarity='cosine', top_k=5):
    qvec = query_to_vector(query, vocab, idf)
    similarities = []
    
    for doc_id, dvec in enumerate(tfidf_matrix):
        if similarity == 'cosine':
            sim = cosine_similarity(qvec, dvec)
        elif similarity == 'jaccard':
            sim = jaccard_coefficient(qvec, dvec)
        elif similarity == 'dice':
            sim = dice_coefficient(qvec, dvec)
        elif similarity == 'dot':
            sim = dot_product(qvec, dvec)
        else:
            sim = cosine_similarity(qvec, dvec)
        
        similarities.append((doc_id, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# -------------------------------
# Usage
# -------------------------------
docs = ["information retrieval system data", 
        "machine learning data", 
        "web search engine"]

# Precompute
vocab, tf_matrix = preprocess(docs)
idf = compute_idf(tf_matrix, vocab)
tfidf_matrix = compute_tfidf(tf_matrix, idf)

# Search with different similarities
print("Cosine:", search("data", docs, vocab, tfidf_matrix, idf, 'cosine'))
print("Jaccard:", search("data", docs, vocab, tfidf_matrix, idf, 'jaccard'))
print("Dice:", search("data", docs, vocab, tfidf_matrix, idf, 'dice'))
print("Dot Product:", search("data", docs, vocab, tfidf_matrix, idf, 'dot'))

# -------------------------------
# Term-Document Boolean Model with File Input
# -------------------------------
import re

with open("documents.txt", "r") as f:
    text = f.read().strip()

# Split documents by 2 or more spaces
docs = re.split(r"\s{2,}", text)

# Step 0: Read documents from file
with open("documents.txt", "r") as f:
    docs = [line.strip() for line in f.readlines() if line.strip()]

# Step 1: Build Vocabulary
processed_docs = [doc.lower().split() for doc in docs]
vocab = sorted(set(term for doc in processed_docs for term in doc))

# Step 2: Build Term-Document Matrix (rows=terms, cols=docs)
term_doc_matrix = []
for term in vocab:
    row = []
    for tokens in processed_docs:
        row.append(1 if term in tokens else 0)
    term_doc_matrix.append(row)

# Function: Get vector of a term
def get_term_vector(term):
    term = term.lower()
    if term not in vocab:
        return [0] * len(docs)
    idx = vocab.index(term)
    return term_doc_matrix[idx]

# Function: Boolean Search
def boolean_search(query):
    query = query.lower().strip()
    
    # Single term
    if " and " not in query and " or " not in query and " not " not in query:
        return [i for i, val in enumerate(get_term_vector(query)) if val == 1]
    
    # AND
    if " and " in query:
        terms = [t.strip() for t in query.split(" and ")]
        result = get_term_vector(terms[0])
        for term in terms[1:]:
            vec = get_term_vector(term)
            result = [a & b for a, b in zip(result, vec)]
        return [i for i, val in enumerate(result) if val == 1]
    
    # OR
    if " or " in query:
        terms = [t.strip() for t in query.split(" or ")]
        result = get_term_vector(terms[0])
        for term in terms[1:]:
            vec = get_term_vector(term)
            result = [a | b for a, b in zip(result, vec)]
        return [i for i, val in enumerate(result) if val == 1]
    
    # NOT
    if " not " in query:
        pos, neg = [t.strip() for t in query.split(" not ")]
        pos_vec = get_term_vector(pos)
        neg_vec = get_term_vector(neg)
        neg_vec = [1 - x for x in neg_vec]
        result = [a & b for a, b in zip(pos_vec, neg_vec)]
        return [i for i, val in enumerate(result) if val == 1]
    
    return []

# -------------------------------
# Print Term-Document Matrix
# -------------------------------
print("Term-Document Matrix:")
print("Terms\\Docs", end="")
for i in range(len(docs)):
    print(f"\tD{i}", end="")
print()

for i, term in enumerate(vocab):
    print(f"{term:<10}", end="")
    for val in term_doc_matrix[i]:
        print(f"\t{val}", end="")
    print()

# -------------------------------
# Example Queries
# -------------------------------
print("\nSearch Results:")
print("'information':", boolean_search("information"))
print("'information and system':", boolean_search("information and system"))
print("'search or query':", boolean_search("search or query"))
print("'system not database':", boolean_search("system not database"))

# -------------------------------
# Inverted Index (Non-OOP Version)
# -------------------------------

# Documents
docs = ["cat dog bird", "dog bird", "cat mouse", "bird eagle", "mouse cat"]

# Step 1: Build inverted index
inverted_index = {}
for i, doc in enumerate(docs):
    for term in set(doc.lower().split()):
        inverted_index.setdefault(term, []).append(i)

print("Inverted Index:", inverted_index)

# -------------------------------
# Helper Functions
# -------------------------------
def get(term):
    """Get posting list"""
    return inverted_index.get(term.lower(), [])

def AND(list1, list2):
    """Intersect two lists"""
    return [x for x in list1 if x in list2]

def OR(list1, list2):
    """Union two lists"""
    return sorted(set(list1 + list2))

def NOT(posting_list):
    """Complement"""
    all_docs = list(range(len(docs)))
    return [x for x in all_docs if x not in posting_list]

def optimize_terms(terms, operation="and"):
    """Reorder terms based on posting list sizes"""
    term_lengths = [(term, len(get(term))) for term in terms]
    if operation == "and":
        # shortest lists first
        return [t for t, _ in sorted(term_lengths, key=lambda x: x[1])]
    else:  # OR
        # longest lists first
        return [t for t, _ in sorted(term_lengths, key=lambda x: x[1], reverse=True)]

# -------------------------------
# Boolean Search
# -------------------------------
def search(query):
    q = query.lower().strip()
   
    if " and " in q:
        terms = [t.strip() for t in q.split(" and ")]
        terms = optimize_terms(terms, "and")  # shortest first
        result = get(terms[0])
        for term in terms[1:]:
            result = AND(result, get(term))
            if not result:  # early exit
                break
        return result
   
    elif " or " in q:
        terms = [t.strip() for t in q.split(" or ")]
        terms = optimize_terms(terms, "or")  # longest first
        result = get(terms[0])
        for term in terms[1:]:
            result = OR(result, get(term))
        return result
   
    elif " not " in q:
        pos, neg = q.split(" not ")
        pos_list = get(pos.strip())
        neg_list = get(neg.strip())
        return AND(pos_list, NOT(neg_list))
   
    else:
        return get(q)

# -------------------------------
# Demo
# -------------------------------
print("\nQuery: 'cat and bird and dog'")
terms = ["cat", "bird", "dog"]
print("Posting list sizes:")
for t in terms:
    print(f"  {t}: {len(get(t))} docs")

print("Optimized order (AND):", optimize_terms(terms, "and"))
print("Result:", search("cat and bird and dog"))

print("\nOR optimization:")
print("Optimized order (OR):", optimize_terms(terms, "or"))
