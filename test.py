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


