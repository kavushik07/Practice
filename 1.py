#Boolean Model with file import
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Download required NLTK data (run once if needed)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Initialize PorterStemmer, stopwords, and operators
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
operators = {"and", "or", "not"}  # Use lowercase operators to match query tokens

# Preprocess function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(t) for t in tokens if t.isalpha() and t not in stop_words and t not in operators]
    return tokens

# Read documents from a text file
def read_documents(file_path):
    # Read CSV file with text and label columns, no header
    df = pd.read_csv(file_path, header=None, names=['text', 'label'])
    return df

# Evaluation metrics
def evaluate_query(retrieved, relevant):
    retrieved = set(retrieved)
    relevant = set(relevant)
    true_positives = len(retrieved & relevant)

    # Precision: |Retrieved ∩ Relevant| / |Retrieved|
    precision = true_positives / len(retrieved) if retrieved else 0.0

    # Recall: |Retrieved ∩ Relevant| / |Relevant|
    recall = true_positives / len(relevant) if relevant else 0.0

    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

# Main processing function
def main(file_path, query):
    # Read documents
    df = read_documents(file_path)
    docs = df['text'].tolist()
    labels = df['label'].tolist()

    # Get relevant document IDs based on labels
    relevant_docs = {i for i, label in enumerate(labels) if label == 'R'}

    # Preprocess documents and query
    preprocess_docs = [preprocess(d) for d in docs]
    preprocess_query = preprocess(query)  # For display only

    # Build inverted index
    terms = sorted(set(t for doc in preprocess_docs for t in doc))
    inverted_index = defaultdict(list)
    for term in terms:
        for doc_id, doc in enumerate(preprocess_docs):
            if term in doc:
                inverted_index[term].append(doc_id)

    # Query processing function
    def query_processing(query, inverted_index, num_docs):
        tokens = word_tokenize(query.lower())  # Use word_tokenize for proper splitting
        result = None
        operator = None
        all_docs = set(range(num_docs))  # Set of all document IDs
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in operators:
                operator = token
                i += 1
                continue
            term = ps.stem(token)
            current_docs = set(inverted_index.get(term, []))
            if operator == "not":
                current_docs = all_docs - current_docs
                operator = None
            if result is None:
                result = current_docs
            elif operator == "or":
                result = result.union(current_docs)
                operator = None
            elif operator == "and":
                result = result.intersection(current_docs)
                operator = None
            i += 1
        return result if result is not None else set()

    # Process query
    result = query_processing(query, inverted_index, len(docs))

    # Evaluate results
    metrics = evaluate_query(result, relevant_docs)

    # Print results
    print("Documents:", docs)
    print("Labels:", labels)
    print("Preprocessed Documents:", preprocess_docs)
    print("Preprocessed Query (terms only):", preprocess_query)
    print("Inverted Index:", dict(inverted_index))
    print("Query:", query)
    print("Relevant Document IDs (from labels):", relevant_docs)
    print("Retrieved Document IDs:", result)
    print("Retrieved Documents:", [docs[i] for i in result])
    print("Evaluation Metrics:", metrics)

# Example usage
if _name_ == "_main_":
    # Write sample data to data.txt for testing
    sample_data = """Information retrieval is fun.,R
Retrieval of information.,R
Fun with information.,NR
Fun is always fun.,NR"""
    with open("data.txt", "w") as f:
        f.write(sample_data)

    file_path = "data.txt"
    query = "information"
    main(file_path, query)

--
Regards,
Sherin
