"""
Mini Search Engine
==================
Structures used: HashMap (dict), Inverted Index, TF-IDF ranking
Author: Meghraoui Fatma Youssra
"""

import math
import re
from collections import defaultdict


# ─────────────────────────────────────────────
#  STOPWORDS (common words we ignore)
# ─────────────────────────────────────────────
STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "of",
    "to", "and", "or", "for", "with", "that", "this", "was",
    "are", "be", "by", "from", "as", "has", "have", "had"
}


# ─────────────────────────────────────────────
#  STEP 1 — TOKENIZER
#  Input : raw string
#  Output: list of clean tokens
# ─────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    text = text.lower()                        # lowercase
    tokens = re.findall(r'\b[a-z]+\b', text)  # keep only words
    tokens = [t for t in tokens if t not in STOPWORDS]  # remove stopwords
    return tokens


# ─────────────────────────────────────────────
#  STEP 2 — INVERTED INDEX BUILDER
#  Builds: word → {doc_id: term_frequency}
# ─────────────────────────────────────────────
class InvertedIndex:
    def __init__(self):
        # HashMap: word → {doc_id → count}
        self.index: dict[str, dict[int, int]] = defaultdict(
            lambda: defaultdict(int))
        self.doc_lengths: dict[int, int] = {}   # total tokens per doc
        self.num_docs: int = 0

    def add_document(self, doc_id: int, text: str):
        tokens = tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.num_docs += 1
        for token in tokens:
            self.index[token][doc_id] += 1

    def get_posting_list(self, word: str) -> dict[int, int]:
        """Returns {doc_id: count} for a given word."""
        return self.index.get(word, {})


# ─────────────────────────────────────────────
#  STEP 3 — TF-IDF SCORER
#  TF(w, d)  = count(w in d) / total_tokens(d)
#  IDF(w)    = log(N / df(w))   [df = docs containing w]
#  Score     = TF × IDF
# ─────────────────────────────────────────────
def compute_tf(count: int, doc_length: int) -> float:
    return count / doc_length if doc_length > 0 else 0.0


def compute_idf(num_docs: int, doc_freq: int) -> float:
    return math.log(num_docs / doc_freq) if doc_freq > 0 else 0.0


def score_documents(query: str, idx: InvertedIndex) -> list[tuple[int, float]]:
    """
    Returns a ranked list of (doc_id, score) for the given query.
    Multi-word query: scores are summed across all query terms.
    """
    query_tokens = tokenize(query)
    scores: dict[int, float] = defaultdict(float)

    for token in query_tokens:
        posting = idx.get_posting_list(token)
        if not posting:
            continue
        idf = compute_idf(idx.num_docs, len(posting))
        for doc_id, count in posting.items():
            tf = compute_tf(count, idx.doc_lengths[doc_id])
            scores[doc_id] += tf * idf

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


# ─────────────────────────────────────────────
#  STEP 4 — SEARCH ENGINE (puts it all together)
# ─────────────────────────────────────────────
class SearchEngine:
    def __init__(self):
        self.index = InvertedIndex()
        self.documents: dict[int, str] = {}   # doc_id → original text
        self._next_id = 0

    def index_document(self, text: str) -> int:
        """Adds a document to the engine. Returns its doc_id."""
        doc_id = self._next_id
        self._next_id += 1
        self.documents[doc_id] = text
        self.index.add_document(doc_id, text)
        return doc_id

    def search(self, query: str, top_n: int = 5) -> list[dict]:
        """
        Search for query and return top_n results.
        Each result: { 'doc_id', 'score', 'snippet' }
        """
        ranked = score_documents(query, self.index)
        results = []
        for doc_id, score in ranked[:top_n]:
            snippet = self.documents[doc_id][:120] + "..."
            results.append({
                "doc_id": doc_id,
                "score": round(score, 4),
                "snippet": snippet
            })
        return results


# ─────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    engine = SearchEngine()

    # Index some sample documents
    docs = [
        "Python is a powerful programming language used for data science and machine learning.",
        "Machine learning algorithms include decision trees, neural networks, and random forests.",
        "Data science involves statistics, programming, and domain expertise.",
        "Neural networks are inspired by the human brain and used in deep learning.",
        "Python libraries like NumPy and Pandas are essential for data analysis.",
        "Deep learning models require large datasets and powerful GPUs to train.",
        "Search engines use inverted indexes to retrieve documents efficiently.",
        "Algorithms and data structures are fundamental to computer science.",
        "Natural language processing enables computers to understand human language.",
        "Kinésithérapie involves physical therapy and rehabilitation exercises.",
    ]

    print("=" * 60)
    print("  Mini Search Engine — Indexing documents...")
    print("=" * 60)

    for doc in docs:
        doc_id = engine.index_document(doc)
        print(f"  [doc {doc_id}] indexed: {doc[:50]}...")

    print(f"\nTotal documents indexed: {engine.index.num_docs}")
    print(f"Unique words in index:   {len(engine.index.index)}\n")

    # Search queries
    queries = [
        "machine learning neural networks",
        "python data science",
        "deep learning GPU",
        "search engine index",
    ]

    for query in queries:
        print("=" * 60)
        print(f"  Query: '{query}'")
        print("=" * 60)
        results = engine.search(query, top_n=3)
        if not results:
            print("  No results found.\n")
        else:
            for i, r in enumerate(results, 1):
                print(f"  {i}. [doc {r['doc_id']}] score={r['score']}")
                print(f"     {r['snippet']}\n")
