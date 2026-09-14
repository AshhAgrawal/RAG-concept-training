"""Keyword ranking plus Reciprocal Rank Fusion, independent of model loading."""

import re

import numpy as np
from rank_bm25 import BM25Okapi

# Remove question filler, while keeping technical terms such as Go, C++, and C#.
STOP_WORDS = set("a an the is are was were be been being do does did has have had "
                 "what which who where when how can could would should please tell me "
                 "about of for to in on at by with and or from this that these those "
                 "his her their he she they it i you my".split())
RRF_CONSTANT = 60


def rank_chunks(embeddings, query_embedding, rows, top_k):
    """Return original row IDs and cosine scores, highest score first."""
    if top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if query_embedding.shape != (embeddings.shape[1],) or not np.isfinite(query_embedding).all():
        raise ValueError("Question embedding has invalid dimensions or values.")
    if not np.isclose(np.linalg.norm(query_embedding), 1.0, atol=1e-5):
        raise ValueError("Question embedding must be normalized.")
    # Unit-length vectors make dot product equal cosine similarity.
    scores = embeddings[rows] @ query_embedding
    best_positions = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(rows[position]), float(scores[position])) for position in best_positions]


def keyword_tokens(text):
    """Match the same lowercase terms in questions and passages; preserve C++/C#."""
    tokens = re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:\+\+|#)?", text.casefold())
    return [token for token in tokens if token not in STOP_WORDS]


class KeywordIndex:
    def __init__(self, chunks):
        # Use actual passage text, not added Candidate/Section embedding labels.
        self.documents = [keyword_tokens(chunk["text"]) for chunk in chunks]
        # Corpus-wide IDF statistics remain stable when candidate filters change.
        self.bm25 = BM25Okapi(self.documents) if any(self.documents) else None

    def search(self, question, rows, ignored_terms=()):
        terms = list(dict.fromkeys(t for t in keyword_tokens(question) if t not in ignored_terms))
        if not terms or self.bm25 is None:
            return [], {}
        scores = self.bm25.get_scores(terms)
        matched = {int(row): sorted(set(terms).intersection(self.documents[row])) for row in rows}
        # No lexical hit means NO place in the keyword ranking and no RRF bonus.
        hits = [(int(row), float(scores[row])) for row in rows if matched[int(row)] and scores[row] > 0]
        hits.sort(key=lambda pair: (-pair[1], pair[0]))
        return hits, matched


def fuse_rankings(semantic, keyword, top_k):
    """Add 1/(60 + rank) from each list; raw cosine and BM25 scores aren't added."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    results = {}
    for method, ranking in (("semantic", semantic), ("keyword", keyword)):
        for rank, (row, score) in enumerate(ranking, start=1):
            item = results.setdefault(row, {"row": row, "rrf": 0.0,
                                          "cosine": None, "bm25": 0.0,
                                          "semantic_rank": None, "keyword_rank": None})
            item["rrf"] += 1.0 / (RRF_CONSTANT + rank)
            item[f"{method}_rank"] = rank
            item["cosine" if method == "semantic" else "bm25"] = score
    return sorted(results.values(), key=lambda item: (-item["rrf"], item["row"]))[:top_k]


def retrieve(embeddings, query_embedding, chunks, rows, question, top_k=3,
             mode="hybrid", candidate_filtered=False, keyword_index=None):
    if mode not in {"hybrid", "semantic", "keyword"}:
        raise ValueError("Unknown retrieval mode.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    semantic = rank_chunks(embeddings, query_embedding, rows, len(rows)) if mode != "keyword" else []
    ignored = set()
    if candidate_filtered:
        # The metadata filter already resolved the person. Don't reward contact
        # headers just because the same name is repeated in the question.
        ignored = {term for row in rows for term in keyword_tokens(chunks[row]["candidate_name"])}
    keyword, matched = (keyword_index or KeywordIndex(chunks)).search(question, rows, ignored)

    if mode == "hybrid":
        results = fuse_rankings(semantic, keyword, top_k)
    else:
        ranking = semantic if mode == "semantic" else keyword
        results = [{"row": row, "rrf": None,
                    "cosine": score if mode == "semantic" else None,
                    "bm25": score if mode == "keyword" else 0.0}
                   for row, score in ranking[:top_k]]
    for result in results:
        result["matched_terms"] = matched.get(result["row"], [])
    return results
