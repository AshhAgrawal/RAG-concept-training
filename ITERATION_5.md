# Iteration 5: hybrid retrieval

`search_resumes.py` now defaults to hybrid search. Existing embeddings are reused; no reindexing or API key is needed. The new dependency is `rank-bm25==0.2.2` in requirements.txt (already installed in this workspace).

```bash
source .venv/bin/activate
python search_resumes.py "What experience does Ashutosh have with Twilio?" --candidate Ashutosh --top-k 1
```

Compare the same query with the previous semantic search or keyword-only search:

```bash
python search_resumes.py "What experience does Ashutosh have with Twilio?" --candidate Ashutosh --mode semantic
python search_resumes.py "Twilio" --candidate Ashutosh --mode keyword
```

`--section experience` and `--top-k` apply to every mode. Keyword mode skips loading the embedding model; both other modes use the cached model offline. The saved index is still required in every mode.

## Important code

The commented implementation lives in `hybrid_retrieval.py`:

1. `keyword_tokens()` lowercases text and removes a small list of question filler words. It preserves terms like C++, C#, Go, Node.js, and SKU-X140. This is a simple English lexical tokenizer, separate from MiniLM's tokenizer. It does not repair OCR, stem words, resolve synonyms, or implement quoted phrase search.
2. `KeywordIndex` builds BM25 statistics over all saved chunk bodies. BM25 rewards matching terms based on frequency, rarity across the corpus, and passage length. We use passage text rather than the added embedding labels. When candidate filtering is explicit, candidate-name tokens are removed from the lexical query to avoid rewarding contact headers solely for the name.
3. The existing `rank_chunks()` computes cosine similarities over eligible rows. Both keyword and semantic retrieval respect candidate/section filters. With just 54 chunks, we rank all eligible rows before taking the final top-k.
4. `fuse_rankings()` combines rank positions using Reciprocal Rank Fusion (RRF): `1 / (60 + semantic_rank) + 1 / (60 + keyword_rank)`. Ranks start at 1. A chunk absent from a list gets no contribution from that list. Keyword results require an actual term match and positive BM25 score. The constant 60 is separate from `--top-k`.
5. The CLI prints RRF, cosine, BM25, and matched keyword terms separately. Raw BM25 and cosine scores are not added together because their scales differ. None is confidence or accuracy. With two lists and this constant, the largest RRF score is about 0.03279; its small size is normal.

## What the comparison actually showed

Run the repeatable local comparison:

```bash
python evaluate_retrieval.py
python -m unittest -v test_search_resumes.py test_hybrid_retrieval.py
```

The evaluation includes exact terms, paraphrases, several employers, and an unsupported forklift-certification query. Expected passage markers are visible in `evaluate_retrieval.py`; output is saved to `data/evaluation/retrieval_comparison.json`.

On the current index, Twilio and Con Edison remain first in both methods. For the labeled Kafka work-experience passage, rank changes from 1 to 2. Customer-support evidence remains second; backend evidence changes from second to third. Each of those single-evidence cases still has its labeled passage in the top three. The employer query covers only one of three labeled employer passages in the top three with either method. These are illustrative labels, not an exhaustive relevance judgment over every passage or a held-out benchmark. We have not demonstrated a general accuracy improvement.

## Remaining limits

- Hybrid results can still include chunks without a keyword match. Use keyword mode when you explicitly want lexical matches only; it requires at least one query term, not every term or an exact phrase.
- With no keyword evidence, hybrid falls back to semantic ranking. Neither method can conclude that an answer is absent merely from this result.
- A top-k list does not guarantee complete candidate or employer coverage. Complete-section retrieval, better job boundaries, and reranking remain possible next improvements.
- Duplicate overlap and malformed PDF text remain unchanged. Adding BM25 does not fix extraction.
- Index updates and deletion handling follow the existing rebuild workflow.

References:
- BM25 library: https://github.com/dorianbrown/rank_bm25
- RRF explanation: https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
