# Iteration 4: search the saved resume embeddings

Run from the project folder:

```bash
source .venv/bin/activate
python search_resumes.py "What backend development experience does Ashutosh have?" --candidate "Ashutosh Agrawal"
```

Optional filters and result count:

```bash
python search_resumes.py "Where did Ashutosh work?" --candidate Ashutosh --section experience --top-k 5
python search_resumes.py "Who has worked with Kafka?" --top-k 5
```

## Follow the five steps in main()

1. `load_index()` loads saved vectors, chunks, and model settings. File checksums detect mismatched index files. `select_rows()` restricts eligible rows using candidate and section metadata.
2. `SentenceTransformer(...)` loads the same model and recorded revision from the local cache. No resume vectors are regenerated and no API calls are made.
3. `model.encode(question, normalize_embeddings=True, ...)` converts the question into one 384-dimensional vector. Document labels need not be added to the question: it is already encoded by the same model into the same vector space.
4. `embeddings[rows] @ query_embedding` calculates one cosine similarity per eligible chunk. This works because both document vectors and the question vector have length 1. `np.argsort(-scores)` selects the highest scores first.
5. Returned row IDs retrieve the original chunk text, candidate, section, PDF filename, and page references for display.

The candidate filter accepts an exact name or an unambiguous substring, ignoring case and extra spaces. It does not automatically detect names in the question or resolve typos. Section labels are the normalized labels such as `experience`, `skills`, and `projects`. Without filters, every stored chunk is eligible.

## What to inspect

Compare retrieved passages with the original resumes. Do they actually support an answer? A high score is similarity, not a probability that the passage answers the question. This search always returns the nearest available chunks: even an unrelated question can get results. We do not use an uncalibrated cutoff to claim that a fact is absent.

Top-k means up to k chunks, not k distinct candidates or complete employment histories. Overlap can produce repeated evidence, and continuation chunks may omit an employer header. For exhaustive questions, later retrieval should expand to complete sections or fetch evidence per resume. The saved `section_id` enables that extension. No LLM answer is generated in this iteration.

Search reads the existing index; rebuild it after changing inputs. If the model cache is missing, run `python embed_resumes.py` with network access to download it and build the index first.

Run focused tests:

```bash
python -m unittest -v test_search_resumes.py
```
