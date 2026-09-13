# Iteration 2: section-aware chunking

Run from the project folder with the environment activated:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python chunk_resumes.py
python -m unittest -v test_chunk_resumes.py
```

The tokenizer is already downloaded. For a fresh setup, download it before running:

```bash
mkdir -p models/all-MiniLM-L6-v2
curl -fL https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json -o models/all-MiniLM-L6-v2/tokenizer.json
```

This is a public tokenizer download, not a resume upload. Parsing and chunking run locally. No embeddings or LLM answers are generated in this iteration.

## How it works

1. Read the JSON from iteration 1, preserving each line's page number.
2. Normalize whole heading lines for comparison: `Technical Skills`, `TECHNICALSKILLS`, and `Technical Skills:` all become `technicalskills`. Map them to `skills` through `HEADING_ALIASES`.
3. Begin a new section at each recognized heading. Retain introductory text as `unclassified`. Unknown headings remain in the current section; they are not discarded. Repeated sections keep separate IDs.
4. Use the first nonempty line as the candidate name. This heuristic matches these six inspected files. Review it for new files: it is not general name recognition, typo matching, or deduplication. Resume IDs are based on the source filename.
5. Split each section independently using MiniLM's tokenizer, with a maximum of 200 tokens INCLUDING candidate and section labels and special tokens. Prefer an extracted line ending when one fits; otherwise split at a token boundary. Include approximately 30 tokens of overlap within the same section. This does not yet identify whole employment entries or guarantee whole sentences.
6. Save `data/chunked/<original filename>.json` and a readable `.txt` preview. JSON includes complete sections plus chunks, exact character spans within section text, source pages, names, IDs, token counts, and a tokenizer checksum.

`text` holds the original excerpt (with outer line whitespace stripped). `embedding_text` adds candidate and section labels; embed that field in the next iteration. Keep full sections for queries that need all employment entries rather than only the top few chunks.

## Options for section extraction

| Approach | How it works | Tradeoff |
| --- | --- | --- |
| Heading rules (ours) | Known headings establish boundaries | Cheap, inspectable, local; misses unfamiliar headings |
| Layout-aware parsing | Uses PDF layout, positioning, and title detection | Better for visual structure; more dependencies and still needs mapping to resume section names |
| LLM extraction | Asks a model to identify sections in text or document images | Flexible with unusual wording; needs validation against source text and consumes inference resources |
| Hybrid | Rules for familiar documents, layout/LLM handling for flagged cases | Useful as document variety grows; more complexity |

Layout-aware example: https://docs.unstructured.io/open-source/concepts/partitioning-strategies

Embedding model and input limit: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Review before embeddings

Open Ashutosh's chunk preview beside the extracted text. Check that employment entries are in `experience`, projects in `projects`, and every chunk has the correct candidate and source. Repeat for the other resumes. Joined words from PDF extraction remain a known limitation; the parser only repairs heading matching, not body text. Unknown headings and multi-column reading order require human review. These tests verify boundaries, token limits, page attribution, and coverage, not perfect semantic interpretation of every resume.
