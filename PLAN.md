# Resume RAG learning project

Build a Python command-line application that answers factual questions using 3–5 resume PDFs and cites its sources. We will use existing models; no model training is required. Implement one milestone at a time, explain the concepts, and inspect its output before continuing.

## Input folder

Place one resume per PDF in `data/resumes/`, using recognizable filenames. Start with English PDFs with selectable text. Scanned PDFs may need OCR after we inspect them.

## Proposed stack

- Python with small, explicit functions so each RAG step is visible.
- Local Sentence Transformers `all-MiniLM-L6-v2` embeddings. The model produces 384-dimensional vectors and truncates inputs beyond 256 word pieces, so chunk sizes must respect its tokenizer limit, including any added labels.
- Initially save vectors in a NumPy file and text/metadata in JSON. Use cosine similarity for retrieval; a dedicated vector database can be a later iteration.
- GroqCloud for answer generation, using an available free-plan text model selected at setup. An API key is required and free usage has rate limits.
- Resume text and embeddings stay local during indexing. Answer generation sends the question and retrieved excerpts to Groq.

## Pipeline

Indexing: PDFs → extracted text → section chunks → embeddings → saved local index.

Answering: question → same embedding model → similar chunks → Groq with source context → answer with citations.

Store the original chunk text alongside each vector: the LLM needs readable evidence, not just embeddings.

## Milestones

1. **Extract and inspect.** Read each PDF and preserve filename and page references. Check reading order, missing text, and layout problems. Deliverable: readable extracted text for every resume.
2. **Chunk by section.** Identify sections such as skills, experience, education, and projects. Split long sections within the embedding model's token budget with modest overlap. Preserve resume ID, section, source pages, and chunk ID; never mix people in one chunk. Deliverable: inspectable chunks.
3. **Embed and persist.** Embed chunks locally and save vectors, text, metadata, and embedding configuration. Reload the index after restarting. Deliverable: a reusable index without re-embedding on every question.
4. **Retrieve before generating.** Embed questions with the same model and display the highest-scoring chunks and source references. Try both paraphrased questions and candidate-specific questions. Deliverable: a search command whose evidence we can inspect.
5. **Generate grounded answers.** Send selected chunks and the question to Groq. Require source citations and a clear insufficient-evidence response. Treat resume content as data, never instructions. Deliverable: an ask command with answers and sources.
6. **Evaluate and finish the CLI.** Create roughly 10–15 questions with expected supporting passages, including missing facts, overlapping skills, and cross-resume queries. Check retrieval coverage and answer grounding separately. Add ingest/search/ask commands, setup instructions, environment-based API-key handling, useful errors, and changed-file reindexing. Keep PDFs, derived data, and secrets out of version control when Git is introduced.

## Key lessons

- Extraction and chunk quality determine what retrieval can find.
- Similarity scores are not probabilities that an answer is correct.
- A global top-k search can miss candidates for “who all” questions. Add per-resume retrieval or broader coverage for these queries.
- Missing evidence does not prove a person lacks a skill.
- If the embedding model changes, rebuild the index.
- Grounding prompts need evaluation; they do not guarantee factual answers.

## Questions before implementation

- How comfortable are you with Python, virtual environments, and API calls?
- Are the PDFs English and selectable text, or scanned images?
- Are local embeddings and sending retrieved excerpts to Groq acceptable for these resumes?

## References

- Embedding model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Groq free-plan rate limits: https://console.groq.com/docs/rate-limits
- Groq supported models: https://console.groq.com/docs/models

Current status: planning only. No application code or dependencies installed.
