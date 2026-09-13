# Iteration 3: generate and save embeddings

From the project folder:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python embed_resumes.py
```

The script reads the existing `data/chunked/*.json` files. It does not re-extract PDFs or rerun chunking. When resumes change, run `extract_resumes.py`, then `chunk_resumes.py`, then `embed_resumes.py`. Deleted or renamed resumes also require removing their old generated files from `data/extracted/` and `data/chunked/` before rebuilding; automatic synchronization is a future iteration.

## What happens

1. Load the chunks in a stable order and attach an `embedding_row` to each one.
2. Load `sentence-transformers/all-MiniLM-L6-v2` on the CPU. The first run downloads the model into `models/embedding-cache/`. Resume text is processed locally.
3. Check the actual input token counts against the loaded model's limit, so text cannot silently be truncated.
4. Pass each chunk's `embedding_text` (candidate label, section label, and passage) to `model.encode()` in batches of 16. `normalize_embeddings=True` makes each vector unit length.
5. Save float32 vectors, matching metadata, and settings to `data/index/`. Rerunning rebuilds and replaces this index.

```text
data/index/
  embeddings.npy  # One row per chunk, 384 numbers per row
  chunks.json     # Text, source, person, section, pages, and embedding_row
  config.json     # Model, revision, dimensions, normalization, and file checksums
```

The initial six resumes have 54 chunks, so the expected matrix shape is `(54, 384)`. Vector row `i` belongs to entry `i` in `chunks.json`. Both must always be loaded from the same build; config checksums let a future reader verify that pairing. The config is written last, but concurrent reading during a rebuild is not supported in this learning CLI.

After a successful first run, explicitly use only the cached model with:

```bash
python embed_resumes.py --offline
```

If the package cannot be imported, use `python -m pip install -r requirements.txt` after activating `.venv`. A bare `pip` command can refer to a different Python installation.

There are no Groq calls, LLM answers, or search commands in this iteration. Retrieval is the next step: embed a question with the same model and normalization, filter metadata as appropriate, and compare it with the saved vectors.

API reference: https://www.sbert.net/docs/package_reference/sentence_transformer/model.html
