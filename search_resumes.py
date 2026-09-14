"""Retrieve resume passages using semantic, keyword, or hybrid search locally."""

import argparse
import hashlib
import json

# Reuse the same project paths and model library as the embedding pipeline.
from embed_resumes import INDEX_DIR, MODEL_CACHE, MODEL_NAME, SentenceTransformer, np
from hybrid_retrieval import rank_chunks, retrieve


def load_index(index_dir=INDEX_DIR):
    """Load and validate vectors and metadata from the same index build."""
    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    # A mixed or partially replaced index could associate a vector with wrong text.
    for name in ("embeddings.npy", "chunks.json"):
        actual = hashlib.sha256((index_dir / name).read_bytes()).hexdigest()
        if actual != config["sha256"][name]:
            raise ValueError("Index checksum mismatch. Rebuild with python embed_resumes.py.")

    embeddings = np.load(index_dir / "embeddings.npy", allow_pickle=False)
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    if not chunks or embeddings.shape != (len(chunks), config["dimensions"]):
        raise ValueError("Embedding matrix does not match the chunk metadata.")
    if config["chunk_count"] != len(chunks):
        raise ValueError("Index chunk count is inconsistent.")
    if any(chunk["embedding_row"] != row for row, chunk in enumerate(chunks)):
        raise ValueError("Chunk rows are out of order. Rebuild the index.")
    if config["model_name"] != MODEL_NAME or not config.get("model_revision"):
        raise ValueError("Expected the project's MiniLM model and a recorded revision.")
    if not config["normalized"] or not np.isfinite(embeddings).all():
        raise ValueError("Expected finite, normalized embeddings.")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise ValueError("Stored embeddings are not unit-length vectors.")
    return embeddings, chunks, config


def normalize_name(name):
    return " ".join(name.casefold().split())


def select_rows(chunks, candidate=None, section=None):
    """Filter metadata before ranking. Preserve the ORIGINAL embedding row IDs."""
    names = sorted({chunk["candidate_name"] for chunk in chunks})
    selected_name = None
    if candidate is not None:
        query_name = normalize_name(candidate)
        if not query_name:
            raise ValueError("Candidate name cannot be empty.")
        exact = [name for name in names if normalize_name(name) == query_name]
        matches = exact or [name for name in names if query_name in normalize_name(name)]
        if not matches:
            raise ValueError(f"Candidate not found. Available names: {', '.join(names)}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous candidate. Use a full name: {', '.join(matches)}")
        selected_name = matches[0]

    rows = [row for row, chunk in enumerate(chunks)
            if (selected_name is None or chunk["candidate_name"] == selected_name)
            and (section is None or chunk["section"] == section)]
    if not rows:
        raise ValueError("No chunks match those candidate/section filters.")
    return np.array(rows, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question in quotation marks.")
    parser.add_argument("--candidate", help="Full name or an unambiguous part of a name.")
    parser.add_argument("--section", help="Optional section label, e.g. experience or skills.")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum passages to show (default: 3).")
    parser.add_argument("--mode", choices=["hybrid", "semantic", "keyword"], default="hybrid",
                        help="Ranking method (default: hybrid = semantic + BM25 with RRF).")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("Question cannot be empty.")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    # STEP 1: Load the saved index. We do NOT regenerate resume embeddings.
    embeddings, chunks, config = load_index()
    section = args.section.strip().lower() if args.section is not None else None
    rows = select_rows(chunks, args.candidate, section)
    print(f"Searching {len(rows)} of {len(chunks)} chunks | mode={args.mode}.", flush=True)

    # STEP 2: Load the SAME model revision that created the saved vectors.
    # Indexing already downloaded it, so search works locally without network access.
    model = SentenceTransformer(
        config["model_name"],
        revision=config["model_revision"],
        device="cpu",
        cache_folder=str(MODEL_CACHE),
        local_files_only=True,
    ) if args.mode != "keyword" else None
    if model is not None and model.get_embedding_dimension() != config["dimensions"]:
        raise ValueError("Loaded model dimensions differ from the saved index.")
    tokens = model.tokenizer(args.question, truncation=False)["input_ids"] if model else []
    if model is not None and len(tokens) > model.max_seq_length:
        raise ValueError(f"Question is too long; shorten it to at most {model.max_seq_length} tokens.")

    # STEP 3: Embed ONLY the question, with the same normalization as the chunks.
    query_embedding = model.encode(
        args.question, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    ) if model is not None else None

    # STEP 4: BM25 matches words; semantic search matches meaning. RRF combines
    # their ranks, not their incompatible raw scores. Filters apply to BOTH.
    results = retrieve(embeddings, query_embedding, chunks, rows, args.question,
                       args.top_k, args.mode, candidate_filtered=args.candidate is not None)

    # STEP 5: Use returned row IDs to print original text and source references.
    print("\nRetrieved passages (RRF, cosine, and BM25 are ranking scores, not confidence).")
    print("Top matches may not answer the question or cover every relevant fact.\n")
    if not results:
        print("No keyword matches in the selected chunks. This does not prove the answer is absent.")
    for rank, result in enumerate(results, start=1):
        row = result["row"]
        chunk = chunks[row]
        pages = ", ".join(str(page) for page in chunk["page_numbers"])
        scores = []
        if result["rrf"] is not None:
            scores.append(f"RRF={result['rrf']:.5f}")
        if result["cosine"] is not None:
            scores.append(f"cosine={result['cosine']:.3f}")
        if args.mode != "semantic":
            scores.append(f"BM25={result['bm25']:.3f}")
        print(f"{rank}. {chunk['candidate_name']} | {chunk['section']} | {' | '.join(scores)}")
        if args.mode != "semantic":
            print(f"Keyword matches: {', '.join(result['matched_terms']) or '(none; semantic match only)'}")
        print(f"Source: {chunk['source_file']} | pages: {pages} | {chunk['chunk_id']}")
        print(chunk["text"])
        print()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Search failed: {error}") from error
