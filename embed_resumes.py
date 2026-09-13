"""Iteration 3: turn saved resume chunks into a reusable local embedding index."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_DIR = ROOT / "data" / "chunked"
INDEX_DIR = ROOT / "data" / "index"
MODEL_CACHE = ROOT / "models" / "embedding-cache"

# Keep model downloads in the project, alongside the tokenizer from iteration 2.
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
from sentence_transformers import SentenceTransformer


def load_chunks():
    """Flatten the JSON files in a stable order, preserving every chunk's metadata."""
    chunks = []
    seen_ids = set()
    for path in sorted(CHUNK_DIR.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("embedding_model") != MODEL_NAME:
            raise ValueError(f"{path.name}: embedding model differs; rerun chunk_resumes.py.")
        for chunk in document["chunks"]:
            if not chunk.get("embedding_text", "").strip():
                raise ValueError(f"{path.name}: a chunk has no embedding_text.")
            if chunk["chunk_id"] in seen_ids:
                raise ValueError(f"Duplicate chunk ID: {chunk['chunk_id']}")
            seen_ids.add(chunk["chunk_id"])
            # This explicit row number links chunks.json to embeddings.npy.
            chunks.append({**chunk, "embedding_row": len(chunks)})
    if not chunks:
        raise ValueError("No chunks found. Run python chunk_resumes.py first.")
    return chunks


def save_index(chunks, embeddings, model):
    """Save vectors and their matching text; publish config last as a manifest."""
    expected_shape = (len(chunks), model.get_embedding_dimension())
    if embeddings.shape != expected_shape or not np.isfinite(embeddings).all():
        raise ValueError(f"Invalid embedding matrix: {embeddings.shape}; expected {expected_shape}.")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise ValueError("Expected unit-length embeddings.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=INDEX_DIR) as temporary:
        staging = Path(temporary)
        np.save(staging / "embeddings.npy", embeddings, allow_pickle=False)
        (staging / "chunks.json").write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        config = {
            "model_name": MODEL_NAME,
            "model_revision": getattr(model[0].auto_model.config, "_commit_hash", None),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(chunks),
            "resume_count": len({chunk["resume_id"] for chunk in chunks}),
            "dimensions": embeddings.shape[1],
            "dtype": str(embeddings.dtype),
            "normalized": True,
            "embedding_field": "embedding_text",
            "max_sequence_length": model.max_seq_length,
            "similarity": "cosine (dot product for normalized vectors)",
            "sha256": {
                name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                for name in ("embeddings.npy", "chunks.json")
            },
        }
        (staging / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        for name in ("embeddings.npy", "chunks.json", "config.json"):
            (staging / name).replace(INDEX_DIR / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use cached model files without downloading.")
    args = parser.parse_args()

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {len({c['resume_id'] for c in chunks})} resumes.", flush=True)
    print("Loading MiniLM locally (first run downloads the model)...", flush=True)
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(MODEL_CACHE),
        local_files_only=args.offline,
    )
    texts = [chunk["embedding_text"] for chunk in chunks]

    # Reject oversized inputs instead of silently truncating resume evidence.
    for chunk, text in zip(chunks, texts):
        token_ids = model.tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
        if len(token_ids) > model.max_seq_length:
            raise ValueError(f"{chunk['chunk_id']} exceeds {model.max_seq_length} tokens; rechunk it.")

    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    save_index(chunks, embeddings, model)
    print(f"Saved embeddings with shape {embeddings.shape} to {INDEX_DIR}")
    print("Files: embeddings.npy, chunks.json, config.json")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Embedding pipeline failed: {error}") from error
