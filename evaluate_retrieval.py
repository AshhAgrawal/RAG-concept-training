"""Small, inspectable comparison on the uploaded resumes; not a general benchmark."""

import json

from search_resumes import load_index, select_rows, SentenceTransformer, MODEL_CACHE, INDEX_DIR
from hybrid_retrieval import KeywordIndex, retrieve

# Labels specify the supporting passages, independently of either search ranking.
# Multiple strings require coverage across passages (e.g. the employer question).
CASES = [
    ("Twilio", "What experience does Ashutosh have with Twilio?", "Ashutosh", ["Twilio"]),
    ("Con Edison", "Re-architected Con Edison REST APIs", "Ashutosh", ["Con Edison"]),
    ("Kafka", "Who has worked with Kafka?", None, ["Integrated Kafka"]),
    ("Paraphrase: support", "How did Ashutosh automate customer support?", "Ashutosh", ["Twilio"]),
    ("Paraphrase: backend", "What backend development experience does Ashutosh have?", "Ashutosh", ["Con Edison"]),
    ("Multiple employers", "Which organizations did Ashutosh work for?", "Ashutosh",
     ["Meltek Inc.", "Sterlite Technologies", "LTIMindtree Citibank"]),
    ("No evidence", "Does Ashutosh have a forklift operator certification?", "Ashutosh", []),
]


def main():
    vectors, chunks, config = load_index()
    model = SentenceTransformer(config["model_name"], revision=config["model_revision"],
                                cache_folder=str(MODEL_CACHE), local_files_only=True, device="cpu")
    queries = model.encode([case[1] for case in CASES], normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    keyword_index = KeywordIndex(chunks)
    report = []
    print("Case | mode | first supporting rank | labeled evidence coverage@3")
    for (label, question, candidate, evidence), vector in zip(CASES, queries):
        rows = select_rows(chunks, candidate)
        groups = [{int(row) for row in rows if phrase.casefold() in chunks[row]["text"].casefold()}
                  for phrase in evidence]
        if any(not group for group in groups):
            raise ValueError(f"Missing expected passage for {label}; review the evaluation labels.")
        entry = {"case": label, "question": question, "candidate": candidate,
                 "expected_passage_markers": evidence, "results": {}}
        for mode in ("semantic", "hybrid"):
            results = retrieve(vectors, vector, chunks, rows, question, len(rows), mode,
                               candidate_filtered=candidate is not None, keyword_index=keyword_index)
            ranking = [result["row"] for result in results]
            first = next((i for i, row in enumerate(ranking, 1) if any(row in group for group in groups)), None)
            covered = sum(bool(group.intersection(ranking[:3])) for group in groups)
            coverage = f"{covered}/{len(groups)}" if groups else "N/A: unsupported query still returns matches"
            print(f"{label} | {mode} | {first or 'N/A'} | {coverage}")
            entry["results"][mode] = {"first_supporting_rank": first,
                "covered_evidence_groups_at_3": covered if groups else None,
                "top_3": [{**item, "chunk_id": chunks[item["row"]]["chunk_id"]} for item in results[:3]]}
        report.append(entry)
    target = INDEX_DIR.parent / "evaluation"
    target.mkdir(exist_ok=True)
    (target / "retrieval_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved data/evaluation/retrieval_comparison.json; labels are illustrative, not exhaustive.")


if __name__ == "__main__":
    main()
