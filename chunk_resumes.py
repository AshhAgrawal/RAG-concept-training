"""Iteration 2: detect resume sections, then split each into token-sized chunks."""

import hashlib
import json
import re
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_PATH = ROOT / "models" / "all-MiniLM-L6-v2" / "tokenizer.json"
MAX_TOKENS = 200  # Includes the person/section prefix and special tokens.
OVERLAP_TOKENS = 30

HEADING_ALIASES = {
    "skills": ["skills", "technical skills", "technologies", "core competencies"],
    "experience": ["experience", "work experience", "professional experience", "employment history"],
    "education": ["education", "academic background", "qualifications"],
    "projects": ["projects", "academic projects", "personal projects", "projects and papers"],
    "summary": ["summary", "professional summary", "profile", "objective"],
    "activities": ["co-curricular activities", "extracurricular activities", "activities", "leadership"],
    "certifications": ["certifications", "certificates"],
    "publications": ["publications", "papers"],
    "awards": ["awards", "honors", "achievements"],
}


def normalize_heading(text):
    """Ignore spacing and punctuation ONLY when comparing whole heading lines."""
    return re.sub(r"[^a-z]", "", text.lower())


HEADINGS = {
    normalize_heading(alias): section
    for section, aliases in HEADING_ALIASES.items()
    for alias in aliases
}


def extract_sections(resume):
    """Preserve text and page boundaries; a known heading begins a new section."""
    sections = []
    current = {"section": "unclassified", "heading": None, "lines": []}
    for page in resume["pages"]:
        for raw_line in page["text"].splitlines():
            line = raw_line.strip()
            if not line:
                continue
            section = HEADINGS.get(normalize_heading(line))
            if section:
                if current["lines"]:
                    sections.append(current)
                current = {"section": section, "heading": line, "lines": []}
            else:
                current["lines"].append({"page_number": page["page_number"], "text": line})
    if current["lines"]:
        sections.append(current)
    return sections


def split_text(text, prefix, tokenizer):
    """Yield original-text spans, preferring line endings and overlapping context."""
    def count(value):
        return len(tokenizer.encode(value).ids)

    start = 0
    while start < len(text):
        remaining = text[start:]
        offsets = tokenizer.encode(remaining, add_special_tokens=False).offsets
        if not offsets:
            break
        budget = MAX_TOKENS - count(prefix)
        if budget <= OVERLAP_TOKENS:
            raise ValueError("Person/section prefix leaves too little space for chunk text.")
        stop = min(budget, len(offsets))
        end = len(text) if stop == len(offsets) else start + offsets[stop][0]
        # Prefer a complete extracted line over a mid-line cut.
        newline = text.rfind("\n", start, end)
        if end < len(text) and newline > start + (end - start) // 2:
            end = newline + 1
        # Verify the exact final input, including prefix and special tokens.
        while count(prefix + text[start:end]) > MAX_TOKENS:
            end -= 1
        if end <= start:
            raise ValueError("Unable to fit any text in a chunk.")
        yield start, end, count(prefix + text[start:end])
        if end == len(text):
            break
        chunk_offsets = tokenizer.encode(text[start:end], add_special_tokens=False).offsets
        overlap_index = max(1, len(chunk_offsets) - OVERLAP_TOKENS)
        next_start = start + chunk_offsets[overlap_index][0] if overlap_index < len(chunk_offsets) else end
        start = max(start + 1, next_start)


def process_resume(resume, tokenizer):
    source = resume["source_file"]
    resume_id = hashlib.sha256(source.encode()).hexdigest()[:12]
    # A transparent heuristic for these six inspected files, not a name resolver.
    candidate_name = next(
        (line.strip() for page in resume["pages"] for line in page["text"].splitlines() if line.strip()),
        source,
    )
    sections = extract_sections(resume)
    chunks = []
    for section_number, section in enumerate(sections, start=1):
        section["section_id"] = f"{resume_id}:s{section_number}"
        text = "\n".join(line["text"] for line in section["lines"])
        section["text"] = text
        spans = []
        position = 0
        for line in section["lines"]:
            spans.append((position, position + len(line["text"]), line["page_number"]))
            position += len(line["text"]) + 1
        section["page_numbers"] = sorted({span[2] for span in spans})
        prefix = f"Candidate: {candidate_name}\nSection: {section['section']}\n"
        for index, (start, end, count) in enumerate(split_text(text, prefix, tokenizer), start=1):
            chunks.append({
                "chunk_id": f"{section['section_id']}:c{index}",
                "resume_id": resume_id,
                "source_file": source,
                "candidate_name": candidate_name,
                "section_id": section["section_id"],
                "section": section["section"],
                "page_numbers": sorted({page for left, right, page in spans if left < end and right > start}),
                "start_char": start,
                "end_char": end,
                "text": text[start:end],
                "embedding_text": prefix + text[start:end],
                "token_count": count,
            })
    return {
        "source_file": source,
        "resume_id": resume_id,
        "candidate_name": candidate_name,
        "candidate_name_method": "first_nonempty_line; review before retrieval",
        "embedding_model": MODEL,
        "tokenizer_sha256": hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest(),
        "max_tokens": MAX_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "sections": sections,
        "chunks": chunks,
    }


def main():
    if not TOKENIZER_PATH.exists():
        raise SystemExit("Tokenizer missing. Follow the download command in ITERATION_2.md.")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    inputs = sorted(path for path in (ROOT / "data" / "extracted").glob("*.json")
                    if path.name.lower().endswith(".pdf.json"))
    if not inputs:
        raise SystemExit("No extracted PDFs found. Run extract_resumes.py first.")
    output_dir = ROOT / "data" / "chunked"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        result = process_resume(json.loads(path.read_text(encoding="utf-8")), tokenizer)
        (output_dir / path.name).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        preview = []
        for chunk in result["chunks"]:
            preview.append(f"[{chunk['chunk_id']}] pages={chunk['page_numbers']} tokens={chunk['token_count']}\n{chunk['embedding_text']}")
        (output_dir / f"{path.stem}.txt").write_text("\n\n---\n\n".join(preview), encoding="utf-8")
        labels = ", ".join(section["section"] for section in result["sections"])
        print(f"{result['candidate_name']}: {labels} → {len(result['chunks'])} chunks")


if __name__ == "__main__":
    main()
