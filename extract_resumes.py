"""Iteration 1: extract resume text and keep its source page information."""

import json
from pathlib import Path

from pypdf import PdfReader


# Resolve paths relative to this script, regardless of the terminal's location.
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "data" / "resumes"
OUTPUT_DIR = PROJECT_DIR / "data" / "extracted"


def extract_resume(pdf_path):
    """Read one PDF and return one record per page."""
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append({"page_number": page_number, "text": text})
        if not text:
            print(f"  WARNING: {pdf_path.name}, page {page_number}: no text found.")

    return {"source_file": pdf_path.name, "pages": pages}


def main():
    pdf_files = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdf_files:
        print(f"No PDFs found in {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0

    for pdf_path in pdf_files:
        try:
            resume = extract_resume(pdf_path)
        except Exception as error:
            # Report a bad PDF and continue inspecting the other resumes.
            print(f"ERROR: {pdf_path.name}: {error}")
            failures += 1
            continue

        # JSON preserves structure for the later chunking step.
        json_path = OUTPUT_DIR / f"{pdf_path.name}.json"
        json_path.write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")

        # Plain text is convenient for human inspection.
        sections = [f"Source: {resume['source_file']}"]
        for page in resume["pages"]:
            sections.append(f"--- Page {page['page_number']} ---\n{page['text']}")
        text_path = OUTPUT_DIR / f"{pdf_path.name}.txt"
        text_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")

        characters = sum(len(page["text"]) for page in resume["pages"])
        print(f"OK: {pdf_path.name}: {len(resume['pages'])} page(s), {characters} characters")

    print(f"\nSaved {len(pdf_files) - failures}/{len(pdf_files)} resumes to {OUTPUT_DIR}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
