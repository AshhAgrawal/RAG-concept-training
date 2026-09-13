"""Checks for section boundaries, source attribution, and lossless chunk coverage."""

import json
import unittest

from chunk_resumes import ROOT, TOKENIZER_PATH, MAX_TOKENS, extract_sections, process_resume
from tokenizers import Tokenizer


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        cls.tokenizer.no_truncation()
        cls.tokenizer.no_padding()

    def test_joined_headings_and_page_continuation(self):
        resume = {"source_file": "sample.pdf", "pages": [
            {"page_number": 1, "text": "Sample Person\nTECHNICALSKILLS:\nPython\nPROFESSIONALEXPERIENCE\nCompany A"},
            {"page_number": 2, "text": "Built APIs\nPROJECTS ANDPAPERS\nDemo"},
        ]}
        result = process_resume(resume, self.tokenizer)
        experience = next(c for c in result["chunks"] if c["section"] == "experience")
        self.assertEqual(experience["text"], "Company A\nBuilt APIs")
        self.assertEqual(experience["page_numbers"], [1, 2])
        self.assertEqual(experience["candidate_name"], "Sample Person")

    def test_prose_is_not_a_heading_and_unknown_text_survives(self):
        result = extract_sections({"pages": [{"page_number": 1,
            "text": "Sample Person\nSkills\nExperience with Python\nCustom topic\nOther text"}]})
        self.assertEqual([s["section"] for s in result], ["unclassified", "skills"])
        self.assertEqual(len(result[1]["lines"]), 3)

    def assert_coverage(self, result):
        for section in result["sections"]:
            chunks = [c for c in result["chunks"] if c["section_id"] == section["section_id"]]
            covered_until = 0
            for chunk in chunks:
                self.assertLessEqual(chunk["start_char"], covered_until)
                self.assertGreater(chunk["end_char"], covered_until)
                self.assertEqual(chunk["text"], section["text"][chunk["start_char"]:chunk["end_char"]])
                count = len(self.tokenizer.encode(chunk["embedding_text"]).ids)
                self.assertEqual(chunk["token_count"], count)
                self.assertLessEqual(count, MAX_TOKENS)
                self.assertTrue(chunk["page_numbers"])
                covered_until = chunk["end_char"]
            self.assertEqual(covered_until, len(section["text"]))

    def test_long_section_without_line_breaks(self):
        resume = {"source_file": "long.pdf", "pages": [{"page_number": 1,
            "text": "Sample Person\nExperience\n" + "Built Python APIs for multilingual applications. " * 160}]}
        result = process_resume(resume, self.tokenizer)
        self.assertGreater(len(result["chunks"]), 2)
        self.assert_coverage(result)

    def test_uploaded_resumes(self):
        paths = list((ROOT / "data" / "extracted").glob("*.json"))
        if not paths:
            self.skipTest("Local resumes are not included in version control.")
        for path in paths:
            with self.subTest(file=path.name):
                result = process_resume(json.loads(path.read_text()), self.tokenizer)
                self.assertTrue({"skills", "experience", "education", "projects"}.issubset(
                    {s["section"] for s in result["sections"]}))
                self.assert_coverage(result)


if __name__ == "__main__":
    unittest.main()
