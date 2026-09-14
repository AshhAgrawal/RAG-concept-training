"""Offline checks of filtering, vector ranking, and saved-index integrity."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from search_resumes import MODEL_NAME, load_index, rank_chunks, select_rows


class SearchTests(unittest.TestCase):
    def test_filtered_ranking_keeps_original_row_ids(self):
        vectors = np.array([[1., 0.], [0., 1.], [0.6, 0.8], [-1., 0.]])
        rows = np.array([1, 2, 3])
        results = rank_chunks(vectors, np.array([1., 0.]), rows, 2)
        self.assertEqual([row for row, score in results], [2, 1])
        self.assertAlmostEqual(results[0][1], 0.6)

    def test_candidate_and_section_filters(self):
        chunks = [
            {"candidate_name": "Ashutosh Agrawal", "section": "skills"},
            {"candidate_name": "Ashutosh Agrawal", "section": "experience"},
            {"candidate_name": "Aneesh Mokashi", "section": "experience"},
        ]
        self.assertEqual(select_rows(chunks, "ASHUTOSH", "experience").tolist(), [1])
        self.assertEqual(select_rows(chunks).tolist(), [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "not found"):
            select_rows(chunks, "Unknown Person")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            select_rows(chunks, "a")
        with self.assertRaisesRegex(ValueError, "No chunks"):
            select_rows(chunks, "Ashutosh", "publications")

    def test_top_k_and_question_validation(self):
        vectors = np.eye(2)
        rows = np.array([0, 1])
        self.assertEqual(len(rank_chunks(vectors, vectors[0], rows, 20)), 2)
        with self.assertRaises(ValueError):
            rank_chunks(vectors, vectors[0], rows, 0)
        with self.assertRaises(ValueError):
            rank_chunks(vectors, np.array([0., 0.]), rows, 1)
        with self.assertRaises(ValueError):
            rank_chunks(vectors, np.array([1., 0., 0.]), rows, 1)

    def test_changed_metadata_is_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            np.save(folder / "embeddings.npy", np.eye(2))
            (folder / "chunks.json").write_text(json.dumps([
                {"embedding_row": 0}, {"embedding_row": 1}]))
            config = {"model_name": MODEL_NAME, "model_revision": "example",
                      "chunk_count": 2, "dimensions": 2, "normalized": True,
                      "sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                 for name in ("embeddings.npy", "chunks.json")}}
            (folder / "config.json").write_text(json.dumps(config))
            self.assertEqual(load_index(folder)[0].shape, (2, 2))
            (folder / "chunks.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_index(folder)


if __name__ == "__main__":
    unittest.main()
