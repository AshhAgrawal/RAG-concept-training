import unittest

import numpy as np

from hybrid_retrieval import KeywordIndex, fuse_rankings, keyword_tokens, retrieve


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"text": "Twilio messaging webhooks", "candidate_name": "Alex"},
            {"text": "API backend services", "candidate_name": "Alex"},
            {"text": "Twilio contact center", "candidate_name": "Blair"},
            {"text": "Education physics degree", "candidate_name": "Blair"},
            {"text": "Kubernetes container deployment", "candidate_name": "Alex"},
        ]
        self.vectors = np.array([[0., 1.], [1., 0.], [0., 1.], [-1., 0.], [-1., 0.]])
        self.query = np.array([1., 0.])

    def test_tokenization_preserves_technical_terms(self):
        self.assertEqual(keyword_tokens("What is C++ C# Go Node.js SKU-X140?"),
                         ["c++", "c#", "go", "node.js", "sku-x140"])

    def test_lexical_evidence_promotes_chunk_and_respects_filters(self):
        result = retrieve(self.vectors, self.query, self.chunks, np.array([0, 1, 4]),
                          "What Twilio experience does Alex have?", candidate_filtered=True)
        self.assertEqual(result[0]["row"], 0)
        self.assertEqual(result[0]["matched_terms"], ["twilio"])
        self.assertNotIn(2, [item["row"] for item in result])

    def test_no_keywords_falls_back_to_semantic_order(self):
        rows = np.array([0, 1, 4])
        result = retrieve(self.vectors, self.query, self.chunks, rows, "unfindabletoken")
        self.assertEqual([item["row"] for item in result], [1, 0, 4])
        self.assertTrue(all(item["keyword_rank"] is None for item in result))
        self.assertEqual(retrieve(self.vectors, None, self.chunks, rows,
                                  "unfindabletoken", mode="keyword"), [])

    def test_rrf_uses_rank_not_raw_score(self):
        result = fuse_rankings([(7, 0.9), (3, 0.1)], [(3, 5000)], 2)
        self.assertEqual(result[0]["row"], 3)
        self.assertAlmostEqual(result[0]["rrf"], 1 / 62 + 1 / 61)

    def test_empty_corpus_and_stopwords(self):
        index = KeywordIndex([{"text": ""}, {"text": ""}])
        self.assertEqual(index.search("Twilio", np.array([0, 1]))[0], [])
        self.assertEqual(KeywordIndex(self.chunks).search("the and", np.array([0, 1]))[0], [])


if __name__ == "__main__":
    unittest.main()
