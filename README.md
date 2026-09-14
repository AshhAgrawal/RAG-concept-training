# Resume RAG

A command-line learning project that searches uploaded resume PDFs. Document processing, embeddings, and retrieval run locally. The current implementation returns relevant passages with source references; LLM-generated answers are the next planned stage.

## Implemented so far

| Implementation | What it does | Main file |
| --- | --- | --- |
| PDF extraction | Converts PDFs into readable text and JSON, preserving filenames and page numbers | [extract_resumes.py](extract_resumes.py) |
| Section-aware chunking | Recognizes headings such as skills and experience, then splits sections into chunks with token limits and overlap | [chunk_resumes.py](chunk_resumes.py) |
| Local embeddings | Uses MiniLM to create 384-dimensional vectors and saves them with aligned chunk metadata | [embed_resumes.py](embed_resumes.py) |
| Semantic retrieval | Embeds a question and ranks existing chunk vectors by cosine similarity | [search_resumes.py](search_resumes.py) |
| BM25 keyword retrieval | Searches the saved passage text for query terms | [hybrid_retrieval.py](hybrid_retrieval.py) |
| Hybrid retrieval | Combines semantic and keyword rankings using Reciprocal Rank Fusion; this is the default search mode | [hybrid_retrieval.py](hybrid_retrieval.py) |
| Search controls | Supports candidate and section filters, result count, and semantic/keyword/hybrid modes | [search_resumes.py](search_resumes.py) |
| Evaluation | Compares semantic and hybrid results on example questions and checks filtering, ranking, and index integrity | [evaluate_retrieval.py](evaluate_retrieval.py) |

The current dataset has six resumes and 54 chunks. Hybrid search reuses the existing embeddings; adding keyword search does not require generating new vectors.

## Design and learning guides

See **[design.md](design.md)** for the 2D architecture diagrams, a traced Twilio search, scoring explanations, run commands, setup, and limitations.

The iteration guides explain [section extraction and chunking](ITERATION_2.md), [embeddings](ITERATION_3.md), [semantic search](ITERATION_4.md), and [hybrid retrieval](ITERATION_5.md).

## Planned

Groq-based answer generation with citations and evidence checks. A frontend can follow later. FAISS and automatic document synchronization are not implemented.
