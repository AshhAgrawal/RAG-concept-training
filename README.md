# Resume RAG

A command-line learning project that searches uploaded resume PDFs and generates cited answers through Groq. Document processing, embeddings, and retrieval run locally; answer generation sends selected passages and the question to Groq.

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
| LLM answers | Sends retrieved evidence to Groq, checks response structure and source IDs, and prints cited statements | [ask_resumes.py](ask_resumes.py) |

The current dataset has six resumes and 54 chunks. Hybrid search reuses the existing embeddings; adding keyword search does not require generating new vectors.

## Design and learning guides

See **[design.md](design.md)** for the 2D architecture diagrams, a traced Twilio search, scoring explanations, run commands, setup, and limitations.

The iteration guides explain [section extraction and chunking](ITERATION_2.md), [embeddings](ITERATION_3.md), [semantic search](ITERATION_4.md), [hybrid retrieval](ITERATION_5.md), and [Groq setup and answers](ITERATION_6.md). Paste your API key in the local `.env` file before running the answer command.

## Planned

Live answer evaluation after configuring a Groq key, stronger evidence checks, and a future frontend. FAISS and automatic document synchronization are not implemented.
