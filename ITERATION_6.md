# Iteration 6: answers through Groq

Open `.env` in this project and replace only the placeholder:

```dotenv
GROQ_API_KEY=paste_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

Put the real key in `.env`, not `.env.example`, Python code, or chat. `.env` is ignored by Git. It has already been created in this workspace; on a fresh checkout, copy `.env.example` to `.env`. The loader reads the file beside the script; exported environment values take precedence.

Run from the project folder:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python ask_resumes.py "What experience does Ashutosh have with Twilio?" --candidate Ashutosh --section experience
```

Inspect the exact prompt locally without a key or network request:

```bash
python ask_resumes.py "What experience does Ashutosh have with Twilio?" --candidate Ashutosh --dry-run
```

## Important calls

1. `load_dotenv()` reads local configuration. The script rejects a missing or placeholder key before retrieval, except in dry-run mode.
2. `retrieve_sources()` uses the saved index, candidate/section filters, cached MiniLM, and existing hybrid ranking. Four chunks are selected by default; `--top-k` accepts 1–8.
3. `build_messages()` labels passages S1, S2, etc., and packages the question and text as JSON. It includes no embedding vectors or similarity scores. A character-size check bounds context; it is not an exact Groq token count.
4. `client.chat.completions.create()` sends one request to Groq with JSON output requested. The prompt requires supported statements, source IDs, and an insufficient-evidence flag. The request has a 45-second timeout and no automatic retries. The output budget is 1,600 tokens; truncated responses are rejected.
5. `validate_answer()` checks format and that every statement cites retrieved source IDs. `render_answer()` adds source markers and resolves filenames/pages from our own metadata.

After retrieval, the question, selected passage text, and source metadata leave the computer for Groq. Extraction and embeddings remain local. No messaging or email tools are involved.

The default model appears in Groq's free-plan limit table. Availability and account limits can change; select a model available to your account using `GROQ_MODEL`. The CLI reports authentication, rate-limit, connection, and HTTP errors without printing the key or raw API response. Remaining on a free plan is an account setting, not a guarantee enforced by this script.

## Validation and limits

```bash
python -m unittest -v test_ask_resumes.py
```

Tests use a mocked HTTP transport, not real Groq credentials. They check the SDK request, citation rendering, malformed citations/JSON, empty evidence, incomplete responses, and context assembly. A dry run verifies local retrieval. Live model behavior still requires your API key and should be checked against supported, unsupported, and multi-passage questions.

Citation validation proves that a cited source was retrieved, not that it supports the claim. The prompt cannot guarantee grounding or resistance to document instructions. Top-k may omit employers, warnings, or other required evidence; this version does not expand whole sections automatically. When only some facts are supported, the prompt asks for a partial answer with an explicit evidence limitation. These behaviors require live evaluation.

References: [Groq quickstart](https://console.groq.com/docs/quickstart), [JSON output](https://console.groq.com/docs/structured-outputs), [free-plan limits](https://console.groq.com/docs/rate-limits).
