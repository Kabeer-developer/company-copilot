# Company Knowledge Copilot — Backend

FastAPI backend for authenticated document ingestion, local embeddings, MongoDB Atlas Vector Search, Ollama grounded answers, and per-user conversations.

## Requirements

- Python 3.11 or newer
- MongoDB Atlas database user and reachable cluster
- Cloudinary account for authenticated document storage
- Tesseract OCR installed and available on `PATH` for image uploads
- Ollama installed, running, and serving the configured model for chat

Copy `.env.example` to `.env`, then replace its placeholders. Keep `.env` private. `JWT_SECRET_KEY` should be a long random secret. Public registration always creates an employee; create the first administrator with the bootstrap command below.

## Setup and startup (PowerShell)

From the `backend` directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set real local credentials before continuing.
python -m app.bootstrap_admin
python -m app.scripts.prepare_vector_search
ollama pull llama3.2:3b
```

Run Ollama in a separate terminal with `ollama serve`, then start the API from the backend directory:

```powershell
..venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

The API listens at `http://127.0.0.1:8000`; interactive OpenAPI documentation is at `/docs`. `GET /health` is a process health check and does not verify external dependencies.

## Atlas Vector Search setup

Before search, retrieval, or chat can work, run `python -m app.scripts.prepare_vector_search` after configuring MongoDB. It backfills each existing chunk's `uploaded_by` from its parent document and creates or verifies a Search index on `document_chunks` named by `VECTOR_SEARCH_INDEX_NAME` (default `chunk_vector_index`). The index definition must have:

- Vector field `embedding`, 384 dimensions, cosine similarity.
- Filter field `uploaded_by`.

The setup script waits for the index to become queryable and fails with an error if a same-named index has a different definition. If Atlas permissions do not allow index creation, create the index manually in Atlas Search using the definition above, then rerun the setup script to verify it. Existing orphan chunks without valid parent ownership are reported and are not assigned an owner.

## Configuration

`.env.example` lists the environment variables read by the application. `LLM_PROVIDER` must be `ollama`; model name, base URL, and timeout are configurable. Embedding uses the configured local Sentence Transformers model and downloads its model files on first use if not cached. Settings are validated at startup: retrieval top K is 1–20, similarity threshold is 0–1, batch size and timeout must be positive, and conversation history is 0–50 messages.

## Tests

Existing test infrastructure uses the Python standard library `unittest`; test doubles and mocks keep unit tests independent of paid or live services:

```powershell
python -m unittest discover -s tests -v
```

These tests do not establish a live Atlas, Cloudinary, OCR/Tesseract, embedding-model-download, or Ollama integration. Exercise those dependencies separately in a configured development environment.

## Manual Postman end-to-end pass

Use a local Postman environment with `baseUrl=http://127.0.0.1:8000`, `adminToken`, `employeeToken`, `otherEmployeeToken`, `documentId`, and `conversationId`. Keep tokens only in the local environment; do not export or commit them.

1. `GET {{baseUrl}}/health` → expect `200` and `{"status":"ok"}`. Open `GET {{baseUrl}}/docs` to inspect the current request/response schemas.
2. `POST /api/auth/login` with the bootstrapped administrator credentials; save `access_token` as `adminToken`. Create two employee accounts with `POST /api/auth/register`; log in both and save their tokens. Registration should return employee role even if an extra `role` field is submitted (the extra field is rejected).
3. As employee A, `POST /api/documents/upload` with a supported PDF, DOCX, TXT, PNG, or JPEG under 10 MB. Save `id` as `documentId`. Verify the response does not expose storage credentials or internal extracted text.
4. Poll `GET /api/documents/{{documentId}}` as the owner until document, chunking, and embedding statuses finish. A failed status should include the safe status fields and be investigated in server logs. Confirm chunks/vectors in Atlas and verify each chunk has the expected document and owner IDs.
5. As employee A, `POST /api/search` and `POST /api/retrieval` with a question grounded in the uploaded document. Check `sources` point to that document. As employee B, repeat the search/retrieval query and verify A's document text is never returned. An administrator may search across employees' documents per current policy.
6. As employee A, `POST /api/chat` with the same question; verify a grounded response, relevant source IDs, or the documented not-found answer when evidence is absent. Ollama must be running with the model named by `LLM_MODEL`.
7. Create a conversation with `POST /api/conversations`; save its `id` as `conversationId`. Send a first question and a follow-up with `POST /api/conversations/{{conversationId}}/messages`. Verify user/assistant messages and source metadata are persisted, and the follow-up is interpreted with bounded history.
8. Read the conversation via `GET /api/conversations/{{conversationId}}`, list via `GET /api/conversations`, then delete via `DELETE /api/conversations/{{conversationId}}`. Confirm the owner's message history is removed. Repeat read/send/delete as employee B and expect `404`.
9. Try employee B to `GET` and `DELETE` employee A's `documentId`; expect `404`. Try employee credentials on `/api/users`; expect `403`. Verify unauthenticated protected requests return `401` and invalid/oversized uploads are rejected (`415`/`413`).
10. Verify a missing/bad Atlas index produces a safe service-unavailable response for retrieval, and a stopped/missing Ollama model produces a safe `502`/`504` response for answer generation. Record these as failure-path checks, not successful external integrations.

Do not use a real customer or sensitive document in this manual pass. The automated tests mock provider/collection boundaries; the steps above are the separate end-to-end checks against your own development services.
