### PDF Chat Backend (Node.js + Express)

This is a minimal Node.js backend that lets you **ask questions about a local `manual.pdf` file or uploaded PDFs** using a simple RAG (Retrieval‑Augmented Generation) pattern with the **Groq API**.

It:
- **Parses a local PDF** with `pdf-parse`
- **Chunks** the PDF with **overlapping windows** (~1000 chars, 200 char overlap) so context isn’t cut mid-sentence
- **Embeds** each chunk with OpenAI (`text-embedding-3-small`) and **retrieves** the top K chunks by **cosine similarity** to the question
- **Re-ranks** those chunks with a quick Groq call to pick the best 1–2 for the final answer
- **Sends the selected chunks** as context to the Groq Chat Completions API (e.g. `llama-3.3-70b-versatile`)
- Exposes a **`POST /upload`** endpoint to upload a PDF (multipart `file` or JSON `file_base64`) and receive a `document_id`
- Exposes a **`POST /ask`** endpoint: `{ "question": "..." }`
- Optionally **logs Q&A history** into a SQLite database using `knex`

---

### 1. Prerequisites

- **Node.js** 18+ installed
- **npm** installed
- A local PDF file named **`manual.pdf`** in the **project root** (same folder as `index.js`)
- A **Groq API key** ([get one at console.groq.com](https://console.groq.com)) for chat (and re-ranking)
- An **OpenAI API key** for **embeddings** (used for vector retrieval)

---

### 2. Install dependencies

From the project root:

```bash
npm install
```

If you don’t have `node_modules` yet (for example on a fresh clone), also run:

```bash
npm install express pdf-parse groq-sdk dotenv knex sqlite3
```

---

### 3. `.env` setup

Create a `.env` file in the project root, based on `.env.example`:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL=llama-3.3-70b-versatile

OPENAI_API_KEY=your-openai-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Optional RAG tuning
# RAG_CHUNK_SIZE=1000
# RAG_CHUNK_OVERLAP=200
# RAG_RETRIEVE_TOP_K=5
# RAG_RERANK_TOP_N=2

PORT=3000
DB_FILE=./chat_history.db
```

**Required:**
- **`GROQ_API_KEY`** – Groq API key for chat and re-ranking.
- **`OPENAI_API_KEY`** – OpenAI API key for embeddings (vector retrieval).

**Optional:**
- **`GROQ_MODEL`** (defaults to `llama-3.3-70b-versatile`)
- **`OPENAI_EMBEDDING_MODEL`** (defaults to `text-embedding-3-small`)
- **`RAG_CHUNK_SIZE`** / **`RAG_CHUNK_OVERLAP`** – chunk size and overlap in characters.
- **`RAG_RETRIEVE_TOP_K`** – number of chunks to retrieve by similarity (default 5).
- **`RAG_RERANK_TOP_N`** – number of chunks to keep after re-ranking for the final prompt (default 2).
- **`PORT`** (defaults to `3000`), **`DB_FILE`** (defaults to `./chat_history.db`)

---

### 4. `manual.pdf` placement

Place your PDF file as:

```text
project-root/
  index.js
  manual.pdf   <-- here
  ...
```

The server will attempt to load `manual.pdf` on startup and chunk it for later retrieval.

---

### 5. Running the server

Development (with `nodemon`, if you install it globally or add it as a dev dependency):

```bash
npm run dev
```

Or just run with Node:

```bash
npm start
```

You should see something like:

```text
PDF chat backend listening on http://localhost:3000
Loaded PDF with X chunks (overlap=200, embeddings ready).
```

Health check:

```bash
curl http://localhost:3000/health
```

---

### 6. Uploading PDFs with `/upload`

Endpoint:
- **Method**: `POST`
- **URL**: `http://localhost:3000/upload`

Option A (multipart form):

```bash
curl -X POST http://localhost:3000/upload \
  -F "file=@manual.pdf"
```

Option B (base64 JSON):

```bash
curl -X POST http://localhost:3000/upload \
  -H "Content-Type: application/json" \
  -d "{\"filename\":\"manual.pdf\",\"file_base64\":\"<BASE64_PDF_CONTENT>\"}"
```

Example response:

```json
{
  "document_id": "f3c6f5b2-9f87-4ebd-8a9d-9ca2c3d8dbf9",
  "source_name": "manual.pdf",
  "chunk_count": 48
}
```

Uploaded documents are processed in-memory for the current server session.

---

### 7. Using the `/ask` endpoint

Endpoint:
- **Method**: `POST`
- **URL**: `http://localhost:3000/ask`
- **Body** (JSON):

```json
{
  "question": "How do I reset the device?"
}
```

Example with `curl`:

```bash
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"How do I reset the device?\"}"
```

Ask against a previously uploaded document:

```bash
curl -X POST http://localhost:3000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What are the warranty terms?\", \"document_id\":\"<YOUR_DOCUMENT_ID>\"}"
```

Example JSON response:

```json
{
  "answer": "To reset the device, press and hold ...",
  "document_id": "f3c6f5b2-9f87-4ebd-8a9d-9ca2c3d8dbf9",
  "contextPreview": "The reset procedure is described in section 3.2 ..."
}
```

---

### 8. How the RAG logic works

- **PDF parsing**: On startup, the server reads `manual.pdf`; uploaded files can also be added via `/upload`.
- **Chunking with overlap**: The text is split into **overlapping windows** (e.g. 1000 characters, 200 character overlap) so boundaries don’t cut mid-sentence.
- **Embeddings**: Each chunk is embedded with **OpenAI** (`text-embedding-3-small`); vectors are kept in memory for similarity search.
- **Retrieval**: When you call `/ask`, the server:
  - Embeds the question, then **retrieves the top K chunks** (default 5) by **cosine similarity**.
  - **Re-ranks** those K chunks with a fast Groq call that picks the best 1–2 passages for the answer.
- **Generation**: The server calls the **Groq Chat Completions API** with the re-ranked chunks as context and your question; the model’s reply is returned as `answer`, with a short `contextPreview`.

This is intentionally simple, but matches the basic **RAG** pattern: *retrieve relevant context → augment the prompt → generate an answer*.

---

### 9. Database logging (bonus)

This project includes a small `db.js` module using **Knex** with **SQLite**:

- File: `db.js`
- Table: `chat_history`
  - `id` (auto‑increment)
  - `created_at` (timestamp)
  - `question` (text)
  - `answer` (text)
  - `context_chunk` (text)

On each successful `/ask` call, the server:

- Ensures the `chat_history` table exists
- Inserts a new row with the `question`, `answer`, and the `context_chunk` used

You can inspect the SQLite file (by default `chat_history.db`) with any SQLite client, or with a basic CLI like:

```bash
sqlite3 chat_history.db "SELECT id, created_at, substr(question, 1, 80) AS q FROM chat_history ORDER BY id DESC LIMIT 10;"
```

If you want to adapt this to another SQL database (PostgreSQL, MySQL, etc.), adjust the `knex` configuration in `db.js` accordingly.

---

### 10. Notes and next steps

- This is a **minimal RAG example**. For better relevance:
  - Replace keyword scoring with **embeddings + vector search**.
  - Use libraries like **LangChain** or **LlamaIndex**.
- For production use, you should add:
  - Input validation / rate limiting
  - Authentication
  - Better error handling and logging

