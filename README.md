### PDF Chat Backend (Node.js + Express)

This is a minimal Node.js backend that lets you **ask questions about a local `manual.pdf` file** using a simple RAG (Retrieval‑Augmented Generation) pattern with the **Groq API**.

It:
- **Parses a local PDF** with `pdf-parse`
- **Chunks** the PDF text into ~1000‑character segments
- **Selects the most relevant chunk** for a question using keyword matching
- **Sends that chunk as context** to the Groq Chat Completions API (e.g. `llama-3.3-70b-versatile`)
- Exposes a **`POST /ask`** endpoint: `{ "question": "..." }`
- Optionally **logs Q&A history** into a SQLite database using `knex`

---

### 1. Prerequisites

- **Node.js** 18+ installed
- **npm** installed
- A local PDF file named **`manual.pdf`** in the **project root** (same folder as `index.js`)
- A **Groq API key** ([get one at console.groq.com](https://console.groq.com))

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
GROQ_MODEL=llama-3.3-70b-versatile   # or llama-3.1-8b-instant, etc.
PORT=3000                           # optional, defaults to 3000
DB_FILE=./chat_history.db           # optional, SQLite file for logging chat history
```

**Required:**
- **`GROQ_API_KEY`** must be set to a valid Groq API key.

**Optional:**
- **`GROQ_MODEL`** (defaults to `llama-3.3-70b-versatile` if not set)
- **`PORT`** (defaults to `3000`)
- **`DB_FILE`** (defaults to `./chat_history.db`)

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
Loaded PDF with X chunks.
```

Health check:

```bash
curl http://localhost:3000/health
```

---

### 6. Using the `/ask` endpoint

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

Example JSON response:

```json
{
  "answer": "To reset the device, press and hold ...",
  "contextPreview": "The reset procedure is described in section 3.2 ..."
}
```

---

### 7. How the RAG logic works

- **PDF parsing**: On startup, the server reads `manual.pdf` and uses `pdf-parse` to extract `parsed.text`.
- **Chunking**: The text is split into **~1000‑character chunks**, roughly by sentence boundaries.
- **Retrieval**: When you call `/ask`, the server:
  - Scores each chunk by how many question keywords it contains (simple keyword matching).
  - Picks the **highest‑scoring chunk** as the most relevant context.
- **Generation**: The server calls the **Groq Chat Completions API** with:
  - A **system message** telling the model to rely on the provided context.
  - A **user message** that includes the chosen PDF chunk and your question.
- The model’s answer is returned as `answer`, plus the `contextPreview` chunk used.

This is intentionally simple, but matches the basic **RAG** pattern: *retrieve relevant context → augment the prompt → generate an answer*.

---

### 8. Database logging (bonus)

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

### 9. Notes and next steps

- This is a **minimal RAG example**. For better relevance:
  - Replace keyword scoring with **embeddings + vector search**.
  - Use libraries like **LangChain** or **LlamaIndex**.
- For production use, you should add:
  - Input validation / rate limiting
  - Authentication
  - Better error handling and logging

