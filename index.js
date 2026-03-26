require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { randomUUID } = require('crypto');
const express = require('express');
const { PDFParse } = require('pdf-parse');
const Groq = require('groq-sdk');
const multer = require('multer');
const jwt = require('jsonwebtoken');
const { logInteraction } = require('./db');
const { getEmbedding, cosineSimilarity } = require('./lib/embeddings');

const app = express();
const PORT = process.env.PORT || 3000;

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));
const upload = multer();

const PDF_PATH = path.join(__dirname, 'manual.pdf');

const CHUNK_SIZE = Number(process.env.RAG_CHUNK_SIZE) || 1000;
const CHUNK_OVERLAP = Number(process.env.RAG_CHUNK_OVERLAP) || 200;
const RETRIEVE_TOP_K = Number(process.env.RAG_RETRIEVE_TOP_K) || 5;
const RERANK_TOP_N = Number(process.env.RAG_RERANK_TOP_N) || 2;
const REQUIRE_AUTH = (process.env.REQUIRE_AUTH || 'false').toLowerCase() === 'true';
const AUTH_ALLOW_API_KEY = (process.env.AUTH_ALLOW_API_KEY || 'true').toLowerCase() === 'true';
const AUTH_ALLOW_JWT = (process.env.AUTH_ALLOW_JWT || 'true').toLowerCase() === 'true';
const AUTH_JWT_SECRET = process.env.AUTH_JWT_SECRET || '';
const API_KEYS = new Set(
  [process.env.AUTH_API_KEY, ...(process.env.AUTH_API_KEYS || '').split(',')]
    .filter(Boolean)
    .map((k) => k.trim())
    .filter(Boolean)
);

/** @type {Map<string, { chunks: { text: string, embedding: number[] }[], sourceName?: string, uploadedAt: string }>} */
const documentStore = new Map();
/** @type {Map<string, { role: 'user' | 'assistant', content: string }[]>} */
const sessionStore = new Map();
let defaultPdfLoaded = false;
let defaultPdfLoadError = null;

function unauthorized(res, message = 'Unauthorized.') {
  return res.status(401).json({ error: message });
}

function requireAuth(req, res, next) {
  if (!REQUIRE_AUTH) return next();

  const canUseApiKey = AUTH_ALLOW_API_KEY && API_KEYS.size > 0;
  const canUseJwt = AUTH_ALLOW_JWT && !!AUTH_JWT_SECRET;
  if (!canUseApiKey && !canUseJwt) {
    return res.status(500).json({
      error: 'Authentication enabled but not configured. Set AUTH_API_KEY/AUTH_API_KEYS and/or AUTH_JWT_SECRET.',
    });
  }

  const xApiKey = req.get('x-api-key');
  if (canUseApiKey && xApiKey && API_KEYS.has(xApiKey.trim())) {
    req.auth = { type: 'api_key' };
    return next();
  }

  const authHeader = req.get('authorization') || '';
  if (authHeader.startsWith('Bearer ')) {
    const token = authHeader.slice('Bearer '.length).trim();

    if (canUseApiKey && API_KEYS.has(token)) {
      req.auth = { type: 'api_key' };
      return next();
    }

    if (canUseJwt) {
      try {
        const payload = jwt.verify(token, AUTH_JWT_SECRET);
        req.auth = { type: 'jwt', payload };
        return next();
      } catch (err) {
        return unauthorized(res, 'Invalid authentication token.');
      }
    }
  }

  return unauthorized(res, 'Missing or invalid authentication credentials.');
}

/**
 * Chunk text with overlapping windows (e.g. 1000 chars with 200 overlap).
 * @param {string} text
 * @param {number} maxLength
 * @param {number} overlap
 * @returns {string[]}
 */
function chunkTextWithOverlap(text, maxLength = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const chunks = [];
  const step = Math.max(1, maxLength - overlap);
  for (let start = 0; start < text.length; start += step) {
    const chunk = text.slice(start, start + maxLength).trim();
    if (chunk) chunks.push(chunk);
    if (start + maxLength >= text.length) break;
  }
  return chunks;
}

/**
 * Retrieve top K chunks by embedding similarity.
 * @param {string} question
 * @param {{ text: string, embedding: number[] }[]} chunks
 * @param {number} k
 * @returns {Promise<{ text: string, index: number }[]>}
 */
async function retrieveTopChunks(question, chunks, k = RETRIEVE_TOP_K) {
  const qEmbedding = await getEmbedding(question);
  const scored = chunks.map((c, index) => ({
    text: c.text,
    index,
    score: cosineSimilarity(qEmbedding, c.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k).map(({ text, index }) => ({ text, index }));
}

/**
 * Re-rank retrieved chunks with LLM: pick best 1–2 for final context.
 * @param {string} question
 * @param {{ text: string, index: number }[]} chunks
 * @param {number} topN
 * @returns {Promise<{ text: string, index: number }[]>}
 */
async function rerankChunks(question, chunks, topN = RERANK_TOP_N) {
  if (chunks.length === 0) return [];
  if (chunks.length <= topN) return chunks.slice(0, topN).map((c) => ({ text: c.text, index: c.index }));

  const passageList = chunks
    .map((c, i) => `[${i + 1}]\n${c.text.slice(0, 400)}${c.text.length > 400 ? '...' : ''}`)
    .join('\n\n');

  const model = (process.env.GROQ_MODEL || 'llama-3.3-70b-versatile').replace(/^groq\//i, '');
  const modelId = model === 'llama-3.1-8b-versatile' ? 'llama-3.1-8b-instant' : model;

  const completion = await groq.chat.completions.create({
    model: modelId,
    messages: [
      {
        role: 'system',
        content:
          'You are a relevance judge. Given a question and numbered passages, reply with only the numbers of the most relevant passages, separated by commas, in order of relevance (e.g. "2, 5"). Use only the numbers that exist. No explanation.',
      },
      {
        role: 'user',
        content: `Question: ${question}\n\nPassages:\n${passageList}\n\nWhich ${Math.min(topN, chunks.length)} passage number(s) are most relevant? Reply with that many numbers, comma-separated.`,
      },
    ],
    temperature: 0,
    max_tokens: 20,
  });

  const reply = (completion.choices?.[0]?.message?.content ?? '').trim();
  const numbers = reply
    .replace(/\s/g, '')
    .split(',')
    .map((s) => parseInt(s, 10))
    .filter((n) => !Number.isNaN(n) && n >= 1 && n <= chunks.length)
    .slice(0, topN);

  const seen = new Set();
  const selected = [];
  for (const n of numbers) {
    const idx = n - 1;
    if (!seen.has(idx)) {
      seen.add(idx);
      selected.push({ text: chunks[idx].text, index: chunks[idx].index });
    }
  }
  if (selected.length === 0) return chunks.slice(0, topN).map((c) => ({ text: c.text, index: c.index }));
  return selected;
}

async function processPdfBuffer(dataBuffer, sourceName = 'uploaded.pdf') {
  const parser = new PDFParse({ data: dataBuffer });
  const textResult = await parser.getText();
  await parser.destroy();

  const rawChunks = chunkTextWithOverlap(textResult.text || '', CHUNK_SIZE, CHUNK_OVERLAP);
  if (rawChunks.length === 0) {
    throw new Error('PDF text is empty or could not be chunked.');
  }

  const chunks = [];
  for (let i = 0; i < rawChunks.length; i++) {
    const embedding = await getEmbedding(rawChunks[i]);
    chunks.push({ text: rawChunks[i], embedding });
    if ((i + 1) % 10 === 0) console.log(`Embedded ${i + 1}/${rawChunks.length} chunks for ${sourceName}...`);
  }

  return chunks;
}

async function loadDefaultPdfOnce() {
  if (defaultPdfLoaded || defaultPdfLoadError) return;

  try {
    if (!fs.existsSync(PDF_PATH)) {
      throw new Error(`PDF file not found at ${PDF_PATH}`);
    }

    const dataBuffer = fs.readFileSync(PDF_PATH);
    const chunks = await processPdfBuffer(dataBuffer, path.basename(PDF_PATH));
    documentStore.set('default', {
      chunks,
      sourceName: path.basename(PDF_PATH),
      uploadedAt: new Date().toISOString(),
    });
    defaultPdfLoaded = true;
    console.log(`Loaded default PDF with ${chunks.length} chunks (overlap=${CHUNK_OVERLAP}, embeddings ready).`);
  } catch (err) {
    defaultPdfLoadError = err;
    if (err?.message && err.message.includes('PDF file not found')) {
      console.warn('No default manual.pdf found. Upload a PDF via /upload to start querying documents.');
    } else {
      console.error('Failed to load default PDF:', err);
    }
  }
}

app.post('/upload', requireAuth, upload.single('file'), async (req, res) => {
  try {
    let fileBuffer = req.file?.buffer;
    let sourceName = req.file?.originalname || 'uploaded.pdf';

    if (!fileBuffer) {
      const { file_base64, filename } = req.body || {};
      if (typeof file_base64 === 'string' && file_base64.trim()) {
        const cleanBase64 = file_base64.includes(',') ? file_base64.split(',').pop() : file_base64;
        fileBuffer = Buffer.from(cleanBase64, 'base64');
        sourceName = typeof filename === 'string' && filename.trim() ? filename.trim() : 'uploaded.pdf';
      }
    }

    if (!fileBuffer || fileBuffer.length === 0) {
      return res.status(400).json({
        error: 'Upload a PDF via multipart field "file" or JSON body field "file_base64".',
      });
    }

    const documentId = randomUUID();
    const chunks = await processPdfBuffer(fileBuffer, sourceName);
    documentStore.set(documentId, {
      chunks,
      sourceName,
      uploadedAt: new Date().toISOString(),
    });

    res.status(201).json({
      document_id: documentId,
      source_name: sourceName,
      chunk_count: chunks.length,
    });
  } catch (err) {
    console.error('Error handling /upload:', err);
    res.status(400).json({ error: err?.message || 'Failed to process uploaded PDF.' });
  }
});

app.post('/ask', requireAuth, async (req, res) => {
  try {
    const { question, document_id, session_id, messages } = req.body || {};

    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: 'Request body must contain a string field "question".' });
    }

    const requestedDocumentId = resolveRequestedDocumentId(document_id);
    const { selectedChunks, context, sources } = await resolveDocumentContext(question, requestedDocumentId);
    const { systemPrompt, userPrompt } = buildAskPrompts(question, context);

    let model = (process.env.GROQ_MODEL || 'llama-3.3-70b-versatile').replace(/^groq\//i, '');
    if (model === 'llama-3.1-8b-versatile') model = 'llama-3.1-8b-instant';

    const { sessionId, priorMessages } = resolveConversationContext(session_id, messages);

    const completion = await groq.chat.completions.create({
      model,
      messages: [
        { role: 'system', content: systemPrompt },
        ...priorMessages,
        { role: 'user', content: userPrompt },
      ],
      temperature: 0.2,
    });

    const answer = completion.choices?.[0]?.message?.content?.trim() || '';
    saveSessionTurn(sessionId, priorMessages, question, answer);

    logInteraction({
      documentId: requestedDocumentId,
      sessionId,
      question,
      answer,
      contextChunk: context,
      sources,
    }).catch((err) => {
      console.error('Failed to log chat interaction:', err);
    });

    res.json({
      answer,
      document_id: requestedDocumentId,
      session_id: sessionId,
      contextPreview: selectedChunks[0]
        ? selectedChunks[0].text.slice(0, 300) + (selectedChunks[0].text.length > 300 ? '...' : '')
        : null,
      sources,
    });
  } catch (err) {
    console.error('Error handling /ask:', err);
    const message = err?.message || 'Unexpected server error.';
    res.status(err?.statusCode || 500).json({ error: message });
  }
});

app.post('/ask/stream', async (req, res) => {
  try {
    const { question, document_id, session_id, messages } = req.body || {};

    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: 'Request body must contain a string field "question".' });
    }

    const requestedDocumentId = resolveRequestedDocumentId(document_id);
    const { selectedChunks, context, sources } = await resolveDocumentContext(question, requestedDocumentId);
    const { systemPrompt, userPrompt } = buildAskPrompts(question, context);
    const { sessionId, priorMessages } = resolveConversationContext(session_id, messages);

    let model = (process.env.GROQ_MODEL || 'llama-3.3-70b-versatile').replace(/^groq\//i, '');
    if (model === 'llama-3.1-8b-versatile') model = 'llama-3.1-8b-instant';

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const stream = await groq.chat.completions.create({
      model,
      stream: true,
      messages: [
        { role: 'system', content: systemPrompt },
        ...priorMessages,
        { role: 'user', content: userPrompt },
      ],
      temperature: 0.2,
    });

    let answer = '';
    let clientClosed = false;
    req.on('close', () => {
      clientClosed = true;
    });

    for await (const chunk of stream) {
      if (clientClosed) break;
      const delta = chunk?.choices?.[0]?.delta?.content || '';
      if (!delta) continue;
      answer += delta;
      res.write(`event: token\ndata: ${JSON.stringify({ token: delta })}\n\n`);
    }

    if (!clientClosed) {
      saveSessionTurn(sessionId, priorMessages, question, answer);
      await logInteraction({
        documentId: requestedDocumentId,
        sessionId,
        question,
        answer,
        contextChunk: context,
        sources,
      });

      res.write(
        `event: done\ndata: ${JSON.stringify({
          answer,
          document_id: requestedDocumentId,
          session_id: sessionId,
          contextPreview: selectedChunks[0]
            ? selectedChunks[0].text.slice(0, 300) + (selectedChunks[0].text.length > 300 ? '...' : '')
            : null,
          sources,
        })}\n\n`
      );
      res.end();
    }
  } catch (err) {
    console.error('Error handling /ask/stream:', err);
    if (!res.headersSent) {
      res.status(err?.statusCode || 500).json({ error: err?.message || 'Unexpected server error.' });
      return;
    }
    res.write(`event: error\ndata: ${JSON.stringify({ error: err?.message || 'Unexpected server error.' })}\n\n`);
    res.end();
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.listen(PORT, () => {
  console.log(`PDF chat backend listening on http://localhost:${PORT}`);
  loadDefaultPdfOnce().catch((err) => {
    console.error('Error preloading PDF:', err);
  });
});
