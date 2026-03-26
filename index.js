require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { randomUUID } = require('crypto');
const express = require('express');
const { PDFParse } = require('pdf-parse');
const Groq = require('groq-sdk');
const multer = require('multer');
const rateLimit = require('express-rate-limit');
const jwt = require('jsonwebtoken');
const { logInteraction, getHistory } = require('./db');
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
const READY_CHECK_GROQ = (process.env.READY_CHECK_GROQ || 'false').toLowerCase() === 'true';
const REQUIRE_AUTH = (process.env.REQUIRE_AUTH || 'false').toLowerCase() === 'true';
const AUTH_ALLOW_API_KEY = (process.env.AUTH_ALLOW_API_KEY || 'true').toLowerCase() === 'true';
const AUTH_ALLOW_JWT = (process.env.AUTH_ALLOW_JWT || 'true').toLowerCase() === 'true';
const AUTH_JWT_SECRET = process.env.AUTH_JWT_SECRET || '';
const ASK_RATE_LIMIT_WINDOW_MS = Number(process.env.ASK_RATE_LIMIT_WINDOW_MS) || 60 * 1000;
const ASK_RATE_LIMIT_IP_MAX = Number(process.env.ASK_RATE_LIMIT_IP_MAX) || 60;
const ASK_RATE_LIMIT_API_KEY_MAX = Number(process.env.ASK_RATE_LIMIT_API_KEY_MAX) || 120;
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

function getRawBearerToken(req) {
  const authHeader = req.get('authorization') || '';
  if (!authHeader.startsWith('Bearer ')) return null;
  return authHeader.slice('Bearer '.length).trim();
}

function getProvidedApiKey(req) {
  const xApiKey = req.get('x-api-key');
  if (xApiKey && API_KEYS.has(xApiKey.trim())) return xApiKey.trim();
  const bearer = getRawBearerToken(req);
  if (bearer && API_KEYS.has(bearer)) return bearer;
  return null;
}

function makeLimiter(maxRequests, namespace) {
  return rateLimit({
    windowMs: ASK_RATE_LIMIT_WINDOW_MS,
    max: maxRequests,
    standardHeaders: true,
    legacyHeaders: false,
    keyGenerator: (req) => {
      const apiKey = getProvidedApiKey(req);
      if (apiKey) return `${namespace}:key:${apiKey}`;
      return `${namespace}:ip:${req.ip}`;
    },
    handler: (req, res) => {
      const hasApiKey = !!getProvidedApiKey(req);
      const scope = hasApiKey ? 'API key' : 'IP';
      return res.status(429).json({
        error: `Rate limit exceeded for ${scope}. Please retry later.`,
      });
    },
  });
}

const askLimiterByIp = makeLimiter(ASK_RATE_LIMIT_IP_MAX, 'ask');
const askLimiterByApiKey = makeLimiter(ASK_RATE_LIMIT_API_KEY_MAX, 'ask');
function askRateLimit(req, res, next) {
  if (getProvidedApiKey(req)) return askLimiterByApiKey(req, res, next);
  return askLimiterByIp(req, res, next);
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

function getDocumentList() {
  return Array.from(documentStore.entries()).map(([documentId, doc]) => ({
    document_id: documentId,
    source_name: doc.sourceName || null,
    chunk_count: doc.chunks.length,
    uploaded_at: doc.uploadedAt,
  }));
}

async function handleAsk(question, requestedDocumentId) {
  await loadDefaultPdfOnce();

  if (defaultPdfLoadError) {
    const err = new Error(defaultPdfLoadError.message || 'Failed to load PDF. Check server logs for details.');
    err.statusCode = 500;
    throw err;
  }

  const doc = documentStore.get(requestedDocumentId);
  if (!doc) {
    const err = new Error(`Document not found for document_id "${requestedDocumentId}".`);
    err.statusCode = 404;
    throw err;
  }

  if (doc.chunks.length === 0) {
    const err = new Error('Document has no available chunks.');
    err.statusCode = 500;
    throw err;
  }

  const retrieved = await retrieveTopChunks(question, doc.chunks, RETRIEVE_TOP_K);
  const selectedChunks = await rerankChunks(question, retrieved, RERANK_TOP_N);
  const context = selectedChunks.join('\n\n---\n\n');

  const systemPrompt =
    'You are a helpful assistant that answers questions based primarily on the provided PDF manual context. ' +
    'If the answer is not clearly supported by the context, say you are not sure and avoid making things up.';

  const userPrompt = [
    'PDF Context:',
    '"""',
    context || 'No relevant context found.',
    '"""',
    '',
    'User question:',
    question,
  ].join('\n');

  let model = (process.env.GROQ_MODEL || 'llama-3.3-70b-versatile').replace(/^groq\//i, '');
  if (model === 'llama-3.1-8b-versatile') model = 'llama-3.1-8b-instant';

  const completion = await groq.chat.completions.create({
    model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ],
    temperature: 0.2,
  });

  const answer = completion.choices?.[0]?.message?.content?.trim() || '';
  return { answer, context, selectedChunks };
}

app.post('/upload', upload.single('file'), async (req, res) => {
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

app.post('/ask', askRateLimit, requireAuth, async (req, res) => {
  try {
    const { question, document_id, session_id, messages } = req.body || {};

    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: 'Request body must contain a string field "question".' });
    }

    const requestedDocumentId =
      typeof document_id === 'string' && document_id.trim() ? document_id.trim() : 'default';
    const { answer, context, selectedChunks } = await handleAsk(question, requestedDocumentId);

    logInteraction({
      documentId: requestedDocumentId,
      question,
      answer,
      contextChunk: context,
    }).catch((err) => {
      console.error('Failed to log chat interaction:', err);
    });

    res.json({
      answer,
      document_id: requestedDocumentId,
      contextPreview: selectedChunks[0] ? selectedChunks[0].slice(0, 300) + (selectedChunks[0].length > 300 ? '...' : '') : null,
    });
  } catch (err) {
    console.error('Error handling /ask:', err);
    const message = err?.message || 'Unexpected server error.';
    res.status(err?.statusCode || 500).json({ error: message });
  }
});

app.post('/documents/:id/ask', async (req, res) => {
  try {
    const { question } = req.body || {};
    const requestedDocumentId = req.params.id;
    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: 'Request body must contain a string field "question".' });
    }

    const { answer, context, selectedChunks } = await handleAsk(question, requestedDocumentId);
    logInteraction({
      documentId: requestedDocumentId,
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
    console.error('Error handling /documents/:id/ask:', err);
    res.status(err?.statusCode || 500).json({ error: err?.message || 'Unexpected server error.' });
  }
});

app.get('/documents', async (req, res) => {
  try {
    await loadDefaultPdfOnce();
    if (defaultPdfLoadError && documentStore.size === 0) {
      return res.status(500).json({
        error: defaultPdfLoadError.message || 'Failed to load default PDF.',
      });
    }

    res.json({ documents: getDocumentList() });
  } catch (err) {
    console.error('Error handling /documents:', err);
    res.status(500).json({ error: err?.message || 'Unexpected server error.' });
  }
});

app.get('/history', requireAuth, async (req, res) => {
  try {
    const maxLimit = 200;
    const rawLimit = req.query.limit == null ? 50 : Number(req.query.limit);
    const rawOffset = req.query.offset == null ? 0 : Number(req.query.offset);
    const limit = Number.isFinite(rawLimit) ? Math.min(maxLimit, Math.max(1, Math.floor(rawLimit))) : NaN;
    const offset = Number.isFinite(rawOffset) ? Math.max(0, Math.floor(rawOffset)) : NaN;
    if (Number.isNaN(limit) || Number.isNaN(offset)) {
      return res.status(400).json({ error: 'Invalid limit/offset. Use numeric values.' });
    }

    const since = typeof req.query.since === 'string' && req.query.since.trim() ? req.query.since.trim() : null;
    if (since && Number.isNaN(new Date(since).getTime())) {
      return res.status(400).json({ error: 'Invalid since value. Use an ISO date string.' });
    }

    const questionContains =
      typeof req.query.question_contains === 'string' && req.query.question_contains.trim()
        ? req.query.question_contains.trim()
        : null;
    const documentId =
      typeof req.query.document_id === 'string' && req.query.document_id.trim()
        ? req.query.document_id.trim()
        : null;
    const sessionId =
      typeof req.query.session_id === 'string' && req.query.session_id.trim()
        ? req.query.session_id.trim()
        : null;

    const items = await getHistory({
      limit,
      offset,
      since,
      questionContains,
      documentId,
      sessionId,
    });

    res.json({
      items,
      pagination: {
        limit,
        offset,
        count: items.length,
        max_limit: maxLimit,
      },
    });
  } catch (err) {
    console.error('Error handling /history:', err);
    res.status(500).json({ error: err?.message || 'Unexpected server error.' });
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.get('/ready', async (req, res) => {
  try {
    await loadDefaultPdfOnce();

    if (defaultPdfLoadError) {
      return res.status(503).json({
        status: 'not_ready',
        reason: defaultPdfLoadError.message || 'Default PDF failed to load.',
      });
    }

    const defaultDoc = documentStore.get('default');
    if (!defaultDoc || !Array.isArray(defaultDoc.chunks) || defaultDoc.chunks.length === 0) {
      return res.status(503).json({
        status: 'not_ready',
        reason: 'Default PDF is not loaded or has no chunks.',
      });
    }

    if (READY_CHECK_GROQ) {
      if (!process.env.GROQ_API_KEY) {
        return res.status(503).json({
          status: 'not_ready',
          reason: 'READY_CHECK_GROQ is enabled but GROQ_API_KEY is missing.',
        });
      }
      await groq.models.list();
    }

    res.json({
      status: 'ready',
      checks: {
        default_pdf_loaded: true,
        default_pdf_chunk_count: defaultDoc.chunks.length,
        groq_checked: READY_CHECK_GROQ,
      },
    });
  } catch (err) {
    res.status(503).json({
      status: 'not_ready',
      reason: err?.message || 'Readiness check failed.',
    });
  }
});

app.listen(PORT, () => {
  console.log(`PDF chat backend listening on http://localhost:${PORT}`);
  loadDefaultPdfOnce().catch((err) => {
    console.error('Error preloading PDF:', err);
  });
});
