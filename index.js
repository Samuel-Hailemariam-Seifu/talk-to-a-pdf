require('dotenv').config();
const fs = require('fs');
const path = require('path');
const express = require('express');
const { PDFParse } = require('pdf-parse');
const Groq = require('groq-sdk');
const { logInteraction } = require('./db');
const { getEmbedding, cosineSimilarity } = require('./lib/embeddings');

const app = express();
const PORT = process.env.PORT || 3000;

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

app.use(express.json());

const PDF_PATH = path.join(__dirname, 'manual.pdf');

const CHUNK_SIZE = Number(process.env.RAG_CHUNK_SIZE) || 1000;
const CHUNK_OVERLAP = Number(process.env.RAG_CHUNK_OVERLAP) || 200;
const RETRIEVE_TOP_K = Number(process.env.RAG_RETRIEVE_TOP_K) || 5;
const RERANK_TOP_N = Number(process.env.RAG_RERANK_TOP_N) || 2;

/** @type {{ text: string, embedding: number[] }[]} */
let pdfChunks = [];
let pdfTextLoaded = false;
let pdfLoadError = null;

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
 * @returns {Promise<string[]>}
 */
async function rerankChunks(question, chunks, topN = RERANK_TOP_N) {
  if (chunks.length === 0) return [];
  if (chunks.length <= topN) return chunks.map((c) => c.text);

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
      selected.push(chunks[idx].text);
    }
  }
  if (selected.length === 0) return chunks.slice(0, topN).map((c) => c.text);
  return selected;
}

async function loadPdfOnce() {
  if (pdfTextLoaded || pdfLoadError) return;

  try {
    if (!fs.existsSync(PDF_PATH)) {
      throw new Error(`PDF file not found at ${PDF_PATH}`);
    }

    if (!process.env.OPENAI_API_KEY) {
      throw new Error(
        'OPENAI_API_KEY is required for embedding-based RAG. Add it to your .env (see .env.example).'
      );
    }

    const dataBuffer = fs.readFileSync(PDF_PATH);
    const parser = new PDFParse({ data: dataBuffer });
    const textResult = await parser.getText();
    await parser.destroy();

    const rawChunks = chunkTextWithOverlap(textResult.text || '', CHUNK_SIZE, CHUNK_OVERLAP);
    pdfChunks = [];

    for (let i = 0; i < rawChunks.length; i++) {
      const embedding = await getEmbedding(rawChunks[i]);
      pdfChunks.push({ text: rawChunks[i], embedding });
      if ((i + 1) % 10 === 0) console.log(`Embedded ${i + 1}/${rawChunks.length} chunks...`);
    }

    pdfTextLoaded = true;
    console.log(`Loaded PDF with ${pdfChunks.length} chunks (overlap=${CHUNK_OVERLAP}, embeddings ready).`);
  } catch (err) {
    pdfLoadError = err;
    console.error('Failed to load PDF:', err);
  }
}

app.post('/ask', async (req, res) => {
  try {
    const { question } = req.body || {};

    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: 'Request body must contain a string field "question".' });
    }

    await loadPdfOnce();

    if (pdfLoadError) {
      return res.status(500).json({
        error: pdfLoadError.message || 'Failed to load PDF. Check server logs for details.',
      });
    }

    if (!pdfTextLoaded || pdfChunks.length === 0) {
      return res.status(500).json({ error: 'PDF text is empty or could not be chunked.' });
    }

    const retrieved = await retrieveTopChunks(question, pdfChunks, RETRIEVE_TOP_K);
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

    logInteraction({
      question,
      answer,
      contextChunk: context,
    }).catch((err) => {
      console.error('Failed to log chat interaction:', err);
    });

    res.json({
      answer,
      contextPreview: selectedChunks[0] ? selectedChunks[0].slice(0, 300) + (selectedChunks[0].length > 300 ? '...' : '') : null,
    });
  } catch (err) {
    console.error('Error handling /ask:', err);
    const message = err?.message || 'Unexpected server error.';
    res.status(500).json({ error: message });
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

app.listen(PORT, () => {
  console.log(`PDF chat backend listening on http://localhost:${PORT}`);
  loadPdfOnce().catch((err) => {
    console.error('Error preloading PDF:', err);
  });
});
