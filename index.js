require('dotenv').config();
const fs = require('fs');
const path = require('path');
const express = require('express');
const { PDFParse } = require('pdf-parse');
const OpenAI = require('openai');
const { logInteraction } = require('./db');

const app = express();
const PORT = process.env.PORT || 3000;

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

app.use(express.json());

const PDF_PATH = path.join(__dirname, 'manual.pdf');

let pdfChunks = [];
let pdfTextLoaded = false;
let pdfLoadError = null;

function chunkText(text, maxLength = 1000) {
  const chunks = [];
  let current = '';

  const sentences = text.split(/(?<=[\.!\?])\s+/);

  for (const sentence of sentences) {
    if ((current + ' ' + sentence).trim().length <= maxLength) {
      current = (current + ' ' + sentence).trim();
    } else {
      if (current) chunks.push(current);
      if (sentence.length > maxLength) {
        for (let i = 0; i < sentence.length; i += maxLength) {
          chunks.push(sentence.slice(i, i + maxLength));
        }
        current = '';
      } else {
        current = sentence;
      }
    }
  }

  if (current) chunks.push(current);
  return chunks;
}

function scoreChunk(question, chunk) {
  const qTokens = question
    .toLowerCase()
    .split(/\W+/)
    .filter(Boolean);

  const chunkLower = chunk.toLowerCase();
  let score = 0;

  for (const token of qTokens) {
    if (chunkLower.includes(token)) {
      score += 1;
    }
  }

  return score;
}

function findBestChunk(question, chunks) {
  if (!chunks || chunks.length === 0) return null;

  let bestChunk = chunks[0];
  let bestScore = scoreChunk(question, bestChunk);

  for (let i = 1; i < chunks.length; i++) {
    const s = scoreChunk(question, chunks[i]);
    if (s > bestScore) {
      bestScore = s;
      bestChunk = chunks[i];
    }
  }

  return bestChunk;
}

async function loadPdfOnce() {
  if (pdfTextLoaded || pdfLoadError) return;

  try {
    if (!fs.existsSync(PDF_PATH)) {
      throw new Error(`PDF file not found at ${PDF_PATH}`);
    }

    const dataBuffer = fs.readFileSync(PDF_PATH);
    const parser = new PDFParse({ data: dataBuffer });
    const textResult = await parser.getText();
    await parser.destroy();

    pdfChunks = chunkText(textResult.text || '');
    pdfTextLoaded = true;
    console.log(`Loaded PDF with ${pdfChunks.length} chunks.`);
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
      return res.status(500).json({ error: 'Failed to load PDF. Check server logs for details.' });
    }

    if (!pdfTextLoaded || pdfChunks.length === 0) {
      return res.status(500).json({ error: 'PDF text is empty or could not be chunked.' });
    }

    const bestChunk = findBestChunk(question, pdfChunks);

    const systemPrompt =
      'You are a helpful assistant that answers questions based primarily on the provided PDF manual context. ' +
      'If the answer is not clearly supported by the context, say you are not sure and avoid making things up.';

    const userPrompt = [
      'PDF Context:',
      '"""',
      bestChunk || 'No relevant context found.',
      '"""',
      '',
      'User question:',
      question,
    ].join('\n');

    const completion = await openai.chat.completions.create({
      model: process.env.OPENAI_MODEL || 'gpt-4o',
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
      contextChunk: bestChunk,
    }).catch((err) => {
      console.error('Failed to log chat interaction:', err);
    });

    res.json({
      answer,
      contextPreview: bestChunk,
    });
  } catch (err) {
    console.error('Error handling /ask:', err);
    res.status(500).json({ error: 'Unexpected server error.' });
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

