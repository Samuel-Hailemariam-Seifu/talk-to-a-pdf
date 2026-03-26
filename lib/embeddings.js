const OpenAI = require('openai');

const openai = process.env.OPENAI_API_KEY
  ? new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    })
  : null;

const EMBEDDING_MODEL = process.env.OPENAI_EMBEDDING_MODEL || 'text-embedding-3-small';
const FALLBACK_DIM = 256;

/**
 * Lightweight deterministic embedding fallback (no external API key required).
 * This keeps retrieval working when OPENAI_API_KEY is not configured.
 * @param {string} text
 * @returns {number[]}
 */
function getFallbackEmbedding(text) {
  const vec = new Array(FALLBACK_DIM).fill(0);
  const tokens = text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);

  for (const token of tokens) {
    let hash = 0;
    for (let i = 0; i < token.length; i++) {
      hash = (hash * 31 + token.charCodeAt(i)) >>> 0;
    }
    vec[hash % FALLBACK_DIM] += 1;
  }

  // L2 normalize so cosine similarity remains meaningful.
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  }

  return vec;
}

/**
 * Get embedding vector for a single text using OpenAI.
 * @param {string} text
 * @returns {Promise<number[]>}
 */
async function getEmbedding(text) {
  if (!text || !text.trim()) return openai ? new Array(1536).fill(0) : new Array(FALLBACK_DIM).fill(0);
  if (!openai) return getFallbackEmbedding(text);
  const res = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: text.slice(0, 8000),
    encoding_format: 'float',
  });
  return res.data?.[0]?.embedding ?? [];
}

/**
 * Cosine similarity between two vectors.
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number}
 */
function cosineSimilarity(a, b) {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

module.exports = {
  getEmbedding,
  cosineSimilarity,
};
