const knex = require('knex');

const db = knex({
  client: 'sqlite3',
  connection: {
    filename: process.env.DB_FILE || './chat_history.db',
  },
  useNullAsDefault: true,
});

async function ensureSchema() {
  const exists = await db.schema.hasTable('chat_history');
  if (!exists) {
    await db.schema.createTable('chat_history', (table) => {
      table.increments('id').primary();
      table.timestamp('created_at').defaultTo(db.fn.now());
      table.text('document_id');
      table.text('session_id');
      table.text('question').notNullable();
      table.text('answer').notNullable();
      table.text('context_chunk');
      table.text('context_sources');
    });
    return;
  }

  const hasDocumentId = await db.schema.hasColumn('chat_history', 'document_id');
  if (!hasDocumentId) {
    await db.schema.alterTable('chat_history', (table) => {
      table.text('document_id');
    });
  }

  const hasSessionId = await db.schema.hasColumn('chat_history', 'session_id');
  if (!hasSessionId) {
    await db.schema.alterTable('chat_history', (table) => {
      table.text('session_id');
    });
  }

  const hasContextSources = await db.schema.hasColumn('chat_history', 'context_sources');
  if (!hasContextSources) {
    await db.schema.alterTable('chat_history', (table) => {
      table.text('context_sources');
    });
  }
}

async function logInteraction({ documentId, sessionId, question, answer, contextChunk, sources }) {
  try {
    await ensureSchema();
    await db('chat_history').insert({
      document_id: documentId || null,
      session_id: sessionId || null,
      question,
      answer,
      context_chunk: contextChunk,
      context_sources: sources ? JSON.stringify(sources) : null,
    });
  } catch (err) {
    console.error('Failed to log interaction:', err);
  }
}

async function getHistory({
  limit = 50,
  offset = 0,
  since,
  questionContains,
  documentId,
  sessionId,
}) {
  await ensureSchema();

  let query = db('chat_history')
    .select(
      'id',
      'created_at',
      'document_id',
      'session_id',
      'question',
      'answer',
      'context_chunk',
      'context_sources'
    )
    .orderBy('id', 'desc')
    .limit(limit)
    .offset(offset);

  if (since) query = query.where('created_at', '>=', since);
  if (questionContains) query = query.where('question', 'like', `%${questionContains}%`);
  if (documentId) query = query.where('document_id', documentId);
  if (sessionId) query = query.where('session_id', sessionId);

  const rows = await query;
  return rows.map((row) => {
    let parsedSources = null;
    if (row.context_sources) {
      try {
        parsedSources = JSON.parse(row.context_sources);
      } catch (err) {
        parsedSources = null;
      }
    }
    return {
      id: row.id,
      created_at: row.created_at,
      document_id: row.document_id,
      session_id: row.session_id,
      question: row.question,
      answer: row.answer,
      context_chunk: row.context_chunk,
      context_sources: parsedSources,
    };
  });
}

module.exports = {
  db,
  ensureSchema,
  logInteraction,
  getHistory,
};

