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
}

async function logInteraction({ documentId, sessionId, question, answer, contextChunk }) {
  try {
    await ensureSchema();
    await db('chat_history').insert({
      document_id: documentId || null,
      session_id: sessionId || null,
      question,
      answer,
      context_chunk: contextChunk,
    });
  } catch (err) {
    console.error('Failed to log interaction:', err);
  }
}

module.exports = {
  db,
  logInteraction,
};

