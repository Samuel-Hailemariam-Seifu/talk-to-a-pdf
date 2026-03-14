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
      table.text('question').notNullable();
      table.text('answer').notNullable();
      table.text('context_chunk');
    });
  }
}

async function logInteraction({ question, answer, contextChunk }) {
  try {
    await ensureSchema();
    await db('chat_history').insert({
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

