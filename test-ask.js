
const question = process.argv[2] || 'What is this document about?';

fetch('http://localhost:3000/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question }),
})
  .then((res) => res.json())
  .then((data) => {
    if (data.error) {
      console.error('Error:', data.error);
      process.exit(1);
    }
    console.log('Question:', question);
    console.log('Answer:', data.answer);
    if (data.contextPreview) {
      console.log('\nContext used (preview):', data.contextPreview.slice(0, 200) + '...');
    }
  })
  .catch((err) => {
    console.error('Request failed. Is the server running on http://localhost:3000?', err.message);
    process.exit(1);
  });
