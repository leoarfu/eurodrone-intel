import { pipeline, cos_sim } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

let embedder, index = [];

async function init() {
  const res = await fetch('./embeddings/index.json');
  index = await res.json();
  embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
}

async function search(query) {
  const output = await embedder(query, { pooling: 'mean', normalize: true });
  const qVec = Array.from(output.data);
  const scored = index.map(item => ({
    ...item,
    score: cos_sim(qVec, item.vector)
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.filter(r => r.score > 0.35).slice(0, 5);
}

document.getElementById('query').addEventListener('input', async (e) => {
  const q = e.target.value.trim();
  const resultsDiv = document.getElementById('results');
  if (!q) { resultsDiv.innerHTML = ''; return; }

  const matches = await search(q);
  if (matches.length === 0) {
    resultsDiv.innerHTML = '<p>No matching entries found.</p>';
    return;
  }

  resultsDiv.innerHTML = '<p>Thinking...</p>';
  const res = await fetch('https://eurodrone-worker.didier08018.workers.dev', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ query: q, excerpts: matches }),
  });
  const { answer } = await res.json();

  resultsDiv.innerHTML = `<div class="answer">${answer}</div>` + matches.map(r => `
    <div class="result">
      <h3>${r.title}</h3>
      <div class="meta">${r.date} — <a href="${r.url}" target="_blank">${r.source}</a></div>
    </div>
  `).join('');
});

init();
