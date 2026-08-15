import { pipeline, cos_sim } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

let embedder, index = [];

async function init() {
  const res = await fetch('../embeddings/index.json');
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
  return scored.slice(0, 8);
}

document.getElementById('query').addEventListener('input', async (e) => {
  const q = e.target.value.trim();
  const resultsDiv = document.getElementById('results');
  if (!q) { resultsDiv.innerHTML = ''; return; }
  const results = await search(q);
  resultsDiv.innerHTML = results.map(r => `
    <div class="result">
      <h3>${r.title}</h3>
      <div class="meta">${r.date} — <a href="${r.url}" target="_blank">${r.source}</a></div>
      <p>${r.excerpt}</p>
    </div>
  `).join('');
});

init();
