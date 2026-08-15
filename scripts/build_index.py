import os, json, re

KNOWLEDGE_DIR = "knowledge"
OUTPUT_FILE = "embeddings/index.json"

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def parse_md(filepath):
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    title_match = re.search(r"^#\s*(.+)$", content, re.MULTILINE)
    date_match = re.search(r"Date:\s*(.+)", content)
    source_match = re.search(r"Source:\s*(.+)", content)
    url_match = re.search(r"URL:\s*(.+)", content)
    body = content.split("---", 1)[-1].strip() if "---" in content else content
    return {
        "title": title_match.group(1).strip() if title_match else os.path.basename(filepath),
        "date": date_match.group(1).strip() if date_match else "",
        "source": source_match.group(1).strip() if source_match else "",
        "url": url_match.group(1).strip() if url_match else "",
        "excerpt": body[:400],
        "text": body
    }

entries = []
for fname in os.listdir(KNOWLEDGE_DIR):
    if fname.endswith(".md"):
        entries.append(parse_md(os.path.join(KNOWLEDGE_DIR, fname)))

if entries:
    vectors = model.encode([e["text"] for e in entries], normalize_embeddings=True)
    for e, v in zip(entries, vectors):
        e["vector"] = v.tolist()
        del e["text"]

os.makedirs("embeddings", exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

print(f"Indexed {len(entries)} entries.")
