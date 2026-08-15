#!/usr/bin/env python3
"""
fetch_content.py - Eurodrone intel daily/weekly web sweep.

Usage:
    python3 fetch_content.py --mode daily
    python3 fetch_content.py --mode weekly

Requires: ANTHROPIC_API_KEY environment variable (set as a GitHub Actions
repo secret, injected into the workflow's env).
Requires: pip install anthropic
"""

import os
import re
import json
import argparse
import unicodedata
from datetime import datetime, timezone

import anthropic

KNOWLEDGE_DIR = "knowledge"
MANIFEST_FILE = "manifest.json"
MODEL = "claude-sonnet-5"

DAILY_PROMPT = """

You must use the web_search tool at least once before answering — do not
rely on your training data, which does not contain current news.

Search the web for Eurodrone (Airbus-led MALE RPAS program, partners
Dassault and Leonardo, program also referred to as "Male RPAS") news from
the last 24 hours. Cover commercial, technical, strategic, and financial
developments.

Do not include any of these URLs, they are already in the archive:
{existing_urls}

Return ONLY a JSON array (no other text, no markdown fences) of genuinely new
items. Each object must have exactly these fields:
  "title": headline, plain text
  "date": publication date, YYYY-MM-DD
  "source": outlet or organization name
  "url": the source URL
  "summary": 3-5 sentence factual summary of the item, plain text, no
             markdown, no inline citations (the url field is the citation)

If there is nothing genuinely new, return an empty JSON array: []
"""

WEEKLY_PROMPT = """

You must use the web_search tool at least once before answering — do not
rely on your training data, which does not contain current news.

Search the web for Eurodrone (Airbus-led MALE RPAS program, partners
Dassault and Leonardo) news and developments from the last 7 days. Cover
commercial, technical, strategic, and financial dimensions. In addition to
news outlets, check parliamentary and defense-procurement sources: German
Bundestag budget documents, French Assemblee Nationale budget documents,
European Defence Agency updates, and Airbus Defence and Space press
releases.

Do not include any of these URLs, they are already in the archive:
{existing_urls}

Return ONLY a JSON array (no other text, no markdown fences) of genuinely new
items not already covered. Each object must have exactly these fields:
  "title": headline, plain text
  "date": publication date, YYYY-MM-DD
  "source": outlet or organization name
  "url": the source URL
  "summary": 3-5 sentence factual summary of the item, plain text, no
             markdown, no inline citations (the url field is the citation)

If there is nothing genuinely new, return an empty JSON array: []
"""


def load_manifest():
    if not os.path.exists(MANIFEST_FILE):
        return []
    with open(MANIFEST_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(urls):
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(set(urls)), f, ensure_ascii=False, indent=2)


def slugify(text, max_len=60):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    return text[:max_len].strip("-") or "untitled"


def extract_json_array(text):
    """Model sometimes wraps output in prose or code fences despite
    instructions; pull out the first [...] block defensively."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in model output")
    return json.loads(match.group(0))


def write_entry(entry):
    date = entry.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = slugify(entry.get("title", "untitled"))
    filename = f"{date}-{slug}.md"
    path = os.path.join(KNOWLEDGE_DIR, filename)

    content = (
        f"# {entry['title']}\n"
        f"Date: {date}\n"
        f"Source: {entry.get('source', '')}\n"
        f"URL: {entry['url']}\n"
        f"---\n"
        f"{entry.get('summary', '').strip()}\n"
    )

    counter = 1
    base_path = path
    while os.path.exists(path):
        path = base_path.replace(".md", f"-{counter}.md")
        counter += 1

    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["daily", "weekly"], required=True)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY environment variable not set")
    
    manifest = load_manifest()
    prompt_template = DAILY_PROMPT if args.mode == "daily" else WEEKLY_PROMPT
    existing_urls = "\n".join(manifest) if manifest else "(none yet)"
    prompt = prompt_template.format(existing_urls=existing_urls)

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 15 if args.mode == "weekly" else 8,
        }],
    )

    search_calls = sum(1 for block in response.content if block.type == "server_tool_use")
    print(f"Web searches performed: {search_calls}")

    # Web-search responses interleave server_tool_use / tool_result blocks
    # with text blocks - concatenate only the text blocks.
    full_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    try:
        entries = extract_json_array(full_text)
    except ValueError as e:
        print(f"WARNING: could not parse model output as JSON: {e}")
        print("Raw output:")
        print(full_text)
        return

    new_urls = []
    written = 0
    for entry in entries:
        if not entry.get("url") or entry["url"] in manifest:
            continue
        try:
            path = write_entry(entry)
            print(f"Wrote {path}")
            new_urls.append(entry["url"])
            written += 1
        except KeyError as e:
            print(f"WARNING: skipping malformed entry, missing {e}: {entry}")

    if new_urls:
        save_manifest(manifest + new_urls)

    print(f"Done. {written} new entries, {len(entries) - written} skipped.")


if __name__ == "__main__":
    main()
