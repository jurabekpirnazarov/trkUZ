"""
ingest.py - load data/pages.json, chunk it, embed it, store in Chroma.

Run (after scraper.py):
    python ingest.py
"""

import os
import re
import json

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions

PAGES = os.getenv("PAGES_FILE", "data/pages.json")
DB_DIR = os.getenv("CHROMA_DIR", "data/chroma")
COLLECTION = os.getenv("CHROMA_COLLECTION", "trk")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    chunks, i = [], 0
    step = max(size - overlap, 1)
    while i < len(text):
        chunks.append(text[i:i + size])
        i += step
    return chunks


def main():
    if not os.path.exists(PAGES):
        print(f"{PAGES} not found. Run scraper.py first.")
        return

    with open(PAGES, encoding="utf-8") as f:
        pages = json.load(f)

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name=EMBED_MODEL
    )
    db = chromadb.PersistentClient(path=DB_DIR)

    # Rebuild the collection from scratch each run.
    try:
        db.delete_collection(COLLECTION)
    except Exception:
        pass
    col = db.create_collection(name=COLLECTION, embedding_function=ef)

    ids, docs, metas = [], [], []
    for p in pages:
        title = p.get("title", "")
        url = p.get("url", "")
        content = p.get("content") or p.get("text") or ""
        for j, ch in enumerate(chunk_text(content)):
            body = f"{title}\n{ch}" if title else ch
            ids.append(f"{url}#{j}")
            docs.append(body)
            metas.append({"url": url, "title": title})

    if not docs:
        print("No content to ingest.")
        return

    # Add in batches to stay under embedding API limits.
    batch = 100
    for i in range(0, len(docs), batch):
        col.add(
            ids=ids[i:i + batch],
            documents=docs[i:i + batch],
            metadatas=metas[i:i + batch],
        )
        print(f"ingested {min(i + batch, len(docs))}/{len(docs)} chunks")

    print(f"Done. {len(docs)} chunks in collection '{COLLECTION}' at {DB_DIR}")


if __name__ == "__main__":
    main()
