"""
rag.py - the retrieval-augmented agent.

Shared by bot.py (Telegram) and evaluate.py (eval harness).
Exposes one main function: answer(question, history) -> (text, sources)
"""

import os
import threading

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

DB_DIR = os.getenv("CHROMA_DIR", "data/chroma")
COLLECTION = os.getenv("CHROMA_COLLECTION", "trk")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_INPUT = 2000          # cap user input length
HISTORY_TURNS = 6         # how many past messages to feed the model

# --- lazy singletons (thread-safe, reused across concurrent users) ---
_client = None
_col = None
_lock = threading.Lock()


def get_client() -> OpenAI:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = OpenAI(
                    api_key=os.environ["OPENAI_API_KEY"],
                    timeout=30,
                    max_retries=2,
                )
    return _client


def get_collection():
    global _col
    if _col is None:
        with _lock:
            if _col is None:
                ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.environ["OPENAI_API_KEY"], model_name=EMBED_MODEL
                )
                db = chromadb.PersistentClient(path=DB_DIR)
                _col = db.get_or_create_collection(
                    name=COLLECTION, embedding_function=ef
                )
    return _col


SYSTEM = """You are the customer-support assistant for the website trk.uz.
Answer using ONLY the provided context taken from the website.
Rules:
- Reply in the SAME language as the user's question.
- If the answer is not in the context, say you don't have that information yet and
  suggest contacting support. Never invent facts, dates, prices, or contacts.
- Be concise, friendly, and helpful.
"""

REWRITE_SYSTEM = """Rewrite the user's latest message into ONE standalone search query,
in the same language, resolving any pronouns or references using the conversation.
Return only the query text, nothing else."""


def rewrite_query(question: str, history: list) -> str:
    """Turn a follow-up like 'when is its deadline?' into a standalone query."""
    if not history:
        return question
    try:
        msgs = [{"role": "system", "content": REWRITE_SYSTEM}]
        msgs += history[-HISTORY_TURNS:]
        msgs.append({"role": "user", "content": question})
        r = get_client().chat.completions.create(
            model=LLM_MODEL, messages=msgs, temperature=0, max_tokens=80
        )
        return (r.choices[0].message.content or "").strip() or question
    except Exception:
        return question  # fall back to the raw question on any failure


def retrieve(query: str, k: int = TOP_K):
    try:
        res = get_collection().query(query_texts=[query], n_results=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return list(zip(docs, metas))
    except Exception:
        return []


def _build_context(chunks):
    parts = []
    for i, (doc, meta) in enumerate(chunks, 1):
        parts.append(f"[Source {i}] {meta.get('title','')} ({meta.get('url','')})\n{doc}")
    return "\n\n".join(parts)


def answer(question: str, history: list = None):
    """Return (answer_text, list_of_source_urls). Never raises."""
    history = history or []
    question = (question or "").strip()[:MAX_INPUT]
    if not question:
        return "Please type a question.", []

    search_query = rewrite_query(question, history)
    chunks = retrieve(search_query)
    if not chunks:
        return ("I don't have information about that yet. "
                "Please contact trk.uz support directly."), []

    context = _build_context(chunks)
    msgs = [{"role": "system", "content": SYSTEM}]
    msgs += history[-HISTORY_TURNS:]
    msgs.append({"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {question}"})

    try:
        r = get_client().chat.completions.create(
            model=LLM_MODEL, messages=msgs, temperature=0.2, max_tokens=500
        )
        text = (r.choices[0].message.content or "").strip()
    except Exception:
        return ("Sorry, I'm having trouble responding right now. "
                "Please try again in a moment."), []

    sources, seen = [], set()
    for _, meta in chunks:
        u = meta.get("url")
        if u and u not in seen:
            seen.add(u)
            sources.append(u)
    return text, sources
