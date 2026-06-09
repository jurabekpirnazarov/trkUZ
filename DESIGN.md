# DESIGN.md

## Goal

A production-shaped (not demo) customer-support assistant for trk.uz, delivered
over Telegram, backed by a knowledge base built from the site's own content.

## Pipeline overview

```
trk.uz --scraper.py--> data/pages.json --ingest.py--> Chroma --rag.py--> bot.py / evaluate.py
```

Four small, single-responsibility modules. `rag.py` holds the agent logic and is
imported by both the bot and the eval harness, so the thing we test is exactly
the thing users talk to.

## Content collection

Many pages on trk.uz are JS-rendered, so the scraper uses **Playwright**
(headless Chromium) rather than plain `requests`. For each page it:

- waits for `networkidle` + a short buffer so late content loads,
- extracts the main article text with **trafilatura** (strips nav/footer/ads),
- skips near-empty pages (`< 80` chars) and de-duplicates by a SHA-256 of the
  text, which kills near-identical template pages,
- follows only internal links, bounded by `MAX_PAGES` and `MAX_DEPTH`.

Output is clean JSON: `{url, title, content}` per page. JSON keeps the URL as a
citation source and decouples scraping from indexing (re-index without re-crawl).

## Chunking & retrieval

- **Chunking:** ~800-character chunks with 150-char overlap, with the page title
  prepended to each chunk. Overlap avoids cutting an answer across a boundary;
  the title gives short chunks topical anchoring (helps multilingual retrieval).
- **Retrieval:** dense vector search in Chroma, top-k = 5.
- **Store:** Chroma `PersistentClient` — zero infra, on-disk, trivial to run
  locally; exactly what the task wants for a self-hostable deliverable.

**Rejected alternatives:**
- *Sentence/semantic chunking* — better boundaries but more complexity; fixed-size
  overlap is good enough for a support FAQ corpus and far simpler.
- *Hybrid BM25 + dense (reranking)* — would raise recall on rare keywords/proper
  nouns; deferred as the highest-value next step (see "Given more time").
- *Postgres/pgvector or a hosted vector DB* — overkill for this scale; adds infra.

## Models & trade-offs

| Component  | Choice                    | Why |
|-----------|----------------------------|-----|
| Embeddings | `text-embedding-3-small`  | cheap, strong multilingual (UZ/RU/EN) |
| LLM        | `gpt-4o-mini`             | low latency + cost, good instruction-following |

Trade-off: both are hosted (network dependency, per-call cost) but make the repo
runnable with a single API key and no GPU. Everything is swappable via `.env`
(e.g. point `EMBED_MODEL` at a local model). gpt-4o-mini keeps median latency at
a few seconds; upgrading to a larger model improves hard questions at higher
cost/latency.

## Conversation handling (multi-turn)

Per-chat history is kept in memory (a bounded `deque`). For follow-ups, a small
**query-rewrite** step turns "when is its deadline?" into a standalone search
query using recent history *before* retrieval — otherwise pronouns retrieve
nothing. The recent history is also passed to the answering model for tone/context.

## Reliability

- **Concurrency:** the bot enables `concurrent_updates`, and every blocking
  OpenAI/Chroma call runs via `asyncio.to_thread`, so one slow request never
  freezes other users.
- **Slow / down API:** OpenAI client has a 30s timeout + 2 retries; all model
  and DB calls are wrapped so failures degrade to a friendly message instead of
  crashing. `rag.answer()` never raises.
- **Bad input:** empty messages are rejected politely; very long messages are
  truncated (2000 chars in, 4000 out for Telegram's limit).
- **Grounding:** the system prompt forbids inventing facts and instructs the bot
  to decline + redirect to support when the context lacks the answer.

## Evaluation

`evaluate.py` runs `eval/questions.json` and reports:

- **Answer correctness** — an LLM judge checks each answer is on-topic and
  grounded in the retrieved sources (catches hallucination). Chosen over exact
  string match because support answers are paraphrased, not verbatim.
- **Out-of-scope refusal rate** — unrelated questions (weather, sports, "write a
  poem") must be declined; measures that the bot stays in its lane.
- **Latency** (avg / p95) — the user-facing cost of model choices.

The set mixes UZ / RU / EN to verify multilingual behavior.

## Key weaknesses & risks

- **Retrieval is dense-only:** rare proper nouns or exact codes can be missed.
  Mitigation planned: hybrid search + reranking.
- **Coverage = crawl quality:** anything not scraped can't be answered. Mitigated
  by honest "I don't know + contact support" rather than guessing.
- **In-memory history:** lost on restart and not shared across processes. Fine for
  a single instance; a real deployment would back it with Redis.
- **Hosted-LLM dependency:** cost and an external point of failure; handled with
  timeouts, retries, and graceful fallbacks.

## Given more time

1. Hybrid retrieval (BM25 + dense) with a cross-encoder reranker — biggest quality
   win for a real FAQ corpus.
2. Incremental re-crawl (only changed pages) on a schedule.
3. Persistent conversation store (Redis) + simple rate limiting per user.
4. A larger, human-labeled eval set with citation-accuracy and faithfulness scores.
5. Streaming responses and a "was this helpful?" feedback loop for live tuning.
