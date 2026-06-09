# trk.uz Support Bot

A customer-support assistant for **trk.uz**. It scrapes the website, builds a
vector knowledge base (Chroma), answers questions with a RAG agent (OpenAI),
and serves users through a **Telegram bot**. Telegram chat *is* the interface —
no web UI.

```
scraper.py    crawl trk.uz  -> data/pages.json
ingest.py     chunk + embed -> Chroma vector DB
rag.py        retrieval + answer (the agent core)
bot.py        Telegram bot (uses rag.py)
evaluate.py   eval harness (uses rag.py) -> numbers
eval/questions.json   the test set
```

## 1. Setup

Requires **Python 3.10+**.

```bash
git clone <your-repo-url>
cd trk-support-bot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium     # browser for the scraper
```

Create your env file and fill in the two keys:

```bash
cp .env.example .env
# edit .env:
#   OPENAI_API_KEY      -> from platform.openai.com
#   TELEGRAM_BOT_TOKEN  -> from @BotFather on Telegram
```

## 2. Build the knowledge base

```bash
python scraper.py     # writes data/pages.json
python ingest.py      # builds Chroma DB at data/chroma
```

## 3. Run the bot

```bash
python bot.py
```

Open your bot in Telegram, send `/start`, and ask questions.
Use `/reset` to clear the conversation. Follow-up questions keep context, e.g.:

```
You: What services does trk.uz offer?
Bot: ...
You: How do I contact them about it?   <- "them / it" resolved from context
```

## 4. Run the evaluation

```bash
python evaluate.py
```

Prints answer correctness, out-of-scope refusal rate, and latency.
Edit `eval/questions.json` to match the content you actually scraped.

## Notes / assumptions

- Default models: `gpt-4o-mini` (chat) and `text-embedding-3-small` (embeddings).
  Both are cheap and multilingual (Uzbek / Russian / English). Override in `.env`.
- The scraper caps at 200 pages / depth 4 (`scraper.py` constants) so a first run
  stays fast. Raise them for fuller coverage.
- Re-running `ingest.py` rebuilds the collection from scratch.
- See `DESIGN.md` for the design decisions and trade-offs.
```
