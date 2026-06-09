"""
bot.py - Telegram bot connected to the RAG agent.

Run (after ingest.py):
    python bot.py

Features:
- Multi-turn memory per chat (follow-up questions keep context).
- Handles concurrent users, empty/very long messages, and slow/down APIs
  without crashing or hanging.
"""

import os
import asyncio
import logging
from collections import defaultdict, deque

from dotenv import load_dotenv
load_dotenv()

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)

import rag

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("bot")

MAX_HISTORY = 12  # messages kept per chat (user + assistant combined)
histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY))


async def start(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Assalomu alaykum! Men trk.uz qo'llab-quvvatlash yordamchisiman. "
        "Sayt haqida istalgan savolingizni bering.\n"
        "Suhbatni tozalash uchun /reset buyrug'ini yuboring."
    )


async def reset(update: Update, _: ContextTypes.DEFAULT_TYPE):
    histories[update.effective_chat.id].clear()
    await update.message.reply_text("Suhbat tozalandi.")


async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()

    if not text:
        await update.message.reply_text("Iltimos, matnli savol yuboring.")
        return
    if len(text) > 2000:
        text = text[:2000]

    history = list(histories[chat_id])
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        # Run the blocking OpenAI/Chroma work in a thread so one slow request
        # never blocks other users.
        reply, sources = await asyncio.to_thread(rag.answer, text, history)
    except Exception:
        log.exception("answer failed")
        await update.message.reply_text(
            "Kechirasiz, xatolik yuz berdi. Iltimos, birozdan so'ng qayta urinib ko'ring."
        )
        return

    histories[chat_id].append({"role": "user", "content": text})
    histories[chat_id].append({"role": "assistant", "content": reply})

    out = reply
    if sources:
        out += "\n\nManbalar:\n" + "\n".join(sources[:3])
    await update.message.reply_text(out[:4000])  # Telegram limit is ~4096


def main():
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = (
        Application.builder()
        .token(token)
        .concurrent_updates(True)   # serve multiple users in parallel
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    log.info("Bot started. Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
