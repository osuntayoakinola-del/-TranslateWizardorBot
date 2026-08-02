"""
TranslateWizard Telegram Bot
An AI-powered language translation assistant for Telegram, backed by the Anthropic API.

Run locally:
    export TELEGRAM_BOT_TOKEN="..."
    export ANTHROPIC_API_KEY="..."
    python bot.py

Deployed on Railway, these two env vars are set in the project's Variables tab.
"""

import logging
import os
from collections import defaultdict

from anthropic import Anthropic
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# Swap models freely: e.g. "claude-haiku-4-5-20251001" for a cheaper/faster bot.
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

DEFAULT_TARGET_LANGUAGE = "English"

TELEGRAM_MAX_LEN = 4000  # stay under Telegram's 4096-char hard limit

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("translatewizard-bot")

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Per-chat target language, e.g. {chat_id: "Spanish"}. Resets on redeploy —
# see README for how to make this persistent with a database.
target_languages: dict[int, str] = defaultdict(lambda: DEFAULT_TARGET_LANGUAGE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_for_telegram(text: str, limit: int = TELEGRAM_MAX_LEN) -> list[str]:
    """Split long text into Telegram-safe chunks, breaking on newlines where possible."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:]
    return chunks


def build_system_prompt(target_language: str) -> str:
    return (
        f"You are TranslateWizard, a translation assistant inside a Telegram "
        f"bot. Detect the language of the user's message and translate it "
        f"into {target_language}. If the message is already in {target_language}, "
        f"translate it into English instead so the user still gets a useful reply. "
        f"Reply with ONLY the translation — no explanations, no notes, no quotes "
        f"around it, unless the user explicitly asks you to explain a phrase or "
        f"nuance. Preserve tone, formatting, and line breaks from the original."
    )


async def translate_text(chat_id: int, user_text: str) -> str:
    """Send the user's message to Claude for translation and return the result."""
    target_language = target_languages[chat_id]
    system_prompt = build_system_prompt(target_language)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_text}],
        )
    except Exception as e:
        logger.exception("Error calling Anthropic API")
        return f"⚠️ Sorry, I hit an error talking to the AI backend:\n`{e}`"

    reply_text = "".join(block.text for block in response.content if block.type == "text")
    if not reply_text:
        return "⚠️ The AI backend returned an empty response — try rephrasing."
    return reply_text


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    current_lang = target_languages[chat_id]
    await update.message.reply_text(
        "🌐 Hi, I'm *TranslateWizard* — your AI translation assistant.\n\n"
        "Send me any message and I'll translate it. Auto-detects the source "
        f"language. Current target language: *{current_lang}*.\n\n"
        "Commands:\n"
        "/setlang <language> — set your target language (e.g. /setlang Spanish)\n"
        "/lang — show your current target language\n"
        "/help — show this message",
        parse_mode=ParseMode.MARKDOWN,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start(update, context)


async def setlang(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text(
            "Tell me which language, e.g. `/setlang French`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    language = " ".join(context.args).strip()
    target_languages[chat_id] = language
    await update.message.reply_text(f"✅ Target language set to *{language}*.", parse_mode=ParseMode.MARKDOWN)


async def show_lang(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    await update.message.reply_text(f"Current target language: *{target_languages[chat_id]}*", parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    reply_text = await translate_text(chat_id, user_text)

    for chunk in split_for_telegram(reply_text):
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(chunk)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception", exc_info=context.error)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("setlang", setlang))
    app.add_handler(CommandHandler("lang", show_lang))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("TranslateWizard bot starting (model=%s)...", MODEL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
