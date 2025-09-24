import logging
import google.generativeai as genai
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import ssl
import certifi
import httpx

# Force httpx (used by python-telegram-bot) to use certifi's CA bundle
def _patched_ssl_context():
    return ssl.create_default_context(cafile=certifi.where())

httpx._default_ssl_context = _patched_ssl_context


# 🔑 Put your keys here
TELEGRAM_TOKEN = "8202724613:AAH-mxzhyBkuv2D6sG49W-OUSHV9ga1C9WQ"
GEMINI_API_KEY = "AIzaSyCveUj7czWR6CLHgzNGLiE6FZ4jMjFreYM"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hey 👋 I'm your Gemini-powered friend! Let's chat.")

# Handle messages
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    logger.info(f"User said: {user_input}")

    try:
        response = model.generate_content(user_input)
        reply = response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        reply = "Sorry, something went wrong 😔"

    await update.message.reply_text(reply)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    print("🤖 Bot is running... Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    main()
