import os
import asyncio
import logging
import ssl
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
import google.generativeai as genai
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class GeminiTelegramBot:
    def __init__(self, telegram_token: str, gemini_api_key: str):
        """Initialize the bot with API keys"""
        self.telegram_token = telegram_token
        self.gemini_api_key = gemini_api_key
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        # List available models and log them
        try:
            available_models = genai.list_models()
            logger.info(f"Available Gemini models: {[m.name for m in available_models]}")
        except Exception as e:
            logger.error(f"Error listing Gemini models: {e}")
        # Use a valid model name here (update after checking logs)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create SSL context that doesn't verify certificates (for corporate networks)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Create custom HTTPX client with disabled SSL verification
        httpx_client = httpx.AsyncClient(
            verify=False,
            timeout=httpx.Timeout(30.0)
        )
        
        # Create custom request object
        request = HTTPXRequest(
            connection_pool_size=8,
            httpx_kwargs={
                "verify": False,
                "timeout": httpx.Timeout(30.0)
            }
        )
        
        # Initialize Telegram application with custom request
        self.application = Application.builder().token(self.telegram_token).request(request).build()
        
        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Store chat histories (in production, use a database)
        self.chat_histories = {}
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command"""
        user_name = update.effective_user.first_name
        welcome_message = f"""
🤖 Hello {user_name}! I'm your Gemini-powered chatbot!

I can help you with:
• Answering questions
• Creative writing
• Code assistance
• General conversation
• And much more!

Just send me a message and I'll respond using Google's Gemini AI.

Commands:
/help - Show this help message
/clear - Clear our conversation history
        """
        await update.message.reply_text(welcome_message)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        help_message = """
🔧 **Available Commands:**

/start - Start the bot and see welcome message
/help - Show this help message  
/clear - Clear conversation history

📝 **How to use:**
Just send me any message and I'll respond using Gemini AI!

⚡ **Tips:**
• I remember our conversation context
• Use /clear to start a fresh conversation
• I can help with coding, writing, questions, and more!
        """
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /clear command - clears chat history"""
        chat_id = update.effective_chat.id
        if chat_id in self.chat_histories:
            del self.chat_histories[chat_id]
        await update.message.reply_text("🗑️ Chat history cleared! Starting fresh conversation.")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle regular text messages"""
        try:
            chat_id = update.effective_chat.id
            user_message = update.message.text
            user_name = update.effective_user.first_name or "User"
            
            logger.info(f"User {user_name} ({chat_id}): {user_message}")
            
            # Send typing indicator
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            
            # Get or create chat history
            if chat_id not in self.chat_histories:
                self.chat_histories[chat_id] = []
            
            # Add user message to history
            self.chat_histories[chat_id].append({
                "role": "user",
                "parts": [user_message]
            })
            
            # Generate response using Gemini
            response = await self.generate_gemini_response(user_message, chat_id)
            
            if response:
                # Add bot response to history
                self.chat_histories[chat_id].append({
                    "role": "model",
                    "parts": [response]
                })
                
                # Send response to user
                await update.message.reply_text(response)
                logger.info(f"Bot response sent to {user_name}")
            else:
                await update.message.reply_text("😅 Sorry, I couldn't generate a response. Please try again!")
                
        except Exception as e:
            logger.error(f"Error in handle_message: {e}")
            await update.message.reply_text("🚨 Sorry, something went wrong. Please try again!")
    
    async def generate_gemini_response(self, user_message: str, chat_id: int) -> Optional[str]:
        """Generate response using Gemini API"""
        try:
            # Get chat history for context
            chat_history = self.chat_histories.get(chat_id, [])
            
            if chat_history:
                # Start chat with history for context
                chat = self.model.start_chat(history=chat_history[:-1])  # Exclude current message
                response = await asyncio.to_thread(chat.send_message, user_message)
            else:
                # First message, no history
                response = await asyncio.to_thread(self.model.generate_content, user_message)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None
    
    async def run(self) -> None:
        """Run the bot"""
        logger.info("Starting Gemini Telegram Bot...")
        await self.application.run_polling()

def main():
    # Load API keys from environment variables
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not TELEGRAM_BOT_TOKEN:
        print("❌ Please set your TELEGRAM_BOT_TOKEN in .env file or environment variables")
        return
    if not GEMINI_API_KEY:
        print("❌ Please set your GEMINI_API_KEY in .env file or environment variables")
        return
    # Create and run the bot
    bot = GeminiTelegramBot(TELEGRAM_BOT_TOKEN, GEMINI_API_KEY)
    bot.application.run_polling()

if __name__ == "__main__":
    main() 