import os
import logging
from typing import Dict
from dotenv import load_dotenv

import google.generativeai as genai
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Load environment variables
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION = (
    "Eres Juan, un simpático y divertido profesor de español de España. "
    "Tu objetivo es ayudar a Megan a mejorar su español (Castellano). "
    "Sigue estas reglas estrictamente:\n"
    "1. Corrige suavemente cualquier error gramatical o de vocabulario que cometa Megan antes de responder.\n"
    "2. Propón nuevos temas de conversación de forma natural si la charla se detiene.\n"
    "3. IMPORTANTE: En cada mensaje, primero escribe tu respuesta en ESPAÑOL. Justo debajo, escribe la TRADUCCIÓN al INGLÉS en cursiva.\n"
    "4. Usa Markdown para que la traducción al inglés esté en cursiva (ej: *Hello, how are you?*).\n"
    "5. Usa un tono cercano, motivador y divertido, típico de un profesor de España que tiene mucha confianza con su alumna."
)

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=SYSTEM_INSTRUCTION,
)

import time

# dictionary to store chat sessions per user
chat_sessions: Dict[int, any] = {}
last_interaction: Dict[int, float] = {}
SESSION_TIMEOUT = 1800  # 30 minutes in seconds

def get_chat_session(user_id: int):
    """Retrieve or create a chat session for a user, checking for timeout."""
    current_time = time.time()
    
    # Check for inactivity timeout
    if user_id in last_interaction:
        if current_time - last_interaction[user_id] > SESSION_TIMEOUT:
            logger.info(f"Session for user {user_id} timed out. Resetting.")
            chat_sessions[user_id] = model.start_chat(history=[])
    
    if user_id not in chat_sessions:
        chat_sessions[user_id] = model.start_chat(history=[])
    
    last_interaction[user_id] = current_time
    return chat_sessions[user_id]

async def send_welcome(update: Update):
    """Send the welcome message and clear session."""
    user_id = update.effective_user.id
    chat_sessions[user_id] = model.start_chat(history=[])
    last_interaction[user_id] = time.time()
    
    keyboard = [[KeyboardButton("🔄 Empezar nueva conversación")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    welcome_msg = (
        "¡Hola Megan! Soy Juan, tu tutor personal de español. 🇪🇸\n\n"
        "Estoy aquí para ayudarte a practicar y que hablemos de lo que tú quieras. "
        "No te preocupes por los errores, ¡así es como se aprende! ¿Qué tal estás hoy?\n\n"
        "*Hello Megan! I'm Juan, your personal Spanish tutor. I'm here to help you practice and we can talk about whatever you want. Don't worry about mistakes, that's how you learn! How are you today?*"
    )
    await update.message.reply_text(welcome_msg, reply_markup=reply_markup, parse_mode="Markdown")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command."""
    await send_welcome(update)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages."""
    user_id = update.effective_user.id
    user_text = update.message.text
    current_time = time.time()

    # Check for the custom menu button
    if user_text == "🔄 Empezar nueva conversación":
        await send_welcome(update)
        return

    # Check for timeout before processing the message to maybe send a new welcome
    is_new_session = False
    if user_id in last_interaction:
        if current_time - last_interaction[user_id] > SESSION_TIMEOUT:
            is_new_session = True

    chat = get_chat_session(user_id)
    
    try:
        # Send message to Gemini
        response = chat.send_message(user_text)
        
        # If it's a new session due to timeout, we could optionally prepend a "Welcome back" 
        # but the user wanted it to "bore todo", so a fresh response is usually enough.
        await update.message.reply_text(response.text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        error_msg = (
            "Lo siento, he tenido un pequeño problema técnico. 😅 "
            "A veces la conexión se satura un poco. ¿Podrías intentar enviarlo de nuevo?\n\n"
            "*Sorry, I had a little technical problem. Sometimes the connection gets a bit saturated. Could you try sending it again?*"
        )
        if "rate limit" in str(e).lower():
            error_msg = (
                "Vaya, parece que hemos hablado demasiado rápido. Espera unos segundos y vuelve a intentarlo. ⏳\n\n"
                "*Oops, looks like we talked too fast. Wait a few seconds and try again.*"
            )
        
        await update.message.reply_text(error_msg, parse_mode="Markdown")

if __name__ == "__main__":
    if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
        logger.error("Missing TELEGRAM_TOKEN or GEMINI_API_KEY environment variables.")
        exit(1)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    logger.info("Bot is starting...")
    app.run_polling()
