import os
import re
import logging
import time
import io
import json
from datetime import time as dt_time
import pytz
from typing import Dict, List, Set
from dotenv import load_dotenv

from openai import OpenAI
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TIMEZONE = os.getenv("TIMEZONE", "Europe/Madrid")

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize OpenAI Client
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set! Check your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

# Persistence for users
DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    # Ensure users.json exists
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump([], f)

def load_users() -> Set[int]:
    ensure_data_dir()
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            try:
                return set(json.load(f))
            except json.JSONDecodeError:
                return set()
    return set()

def save_user(user_id: int):
    users = load_users()
    if user_id not in users:
        users.add(user_id)
        with open(USERS_FILE, "w") as f:
            json.dump(list(users), f)

def strip_emojis(text: str) -> str:
    """Remove emojis for cleaner TTS."""
    return re.sub(r'[^\w\s,.!?¿¡áéíóúÁÉÍÓÚñÑ]', '', text)

SYSTEM_PROMPT = (
    "Eres Juanito, un niño de 10 años de MADRID, ESPAÑA. Hablas con un marcado acento CASTELLANO. "
    "Tu misión es ayudar a tu amiga Megan a perfeccionar su español de España. "
    "Sigue estas REGLAS DE ORO O TE LLEVARÁS UN TIRÓN DE OREJAS:\n"
    "1. VOCABULARIO: Usa palabras de España: 'vale', 'guay', 'mola', 'tío', 'vosotros'. NUNCA uses 'ustedes'.\n"
    "2. CORRECCIONES: Corrige CUALQUIER error de Megan. Sé estricto.\n"
    "3. SÉ PROACTIVO: Termina SIEMPRE con una PREGUNTA divertida.\n"
    "4. FORMATO OBLIGATORIO (NO TE SALTES NADA):\n"
    "   [Tu respuesta en español madrileño terminando en PREGUNTA]\n"
    "   ```\n"
    "   [Full English translation of EVERYTHING above, including corrections and the question]\n"
    "   ```\n"
    "   --- \n"
    "   💡 Ideas para responder:\n"
    "   - [Idea 1 en español] ([English translation 1])\n"
    "   - [Idea 2 en español] ([English translation 2])\n"
    "5. LAS IDEAS PARA RESPONDER SON OBLIGATORIAS en cada mensaje."
)

# Global stores
chat_histories: Dict[int, List[dict]] = {}
last_interaction: Dict[int, float] = {}
SESSION_TIMEOUT = 1800  # 30 minutes in seconds

def get_chat_history(user_id: int) -> List[dict]:
    """Retrieve or create a chat history for a user, checking for timeout."""
    current_time = time.time()
    
    # Check for inactivity timeout
    if user_id in last_interaction:
        if current_time - last_interaction[user_id] > SESSION_TIMEOUT:
            logger.info(f"Session for user {user_id} timed out. Resetting.")
            chat_histories[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if user_id not in chat_histories:
        chat_histories[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    last_interaction[user_id] = current_time
    return chat_histories[user_id]

async def process_interaction(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str):
    """Core logic to handle both text and voice transcriptions."""
    user_id = update.effective_user.id
    save_user(user_id) # Ensure user is in our notification list

    history = get_chat_history(user_id)
    history.append({"role": "user", "content": user_text})
    
    try:
        # Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            temperature=0.8
        )
        
        response_text = response.choices[0].message.content
        history.append({"role": "assistant", "content": response_text})

        # --- Send text first ---
        await update.message.reply_text(response_text, parse_mode="Markdown")

        # --- Generate and send Audio (OpenAI TTS) ---
        first_line = response_text.split('\n')[0].strip()
        spanish_for_tts = strip_emojis(first_line)

        if spanish_for_tts:
            try:
                tts_response = client.audio.speech.create(
                    model="tts-1",
                    voice="echo",
                    input=spanish_for_tts,
                    speed=1.1
                )
                audio_file = io.BytesIO(tts_response.content)
                audio_file.name = "voice.mp3"
                await update.message.reply_voice(voice=audio_file)
            except Exception as tts_e:
                logger.error(f"TTS Error: {tts_e}")
        
    except Exception as e:
        logger.error(f"OpenAI Error: {e}")
        error_msg = "¡Uy! ¡Se me ha roto el juguete! 😅 ¿Puedes decírmelo otra vez?"
        await update.message.reply_text(error_msg)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages."""
    user_text = update.message.text
    if user_text == "🔄 Empezar nueva conversación":
        await start(update, context)
        return
    await process_interaction(update, context, user_text)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming voice messages using Whisper."""
    user_id = update.effective_user.id
    logger.info(f"Received voice message from {user_id}")
    
    try:
        # Download the voice file
        voice_file = await update.message.voice.get_file()
        voice_data = await voice_file.download_as_bytearray()
        
        # Use Whisper to transcribe
        # Whisper requires a file-like object with a proper name for format detection
        audio_buffer = io.BytesIO(voice_data)
        audio_buffer.name = "voice.ogg" # Telegram voice is usually OGG/Opus
        
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_buffer
        )
        
        transcribed_text = transcription.text
        logger.info(f"Transcribed voice: {transcribed_text}")
        
        # Send a small confirmation text
        await update.message.reply_text(f"🎤 _Juanito te ha escuchado:_ \"{transcribed_text}\"", parse_mode="Markdown")
        
        # Process as a normal message
        await process_interaction(update, context, transcribed_text)
        
    except Exception as e:
        logger.error(f"Whisper Error: {e}")
        await update.message.reply_text("¡Uy! No te he oído bien, ¿puedes repetirlo? 👂")

async def daily_word_job(context: ContextTypes.DEFAULT_TYPE):
    """Send a Word of the Day to all users."""
    users = load_users()
    logger.info(f"Running daily job for {len(users)} users")
    
    for user_id in users:
        try:
            # Generate a "Word of the Day" using GPT-4o-mini
            prompt = "Genera una 'Palabra del día' en español de España para un estudiante. Incluye la palabra, su significado sencillo, un ejemplo de uso madrileño y una pregunta corta para el estudiante. Responde en el formato de Juanito (niño de 10 años)."
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            )
            msg = f"🌅 ¡Buenos días! Es hora de aprender:\n\n{response.choices[0].message.content}"
            await context.bot.send_message(chat_id=user_id, text=msg, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Error in daily job for {user_id}: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command."""
    user_id = update.effective_user.id
    save_user(user_id)
    chat_histories[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    last_interaction[user_id] = time.time()
    
    keyboard = [[KeyboardButton("🔄 Empezar nueva conversación")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    welcome_msg = (
        "¡Hola Megan! ¡Soy Juanito! 🧒🇪🇸\n\n"
        "¡Vamos a jugar a hablar español! Puedes escribirme o **enviarme un audio**. ¿Qué has hecho hoy?\n\n"
        "```\n"
        "Hello Megan! I'm Juanito! Let's play speaking Spanish! You can write to me or send me a voice message. What have you done today?\n"
        "```"
    )
    await update.message.reply_text(welcome_msg, reply_markup=reply_markup, parse_mode="Markdown")

if __name__ == "__main__":
    if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
        logger.error("Missing tokens!")
        exit(1)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Scheduled for 9:00 AM daily in the configured timezone
    tz = pytz.timezone(TIMEZONE)
    app.job_queue.run_daily(daily_word_job, time=dt_time(hour=9, minute=0, second=0, tzinfo=tz))

    logger.info(f"Bot is starting with Whisper and Daily Jobs (Timezone: {TIMEZONE})...")
    app.run_polling()
