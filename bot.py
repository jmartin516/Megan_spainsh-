import os
import re
import sys
import logging
import time
import io
import json
import tempfile
from datetime import date, time as dt_time
import pytz
from typing import Dict, List, Set
from dotenv import load_dotenv

# XTTS (TTS) requires Python 3.10+ due to dependencies like 'bangla' (bool | None syntax)
if sys.version_info < (3, 10):
    print("This bot requires Python 3.10+ for XTTS. Current:", sys.version)
    print("Create a venv with Python 3.10+: python3.10 -m venv venv && source venv/bin/activate")
    sys.exit(1)

from openai import OpenAI
from TTS.api import TTS
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
TIMEZONE = os.getenv("TIMEZONE", "America/Los_Angeles")  # Seattle (Pacific)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize OpenAI Client
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set! Check your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

# Persistence for users and words of the day
DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
WORDS_OF_THE_DAY_FILE = os.path.join(DATA_DIR, "words_of_the_day.json")
_BOT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEAKER_WAV = os.path.join(_BOT_DIR, "MI voz.wav") if os.path.exists(os.path.join(_BOT_DIR, "MI voz.wav")) else os.path.join(DATA_DIR, "speaker_voice.wav")

_xtts_model = None

def get_xtts():
    """Load the XTTS model once (CPU or CUDA)."""
    global _xtts_model
    if _xtts_model is None:
        # Accept Coqui CPML non-commercial terms so XTTS works in Docker/headless (no stdin)
        os.environ["COQUI_TOS_AGREED"] = "1"
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading XTTS model on {device}...")
        _xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _xtts_model

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    # Ensure users.json exists
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump([], f)
    # Ensure words_of_the_day.json exists
    if not os.path.exists(WORDS_OF_THE_DAY_FILE):
        with open(WORDS_OF_THE_DAY_FILE, "w") as f:
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

def load_used_words() -> List[str]:
    """Load the list of words already sent as word of the day."""
    ensure_data_dir()
    if os.path.exists(WORDS_OF_THE_DAY_FILE):
        with open(WORDS_OF_THE_DAY_FILE, "r") as f:
            try:
                data = json.load(f)
                return [item["word"] for item in data] if isinstance(data, list) and data and isinstance(data[0], dict) else (data if isinstance(data, list) else [])
            except (json.JSONDecodeError, KeyError):
                return []
    return []

def save_word_of_the_day(word: str):
    """Append the word of the day to the JSON so we don't repeat it."""
    ensure_data_dir()
    data = []
    if os.path.exists(WORDS_OF_THE_DAY_FILE):
        with open(WORDS_OF_THE_DAY_FILE, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    data.append({"word": word.strip(), "date": str(date.today())})
    with open(WORDS_OF_THE_DAY_FILE, "w") as f:
        json.dump(data, f, indent=2)

def strip_emojis(text: str) -> str:
    """Remove emojis for cleaner TTS."""
    return re.sub(r'[^\w\s,.!?¿¡áéíóúÁÉÍÓÚñÑ]', '', text)

SYSTEM_PROMPT = (
    "Eres Juan, un niño de 14 años de MADRID, ESPAÑA. Hablas con un marcado acento CASTELLANO. "
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

        # --- Generate and send Audio (XTTS - cloned voice from data/speaker_voice.wav or MI voz.wav) ---
        first_line = response_text.split('\n')[0].strip()
        spanish_for_tts = strip_emojis(first_line)

        if spanish_for_tts and os.path.exists(SPEAKER_WAV):
            try:
                tts = get_xtts()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                tts.tts_to_file(
                    text=spanish_for_tts,
                    speaker_wav=SPEAKER_WAV,
                    language="es",
                    file_path=tmp_path,
                )
                with open(tmp_path, "rb") as f:
                    audio_file = io.BytesIO(f.read())
                os.unlink(tmp_path)
                audio_file.seek(0)
                audio_file.name = "voice.wav"
                await update.message.reply_voice(voice=audio_file)
            except Exception as tts_e:
                logger.error(f"TTS Error: {tts_e}")
        elif spanish_for_tts and not os.path.exists(SPEAKER_WAV):
            logger.warning("data/speaker_voice.wav (or MI voz.wav) not found: add a short Spanish audio sample to clone the voice.")
        
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

def _extract_word_from_response(content: str) -> str:
    """Extract the word from the response (first line must be 'PALABRA: <word>')."""
    first_line = content.split("\n")[0].strip()
    if first_line.upper().startswith("PALABRA:"):
        return first_line[8:].strip()  # after "PALABRA:"
    # Fallback: take first word or first quoted thing
    return first_line.split()[0] if first_line else ""

async def daily_word_job(context: ContextTypes.DEFAULT_TYPE):
    """Send a Word of the Day to all users. Words are saved in JSON so we don't repeat."""
    users = load_users()
    logger.info(f"Running daily job for {len(users)} users")
    
    used_words = load_used_words()
    used_list = ", ".join(used_words[-50:]) if used_words else "(none yet)"
    prompt = (
        "Generate a 'Word of the day' in Spanish from Spain for a student. "
        "Include the word, a simple meaning, a Madrid-style usage example, and a short question for the student. "
        "Reply in Juanito's format (10-year-old kid).\n\n"
        "IMPORTANT: Never repeat a word we have already used. Words already used: " + used_list + ".\n\n"
        "Your response MUST start exactly with this line (then a blank line):\n"
        "PALABRA: <the chosen word>\n"
        "After that line, write the rest (meaning, example, question) in Juanito's style."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        word = _extract_word_from_response(content)
        if word:
            save_word_of_the_day(word)
            logger.info(f"Word of the day saved: {word}")
        # Remove the "PALABRA: xxx" line from the message we send (optional, so user sees clean text)
        lines = content.split("\n")
        if lines and lines[0].upper().strip().startswith("PALABRA:"):
            content = "\n".join(lines[1:]).strip()
        msg = f"🌅 ¡Buenos días! Es hora de aprender:\n\n{content}"
        for user_id in users:
            try:
                await context.bot.send_message(chat_id=user_id, text=msg, parse_mode="Markdown")
            except Exception as e:
                logger.error(f"Error in daily job for {user_id}: {e}")
    except Exception as e:
        logger.error(f"Error generating word of the day: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command."""
    user_id = update.effective_user.id
    save_user(user_id)
    chat_histories[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    last_interaction[user_id] = time.time()
    
    keyboard = [[KeyboardButton("🔄 Empezar nueva conversación")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    welcome_msg = (
        "¡Hola Megan! ¡Soy Juan! 🧒🇪🇸\n\n"
        "¡Vamos a jugar a hablar español! Puedes escribirme o **enviarme un audio**. ¿Qué has hecho hoy?\n\n"
        "```\n"
        "Hello Megan! I'm Juan! Let's play speaking Spanish! You can write to me or send me a voice message. What have you done today?\n"
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
