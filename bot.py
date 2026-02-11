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
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
    ConversationHandler,
)

# Load environment variables
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TIMEZONE = (os.getenv("TIMEZONE") or "America/Los_Angeles").strip() or "America/Los_Angeles"
# Name of the person learning Spanish (used in prompts and welcome message). Set in .env for your own use.
STUDENT_NAME = (os.getenv("STUDENT_NAME") or "Megan").strip() or "Megan"
# TTS speech rate: < 1.0 slower, > 1.0 faster. Default 0.9 (a bit slower for learners).
try:
    TTS_SPEED = float(os.getenv("TTS_SPEED", "0.9"))
except ValueError:
    TTS_SPEED = 0.9
TTS_SPEED = max(0.5, min(2.0, TTS_SPEED))

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

def split_main_and_ideas(text: str):
    """Split response into main part and 'Ideas para responder' block. Avoids sending ideas twice."""
    if not text:
        return text, ""
    start = text.find("💡")
    if start == -1:
        start = text.find("---")
    if start == -1:
        start = text.find("Ideas para responder")
    if start == -1:
        return text, ""
    ideas_block = text[start:].strip()
    if len(ideas_block) <= 10:
        return text, ""
    main_text = text[:start].strip()
    return main_text, ideas_block

# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE GAME FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

# Game state storage
game_sessions: Dict[int, dict] = {}

# Conversation states
MENU, QUIZ, ROLEPLAY, TOPIC_CHAT = range(4)

# Quiz questions database (Spanish -> English)
QUIZ_QUESTIONS = [
    {"question": "¿Cómo se dice 'hello' en español?", "options": ["Hola", "Adiós", "Gracias", "Por favor"], "correct": 0},
    {"question": "¿Qué significa 'gracias'?", "options": ["Please", "Thank you", "Sorry", "Goodbye"], "correct": 1},
    {"question": "¿Cómo se dice 'good morning'?", "options": ["Buenas noches", "Buenos días", "Buenas tardes", "Hola"], "correct": 1},
    {"question": "¿Qué significa 'perro'?", "options": ["Cat", "Dog", "Bird", "Fish"], "correct": 1},
    {"question": "¿Cómo se dice 'water'?", "options": ["Vino", "Cerveza", "Agua", "Leche"], "correct": 2},
    {"question": "¿Qué significa 'biblioteca'?", "options": ["Bookstore", "Library", "School", "Office"], "correct": 1},
    {"question": "¿Cómo se dice 'I don't understand'?", "options": ["No sé", "No comprendo", "No quiero", "No puedo"], "correct": 1},
    {"question": "¿Qué significa 'tengo hambre'?", "options": ["I'm thirsty", "I'm hungry", "I'm tired", "I'm happy"], "correct": 1},
]

# Roleplay scenarios
ROLEPLAY_SCENARIOS = {
    "restaurant": {
        "name": "🍽️ En el Restaurante",
        "description": "Eres cliente en un restaurante español. Pide comida, pregunta por el menú, paga la cuenta.",
        "context": "You are at a restaurant in Madrid. You need to order food, ask about the menu, and pay. Be polite but friendly like a local."
    },
    "shopping": {
        "name": "🛍️ De Compras",
        "description": "Vas de compras por las tiendas de Madrid. Pregunta precios, tallas, colores.",
        "context": "You are shopping in Madrid. Ask about prices, sizes, colors. Try to bargain a little - it's fun!"
    },
    "directions": {
        "name": "🗺️ Pidiendo Direcciones",
        "description": "Estás perdido en Madrid. Pide direcciones para llegar a la Puerta del Sol.",
        "context": "You are lost in Madrid and need to get to Puerta del Sol. Ask for directions using local expressions."
    },
    "greetings": {
        "name": "👋 Saludos y Presentaciones",
        "description": "Conoces a un amigo de Juan en el parque. Preséntate y haz conversación.",
        "context": "You meet Juan's friend at Retiro Park. Introduce yourself, talk about where you're from, your hobbies. Use 'tú' form."
    }
}

TOPIC_PROMPTS = {
    "comida": "Hablemos de COMIDA. ¿Cuál es tu comida favorita? ¿Has probado la paella o las tapas? ¡Cuéntame!",
    "viajes": "Hablemos de VIAJES. ¿A qué lugares has ido? ¿Te gustaría visitar España? ¡Cuéntame tus aventuras!",
    "familia": "Hablemos de FAMILIA. ¿Tienes hermanos? ¿Cómo es tu familia? ¡Cuéntame sobre ellos!",
    "hobbies": "Hablemos de HOBBIES. ¿Qué te gusta hacer en tu tiempo libre? ¿Deportes, música, arte?",
    "trabajo": "Hablemos de TU DÍA. ¿Qué hiciste hoy? ¿Qué planes tienes para mañana?"
}

def get_main_menu_keyboard():
    """Return the main menu keyboard."""
    keyboard = [
        [KeyboardButton("💬 Modo Chat"), KeyboardButton("🎯 Quiz")],
        [KeyboardButton("🎭 Roleplay"), KeyboardButton("📚 Tema del Día")],
        [KeyboardButton("ℹ️ Ayuda"), KeyboardButton("🔄 Reiniciar")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_quiz_keyboard(question_idx: int):
    """Return inline keyboard for quiz options."""
    question = QUIZ_QUESTIONS[question_idx]
    keyboard = []
    for i, option in enumerate(question["options"]):
        callback_data = f"quiz:{question_idx}:{i}:{question['correct']}"
        keyboard.append([InlineKeyboardButton(option, callback_data=callback_data)])
    return InlineKeyboardMarkup(keyboard)

def get_roleplay_keyboard():
    """Return inline keyboard for roleplay scenarios."""
    keyboard = []
    for key, scenario in ROLEPLAY_SCENARIOS.items():
        keyboard.append([InlineKeyboardButton(scenario["name"], callback_data=f"roleplay:{key}")])
    keyboard.append([InlineKeyboardButton("🔙 Volver al Menú", callback_data="menu:back")])
    return InlineKeyboardMarkup(keyboard)

def get_topic_keyboard():
    """Return inline keyboard for topic selection."""
    keyboard = [
        [InlineKeyboardButton("🍽️ Comida", callback_data="topic:comida"),
         InlineKeyboardButton("✈️ Viajes", callback_data="topic:viajes")],
        [InlineKeyboardButton("👨‍👩‍👧‍👦 Familia", callback_data="topic:familia"),
         InlineKeyboardButton("🎨 Hobbies", callback_data="topic:hobbies")],
        [InlineKeyboardButton("📅 Mi Día", callback_data="topic:trabajo"),
         InlineKeyboardButton("🔙 Volver", callback_data="menu:back")]
    ]
    return InlineKeyboardMarkup(keyboard)

def save_game_stats(user_id: int, correct: bool = False, points: int = 0):
    """Save game statistics for a user."""
    stats_file = os.path.join(DATA_DIR, f"stats_{user_id}.json")
    stats = {"total_games": 0, "correct_answers": 0, "total_points": 0}
    
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            try:
                stats = json.load(f)
            except:
                pass
    
    stats["total_games"] += 1
    if correct:
        stats["correct_answers"] += 1
    stats["total_points"] += points
    
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    return stats

def load_game_stats(user_id: int) -> dict:
    """Load game statistics for a user."""
    stats_file = os.path.join(DATA_DIR, f"stats_{user_id}.json")
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            try:
                return json.load(f)
            except:
                pass
    return {"total_games": 0, "correct_answers": 0, "total_points": 0}

def _system_prompt():
    return (
        f"Eres Juan, un niño de 14 años de MADRID, ESPAÑA. Hablas con un marcado acento CASTELLANO. "
        f"Tu misión es ayudar a tu amiga {STUDENT_NAME} a perfeccionar su español de España. "
        "Sigue estas REGLAS DE ORO O TE LLEVARÁS UN TIRÓN DE OREJAS:\n"
        "1. VOCABULARIO: Usa palabras de España: 'vale', 'guay', 'mola', 'tío', 'vosotros'. NUNCA uses 'ustedes'.\n"
        f"2. CORRECCIONES: Corrige CUALQUIER error de {STUDENT_NAME}. Sé estricto.\n"
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
        f"5. OBLIGATORIO: Incluye SIEMPRE la sección '💡 Ideas para responder' con al menos 2 ideas concretas (frases que {STUDENT_NAME} pueda decir para seguir la conversación). Sin excepción."
    )


SYSTEM_PROMPT = _system_prompt()

# Global stores
chat_histories: Dict[int, List[dict]] = {}
last_interaction: Dict[int, float] = {}
user_scores: Dict[int, dict] = {}  # New: Gamification scores
user_modes: Dict[int, str] = {}    # New: Current mode per user
SESSION_TIMEOUT = 1800  # 30 minutes in seconds

# Conversation states
MENU, QUIZ, CONVERSATION, GAME_GUESS, PRACTICE = range(5)

# Topic options for guided practice
TOPICS = {
    "comida": "🍽️ Comida y restaurantes",
    "viajes": "✈️ Viajes y vacaciones", 
    "familia": "👨‍👩‍👧‍👦 Familia y amigos",
    "trabajo": "💼 Trabajo y estudios",
    "tiempo": "🌤️ El tiempo y actividades",
    "compras": "🛍️ Compras y dinero"
}

# Game modes description
GAME_MODES = {
    "conversation": "💬 Conversación libre",
    "quiz": "🎯 Modo Quiz",
    "guess": "🎮 Adivina la palabra",
    "practice": "📚 Practicar tema",
    "word": "📖 Palabra del día"
}

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


def get_user_score(user_id: int) -> dict:
    """Get or initialize user score for gamification."""
    if user_id not in user_scores:
        user_scores[user_id] = {
            "points": 0,
            "streak": 0,
            "quizzes_completed": 0,
            "correct_answers": 0,
            "words_learned": 0,
            "last_active": date.today().isoformat()
    }
    return user_scores[user_id]


def add_points(user_id: int, points: int, reason: str = ""):
    """Add points to user's score."""
    score = get_user_score(user_id)
    score["points"] += points
    logger.info(f"User {user_id} earned {points} points ({reason}). Total: {score['points']}")
    return score["points"]


def get_main_menu_keyboard() -> ReplyKeyboardMarkup:
    """Create the main menu keyboard with buttons."""
    keyboard = [
        [KeyboardButton("💬 Conversación libre"), KeyboardButton("🎯 Modo Quiz")],
        [KeyboardButton("🎮 Adivina la palabra"), KeyboardButton("📚 Practicar tema")],
        [KeyboardButton("📖 Palabra del día"), KeyboardButton("🏆 Mi puntuación")],
        [KeyboardButton("🔄 Empezar nueva conversación")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, input_field_placeholder="¿Qué quieres practicar?")


def get_topics_keyboard() -> InlineKeyboardMarkup:
    """Create inline keyboard for topic selection."""
    keyboard = []
    for key, label in TOPICS.items():
        keyboard.append([InlineKeyboardButton(label, callback_data=f"topic:{key}")])
    keyboard.append([InlineKeyboardButton("🔙 Volver al menú", callback_data="menu:back")])
    return InlineKeyboardMarkup(keyboard)


def get_game_modes_keyboard() -> InlineKeyboardMarkup:
    """Create inline keyboard for game mode selection."""
    keyboard = [
        [InlineKeyboardButton("💬 Conversación libre", callback_data="mode:conversation")],
        [InlineKeyboardButton("🎯 Modo Quiz", callback_data="mode:quiz")],
        [InlineKeyboardButton("🎮 Adivina la palabra", callback_data="mode:guess")],
        [InlineKeyboardButton("📚 Practicar tema", callback_data="mode:practice")],
        [InlineKeyboardButton("📖 Palabra del día", callback_data="mode:word")]
    ]
    return InlineKeyboardMarkup(keyboard)

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

        # --- Send main reply first (without ideas block), then ideas as second message to avoid duplicate ---
        main_text, ideas_block = split_main_and_ideas(response_text)
        await update.message.reply_text(main_text, parse_mode="Markdown")
        if ideas_block:
            try:
                await update.message.reply_text(ideas_block, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(ideas_block)

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
                    speed=TTS_SPEED,
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

async def show_score(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show user's current score and stats."""
    user_id = update.effective_user.id
    score = get_user_score(user_id)
    
    score_msg = (
        f"🏆 *Tu puntuación, {STUDENT_NAME}*\n\n"
        f"⭐ Puntos totales: *{score['points']}*\n"
        f"🔥 Racha actual: *{score['streak']}* días\n"
        f"🎯 Quizzes completados: *{score['quizzes_completed']}*\n"
        f"✅ Respuestas correctas: *{score['correct_answers']}*\n"
        f"📖 Palabras aprendidas: *{score['words_learned']}*\n\n"
        "¡Sigue practicando para ganar más puntos! 💪"
    )
    await update.message.reply_text(score_msg, parse_mode="Markdown")


async def start_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start a quiz mode with multiple choice questions."""
    user_id = update.effective_user.id
    user_modes[user_id] = "quiz"
    
    # Generate a quiz question using GPT
    prompt = (
        "Generate a Spanish language quiz question for a beginner/intermediate student. "
        "Format your response EXACTLY like this:\n\n"
        "PREGUNTA: [Question in Spanish]\n"
        "A) [Option A]\n"
        "B) [Option B]\n"
        "C) [Option C]\n"
        "D) [Option D]\n"
        "CORRECTA: [A/B/C/D]\n"
        "EXPLICACION: [Brief explanation in Spanish with English translation]"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        
        quiz_text = response.choices[0].message.content
        
        # Parse the quiz
        lines = quiz_text.strip().split('\n')
        question = ""
        options = {}
        correct = ""
        explanation = ""
        
        for line in lines:
            if line.startswith("PREGUNTA:"):
                question = line[9:].strip()
            elif line.startswith("A)"):
                options["A"] = line[2:].strip()
            elif line.startswith("B)"):
                options["B"] = line[2:].strip()
            elif line.startswith("C)"):
                options["C"] = line[2:].strip()
            elif line.startswith("D)"):
                options["D"] = line[2:].strip()
            elif line.startswith("CORRECTA:"):
                correct = line[9:].strip().upper()
            elif line.startswith("EXPLICACION:"):
                explanation = line[12:].strip()
        
        # Store quiz data in context
        context.user_data['current_quiz'] = {
            'question': question,
            'options': options,
            'correct': correct,
            'explanation': explanation
        }
        
        # Create inline keyboard with options
        keyboard = [
            [InlineKeyboardButton(f"A) {options['A']}", callback_data="quiz:A")],
            [InlineKeyboardButton(f"B) {options['B']}", callback_data="quiz:B")],
            [InlineKeyboardButton(f"C) {options['C']}", callback_data="quiz:C")],
            [InlineKeyboardButton(f"D) {options['D']}", callback_data="quiz:D")]
        ]
        
        await update.message.reply_text(
            f"🎯 *Modo Quiz*\n\n{question}",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
        
    except Exception as e:
        logger.error(f"Quiz generation error: {e}")
        await update.message.reply_text(
            "¡Ups! No pude crear el quiz. ¿Volvemos a intentarlo? 🎯",
            reply_markup=get_main_menu_keyboard()
        )


async def handle_quiz_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle quiz answer callback."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    answer = query.data.split(":")[1]
    quiz = context.user_data.get('current_quiz')
    
    if not quiz:
        await query.edit_message_text("El quiz expiró. ¡Intentemos otro! 🎯")
        return
    
    is_correct = answer == quiz['correct']
    score = get_user_score(user_id)
    score['quizzes_completed'] += 1
    
    if is_correct:
        score['correct_answers'] += 1
        points = add_points(user_id, 10, "Quiz correct answer")
        result_emoji = "✅"
        result_text = f"¡Correcto! ¡Muy bien, {STUDENT_NAME}! 🎉"
    else:
        points = add_points(user_id, 2, "Quiz attempt")
        result_emoji = "❌"
        result_text = f"¡Casi! La respuesta correcta era: *{quiz['correct']}) {quiz['options'][quiz['correct']]}*"
    
    response = (
        f"{result_emoji} *{result_text}*\n\n"
        f"💡 {quiz['explanation']}\n\n"
        f"⭐ Puntos ganados: +{10 if is_correct else 2}\n"
        f"🏆 Total: {points} puntos"
    )
    
    # Add buttons for next action
    keyboard = [
        [InlineKeyboardButton("🎯 Otro quiz", callback_data="quiz:next")],
        [InlineKeyboardButton("🔙 Menú principal", callback_data="menu:back")]
    ]
    
    await query.edit_message_text(
        response,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )


async def start_guess_game(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start the 'Guess the Word' game."""
    user_id = update.effective_user.id
    user_modes[user_id] = "guess"
    
    # Generate a word to guess
    prompt = (
        "Generate a simple Spanish word for a beginner to guess. "
        "Provide a riddle/clue in Spanish. Format:\n\n"
        "PALABRA: [The word]\n"
        "PISTA: [A riddle/clue in Spanish describing the word]\n"
        "FACIL: [An easier hint]\n"
        "DIFICULTAD: [easy/medium/hard]"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9
        )
        
        content = response.choices[0].message.content
        lines = content.strip().split('\n')
        
        word = ""
        clue = ""
        easy_hint = ""
        difficulty = "medium"
        
        for line in lines:
            if line.startswith("PALABRA:"):
                word = line[8:].strip().lower()
            elif line.startswith("PISTA:"):
                clue = line[6:].strip()
            elif line.startswith("FACIL:"):
                easy_hint = line[6:].strip()
            elif line.startswith("DIFICULTAD:"):
                difficulty = line[11:].strip()
        
        context.user_data['guess_word'] = {
            'word': word,
            'clue': clue,
            'easy_hint': easy_hint,
            'difficulty': difficulty,
            'attempts': 0
        }
        
        difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(difficulty, "🟡")
        
        await update.message.reply_text(
            f"🎮 *¡Adivina la palabra!* {difficulty_emoji}\n\n"
            f"🤔 *Pista:* {clue}\n\n"
            f"Escribe la palabra en español. ¡Tienes 3 intentos!\n\n"
            f"💡 Escribe 'pista' para una ayuda más fácil (pero ganarás menos puntos)",
            parse_mode="Markdown"
        )
        
    except Exception as e:
        logger.error(f"Guess game error: {e}")
        await update.message.reply_text(
            "¡Ups! No pude crear el juego. ¿Volvemos al menú? 🎮",
            reply_markup=get_main_menu_keyboard()
        )


async def handle_guess_attempt(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str):
    """Handle word guessing attempts."""
    user_id = update.effective_user.id
    guess_data = context.user_data.get('guess_word')
    
    if not guess_data:
        await update.message.reply_text(
            "No hay juego activo. ¡Empecemos uno nuevo! 🎮",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    user_text_lower = user_text.lower().strip()
    
    # Check if asking for hint
    if user_text_lower == "pista":
        await update.message.reply_text(
            f"💡 *Pista fácil:* {guess_data['easy_hint']}\n\n"
            f"(Usar pista = menos puntos)",
            parse_mode="Markdown"
        )
        guess_data['used_hint'] = True
        return
    
    guess_data['attempts'] += 1
    correct_word = guess_data['word'].lower()
    
    # Check answer
    if user_text_lower == correct_word:
        # Correct!
        base_points = {"easy": 15, "medium": 25, "hard": 40}.get(guess_data['difficulty'], 20)
        if guess_data.get('used_hint'):
            base_points = base_points // 2
        
        points = add_points(user_id, base_points, f"Guessed word in {guess_data['attempts']} attempts")
        
        await update.message.reply_text(
            f"🎉 *¡Correcto!* ¡La palabra era: {guess_data['word'].upper()}!\n\n"
            f"⭐ +{base_points} puntos\n"
            f"🏆 Total: {points} puntos\n"
            f"🎯 Intentos: {guess_data['attempts']}/3\n\n"
            f"¡Muy bien, {STUDENT_NAME}!",
            reply_markup=get_main_menu_keyboard(),
            parse_mode="Markdown"
        )
        context.user_data.pop('guess_word', None)
        user_modes[user_id] = "menu"
        
    elif guess_data['attempts'] >= 3:
        # Game over
        await update.message.reply_text(
            f"😅 *¡Se acabaron los intentos!*\n\n"
            f"La palabra era: *{guess_data['word'].upper()}*\n\n"
            f"¡No pasa nada! Inténtalo de nuevo 🎮",
            reply_markup=get_main_menu_keyboard(),
            parse_mode="Markdown"
        )
        context.user_data.pop('guess_word', None)
        user_modes[user_id] = "menu"
        
    else:
        # Wrong but can try again
        remaining = 3 - guess_data['attempts']
        await update.message.reply_text(
            f"❌ *No es correcto*\n\n"
            f"Te quedan {remaining} intentos.\n"
            f"💡 Escribe 'pista' si necesitas ayuda",
            parse_mode="Markdown"
        )


async def show_topics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show topic selection for guided practice."""
    await update.message.reply_text(
        "📚 *Elige un tema para practicar:*\n\n"
        "Selecciona uno y practicaremos vocabulario específico",
        reply_markup=get_topics_keyboard(),
        parse_mode="Markdown"
    )


async def handle_topic_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle topic selection callback."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    topic_key = query.data.split(":")[1]
    topic_name = TOPICS.get(topic_key, topic_key)
    
    user_modes[user_id] = "practice"
    
    # Generate practice content for this topic
    prompt = (
        f"Create a vocabulary practice session about '{topic_key}' in Spanish. "
        f"Give {STUDENT_NAME} 5 useful phrases/words in Spanish about this topic, "
        "with their English translations. Then ask them to use one in a sentence. "
        "Make it fun and encouraging, like a game!"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        
        content = response.choices[0].message.content
        
        # Add buttons to continue or change topic
        keyboard = [
            [InlineKeyboardButton("🔄 Más vocabulario", callback_data=f"topic:{topic_key}")],
            [InlineKeyboardButton("📚 Cambiar tema", callback_data="mode:practice")],
            [InlineKeyboardButton("🔙 Menú principal", callback_data="menu:back")]
        ]
        
        await query.edit_message_text(
            f"📚 *{topic_name}*\n\n{content}",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )
        
    except Exception as e:
        logger.error(f"Topic practice error: {e}")
        await query.edit_message_text(
            "¡Ups! Error cargando el tema. ¿Intentamos otro? 📚",
            reply_markup=get_topics_keyboard()
        )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all inline keyboard callbacks."""
    query = update.callback_query
    data = query.data
    
    if data.startswith("quiz:"):
        if data == "quiz:next":
            await start_quiz(update, context)
        else:
            await handle_quiz_answer(update, context)
    
    elif data.startswith("topic:"):
        await handle_topic_selection(update, context)
    
    elif data.startswith("mode:"):
        mode = data.split(":")[1]
        if mode == "conversation":
            await query.edit_message_text("💬 Modo conversación activado. ¡Hablemos!")
            await start_conversation_mode(update, context)
        elif mode == "quiz":
            await start_quiz(update, context)
        elif mode == "guess":
            await start_guess_game(update, context)
        elif mode == "practice":
            await show_topics(update, context)
        elif mode == "word":
            await send_word_of_day_manual(update, context)
    
    elif data == "menu:back":
        await query.edit_message_text("🔙 Volviendo al menú principal...")
        await start(update, context)


async def start_conversation_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start free conversation mode."""
    user_id = update.effective_user.id
    user_modes[user_id] = "conversation"
    
    # Get the effective message object
    if update.callback_query:
        message = update.callback_query.message
    else:
        message = update.message
    
    await message.reply_text(
        f"💬 *Modo Conversación*\n\n"
        f"¡Perfecto, {STUDENT_NAME}! Cuéntame algo... ¿qué has hecho hoy? "
        f"¿Tienes algún plan? ¿Quieres practicar algo específico?\n\n"
        f"Puedes escribirme o enviarme un audio 🎤",
        reply_markup=get_main_menu_keyboard(),
        parse_mode="Markdown"
    )


async def send_word_of_day_manual(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send word of the day on demand."""
    # Use callback or message depending on source
    if update.callback_query:
        await daily_word_job(context)
    else:
        await daily_word_job(context)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages with mode routing."""
    user_id = update.effective_user.id
    user_text = update.message.text
    current_mode = user_modes.get(user_id, "menu")
    
    # Handle menu buttons first
    if user_text == "🔄 Empezar nueva conversación":
        await start(update, context)
        return
    elif user_text == "🏆 Mi puntuación":
        await show_score(update, context)
        return
    elif user_text == "💬 Conversación libre":
        await start_conversation_mode(update, context)
        return
    elif user_text == "🎯 Modo Quiz":
        await start_quiz(update, context)
        return
    elif user_text == "🎮 Adivina la palabra":
        await start_guess_game(update, context)
        return
    elif user_text == "📚 Practicar tema":
        await show_topics(update, context)
        return
    elif user_text == "📖 Palabra del día":
        await daily_word_job(context)
        return
    
    # Handle game modes
    if current_mode == "guess":
        await handle_guess_attempt(update, context, user_text)
        return
    elif current_mode == "quiz":
        # In quiz mode, only callback buttons should respond
        await update.message.reply_text(
            "🎯 Estás en modo Quiz. Responde usando los botones de arriba 👆",
            reply_markup=get_main_menu_keyboard()
        )
        return
    
    # Default: conversation mode
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
    """Handle the /start command with interactive menu."""
    user_id = update.effective_user.id
    save_user(user_id)
    chat_histories[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    last_interaction[user_id] = time.time()
    user_modes[user_id] = "menu"
    
    score = get_user_score(user_id)
    
    welcome_msg = (
        f"¡Hola {STUDENT_NAME}! ¡Soy Juan! 🧒🇪🇸\n\n"
        "¡Vamos a jugar y aprender español juntos! Elige cómo quieres practicar hoy:\n\n"
        "💬 *Conversación* - Hablamos de lo que quieras\n"
        "🎯 *Modo Quiz* - Preguntas de opción múltiple\n"
        "🎮 *Adivina* - Adivina la palabra secreta\n"
        "📚 *Practicar tema* - Practica vocabulario específico\n"
        "📖 *Palabra del día* - Aprende una palabra nueva\n\n"
        f"🏆 *Tus puntos:* {score['points']} | 🔥 *Racha:* {score['streak']} días\n\n"
        "```\n"
        f"Hi {STUDENT_NAME}! I'm Juan! Let's play and learn Spanish together! Choose how you want to practice today:\n\n"
        "💬 Conversation - Chat about anything\n"
        "🎯 Quiz Mode - Multiple choice questions\n"
        "🎮 Guess - Guess the secret word\n"
        "📚 Practice topic - Practice specific vocabulary\n"
        "📖 Word of the day - Learn a new word\n"
        "```"
    )
    
    await update.message.reply_text(
        welcome_msg, 
        reply_markup=get_main_menu_keyboard(), 
        parse_mode="Markdown"
    )

if __name__ == "__main__":
    if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
        logger.error("Missing tokens!")
        exit(1)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Scheduled for 9:00 AM daily in the configured timezone
    tz = pytz.timezone(TIMEZONE)
    app.job_queue.run_daily(daily_word_job, time=dt_time(hour=9, minute=0, second=0, tzinfo=tz))

    logger.info(f"🚀 Bot starting with Interactive Modes (Timezone: {TIMEZONE})...")
    logger.info(f"🎮 Available modes: {list(GAME_MODES.keys())}")
    app.run_polling()
