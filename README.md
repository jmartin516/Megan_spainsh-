# Megan Spanish — Telegram bot

A **Telegram bot** that helps learners practice **Spanish (Spain)** by chatting with **Juan**, a 14-year-old character from Madrid. It uses GPT for conversation, Whisper for voice input, and XTTS for voice cloning so replies can be spoken in a custom voice.

---

## What it does

- **Conversation:** Text or voice messages in Spanish; Juan replies in Madrid Spanish with corrections, translations, and follow-up questions.
- **Voice in, voice out:** Send voice notes → transcribed with **Whisper** → reply as text + **voice** (XTTS with a cloned speaker).
- **Word of the day:** Every day at 9:00 (configurable timezone, default Seattle / Pacific) the bot sends a “word of the day” to all users; used words are stored in JSON so they are not repeated.
- **Session memory:** Keeps chat context per user with a 30-minute inactivity timeout.

---

## Stack

| Layer        | Tech |
|-------------|------|
| Bot / jobs   | `python-telegram-bot` (with job queue) |
| LLM         | OpenAI **GPT-4o-mini** |
| Speech → text| OpenAI **Whisper** |
| Text → speech| **Coqui XTTS** (voice cloning from a short WAV sample) |
| Runtime     | **Python 3.10+** (required for XTTS dependencies) |

---

## Prerequisites

- **Python 3.10+**
- **Telegram Bot Token** ([BotFather](https://t.me/BotFather))
- **OpenAI API key**
- A short **Spanish voice sample** (WAV, ~3–10 seconds) for XTTS cloning

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USER/Megan_spainsh-.git
cd Megan_spainsh-
python3.10 -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root:

```env
TELEGRAM_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
TIMEZONE=America/Los_Angeles
```

`TIMEZONE` is optional; default is Seattle (Pacific). Use any [pytz](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones) name (e.g. `Europe/Madrid`).

### 3. Voice sample for XTTS

The bot looks for a speaker WAV in this order:

1. **`MI voz.wav`** in the project root  
2. **`data/speaker_voice.wav`**

Use a single clear Spanish recording, 3–10 seconds, no music. If neither file exists, the bot still runs but will not send voice replies (only text).

### 4. Run

```bash
python bot.py
```

---

## Docker

The repo includes a **Dockerfile** (Python 3.10) and a **GitHub Actions** workflow that builds and pushes an image to GitHub Container Registry.

**Build and run locally:**

```bash
docker build -t megan-spanish-bot .
docker run --env-file .env -v $(pwd)/data:/app/data megan-spanish-bot
```

Mount `data/` so that `users.json` and `words_of_the_day.json` persist across restarts. Pass `TELEGRAM_TOKEN`, `OPENAI_API_KEY`, and optionally `TIMEZONE` via `--env-file` or `-e`. Include the voice file in the image (e.g. copy `MI voz.wav` into the build context) or mount it into `/app`.

---

## Env vars reference

| Variable         | Required | Description |
|------------------|----------|-------------|
| `TELEGRAM_TOKEN` | Yes      | Bot token from BotFather |
| `OPENAI_API_KEY`| Yes      | OpenAI API key (GPT + Whisper) |
| `TIMEZONE`      | No       | Timezone for daily “word of the day” (default: `America/Los_Angeles`) |

---

## Project layout

```
.
├── bot.py              # Main bot (handlers, GPT, Whisper, XTTS, daily job)
├── requirements.txt
├── Dockerfile
├── data/               # Created at runtime: users.json, words_of_the_day.json
├── MI voz.wav          # Optional: XTTS speaker sample (or data/speaker_voice.wav)
└── .env                # Not in repo; add TELEGRAM_TOKEN, OPENAI_API_KEY
```

---

## License

MIT (or your choice). Use and adapt as you like.
