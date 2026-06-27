# Telegram Voice Language Coach Bot

This bot helps you practice spoken languages in Telegram:

- accepts your voice message
- transcribes speech to text
- analyzes mistakes with an LLM
- shows feedback in one navigable card instead of sending every section at once
- sends a voice response in your selected language
- keeps recent spoken turns in context to continue the conversation naturally
- remembers user-provided, non-sensitive details such as hobbies or favourite team
- tracks scores and a practice streak for every study language
- saves useful words from conversations into a personal vocabulary

## 1) Create your bot token

1. Open [@BotFather](https://t.me/BotFather) in Telegram.
2. Run `/newbot`.
3. Copy the bot token.

## 2) Prepare environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Configure secrets

```bash
copy .env.example .env
```

Edit `.env` and set:

- `TELEGRAM_BOT_TOKEN`
- `OPENAI_API_KEY`

## 4) Run bot

```bash
python bot.py
```

## 5) Use

1. Open your bot in Telegram.
2. Send `/start`.
3. Select a language with the `🌐 Язык` button (or `/setlang <language>`).
4. Send a voice message in selected language (10-40 seconds is ideal).

The bot also shows a persistent button menu for progress, vocabulary, rewards, level, language and role-play scenes—commands remain available too.

You will receive:

- an interactive feedback card: overview, mistakes, corrected version, task and transcript open with buttons
- a `🇷🇺 На русском` switch for Russian versions of errors, corrections and tasks
- Latin reading next to examples and vocabulary for Chinese, Japanese, Korean, Arabic, Greek, Hindi and Persian
- voice response in the same language (longer conversational reply; may arrive as several voice messages if needed)

When studying **Turkish** or **Russian**, the main feedback card stays compact; the bot still prepares the additional translation and pronunciation information for the analysis.

After each voice reply from the bot, a **text translation** of that speech is sent: English or Turkish voice → Russian; Russian voice → Turkish. If the model omits the inline translation block, the bot translates the voice text in a second API call.
- scores (overall/grammar/vocabulary/clarity)
- one-minute homework task

Optional `.env` tuning for voice length:

- `TTS_SPEED` — values below `1.0` make speech slower and the same text sound longer (for example `0.88`-`0.95`).
- `TTS_MAX_CHARS_PER_CHUNK` — max characters per TTS request (API limit 4096; default 3800). Long replies are split automatically.

The dialogue and feedback model defaults to `gpt-5.5`. You can override it with `ANALYSIS_MODEL` in `.env`; a stronger model may cost more and respond more slowly.

## Notes

- This MVP uses OpenAI APIs for transcription, analysis and speech generation.
- Bot stores your learning history in SQLite (`progress.db`) for personalized feedback.
- Commands:
  - `/progress` - show score change, skill averages, streak and frequent mistakes
  - `/words` - show recently added words for the current study language
  - `/rewards` - show XP, title and your current daily streak
  - `/scenario restaurant|airport|interview|shopping|football|date|off` - choose or turn off a real-life role-play
  - `/setlevel A1|A2|B1|B2|C1|C2` - set learning level
  - `/setlang <language>` - set study language
- Text messages (not starting with `/`): with **Turkish** as the study language, ask in English, Russian, or Turkish. With **English** or **Russian** as the study language, ask about Turkish vocabulary or grammar; answers follow the main study language (English or Russian).
- For production, consider adding:
  - pronunciation scoring based on phoneme-level tools
  - rate limiting and better error retries

Supported languages: English, Russian, Turkish, Serbian, Ukrainian, Tatar, Chechen, Nogai, Mandarin Chinese, Japanese, Korean, Indonesian, Arabic, Kurdish (Kurmanji), German, French, Spanish, Italian, Portuguese, Polish, Dutch, Greek, Hindi, Persian and Swedish.
