# Telegram Voice Language Coach

A Telegram bot for practising spoken languages with voice transcription, AI-powered feedback, text-to-speech replies, progress tracking, vocabulary, and role-play scenarios.

## Highlights

- Voice-message transcription and conversational replies
- Grammar, vocabulary, and clarity feedback
- Personal vocabulary and progress tracking
- Study streaks, XP, levels, and rewards
- Role-play scenarios for real-world practice
- Support for more than 20 languages
- SQLite persistence for local development

## Tech stack

- Python 3.11+
- aiogram 3
- OpenAI API
- SQLite
- python-dotenv

## Quick start

1. Clone the repository and create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create your local environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

5. Add your own credentials to `.env`:

```env
TELEGRAM_BOT_TOKEN=your-token
OPENAI_API_KEY=your-key
```

6. Run the bot:

```bash
python bot.py
```

## Commands

- `/start` — open the main menu
- `/setlang <language>` — choose a study language
- `/setlevel A1|A2|B1|B2|C1|C2` — choose your level
- `/progress` — view learning statistics
- `/words` — view saved vocabulary
- `/rewards` — view XP and streaks
- `/scenario restaurant|airport|interview|shopping|football|date|off` — manage role-play mode

## Privacy and security

- Never commit `.env`, API keys, Telegram tokens, or production databases.
- The local `progress.db` file may contain user learning history and must remain outside Git.
- Use a separate test bot and test API credentials during development.
- Review data-retention and privacy requirements before deploying for real users.

## Project status

This is an educational MVP. Before production use, it should be split into smaller modules and extended with automated tests, rate limiting, structured logging, migrations, and stronger retry handling.

## Responsible use

Use the bot only with data and accounts you are authorized to process. Do not publish user conversations or personal learning history.
