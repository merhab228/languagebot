import asyncio
import io
import logging
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI


@dataclass
class Settings:
    telegram_token: str
    openai_api_key: str
    transcribe_model: str
    analysis_model: str
    tts_model: str
    tts_voice: str
    tts_speed: float
    tts_max_chars_per_chunk: int
    db_path: str


def load_settings() -> Settings:
    load_dotenv()

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    transcribe_model = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe").strip()
    analysis_model = os.getenv("ANALYSIS_MODEL", "gpt-5.5").strip()
    tts_model = os.getenv("TTS_MODEL", "gpt-4o-mini-tts").strip()
    tts_voice = os.getenv("TTS_VOICE", "alloy").strip()
    tts_speed = float(os.getenv("TTS_SPEED", "0.92").strip())
    tts_max_chars_per_chunk = int(os.getenv("TTS_MAX_CHARS_PER_CHUNK", "3800").strip())
    db_path = os.getenv("DB_PATH", "progress.db").strip()

    if not telegram_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is missing in .env")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is missing in .env")

    return Settings(
        telegram_token=telegram_token,
        openai_api_key=openai_api_key,
        transcribe_model=transcribe_model,
        analysis_model=analysis_model,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_speed=max(0.25, min(4.0, tts_speed)),
        tts_max_chars_per_chunk=max(500, min(4096, tts_max_chars_per_chunk)),
        db_path=db_path,
    )


INSTRUCTION_TEXT = (
    "Send me a voice message in your selected study language.\n"
    "I will:\n"
    "1) transcribe what you said,\n"
    "2) analyze your mistakes,\n"
    "3) send a short memo in text,\n"
    "4) answer with voice in the same language.\n"
    "5) send a text translation of the voice reply (EN/TR→RU, RU→TR).\n\n"
    "For Turkish study: text feedback is sent in Turkish and Russian.\n"
    "For Russian study: text feedback is sent in Russian and Turkish; accent and endings are analyzed.\n\n"
    "Tip: speak for 10-40 seconds for best quality.\n"
    "Use /progress to see your scores, streak and common mistakes.\n"
    "Use /words to repeat your personal vocabulary.\n"
    "Use /rewards to see XP, level and daily streak.\n"
    "Use /scenario restaurant|airport|interview|shopping|football|date|off for a real-life role-play.\n"
    "Use /setlevel A2 (or A1..C2) to change training level.\n"
    "Use the 🌐 Language button (or /setlang <language>) to choose a study language.\n\n"
    "Text questions:\n"
    "- While studying Turkish, you may type questions in English or Russian (or Turkish).\n"
    "- While studying English or Russian, you may ask about Turkish words or grammar in a text message."
)

SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "english": {
        "label": "English",
        "coach_language": "English",
        "fallback_reply": "Great effort. Keep speaking every day and focus on clear, short sentences.",
    },
    "russian": {
        "label": "Russian",
        "coach_language": "Russian",
        "fallback_reply": "Отличная работа. Говорите каждый день и старайтесь строить короткие, ясные фразы.",
    },
    "turkish": {
        "label": "Turkish",
        "coach_language": "Turkish",
        "fallback_reply": "Harika bir çaba. Her gun konusmaya devam et ve cumlelerini kisa ve net tut.",
    },
    "serbian": {
        "label": "Serbian",
        "coach_language": "Serbian",
        "fallback_reply": "Odlican trud. Nastavi da govoris svakog dana i koristi jasne, kratke recenice.",
    },
    "ukrainian": {
        "label": "Ukrainian",
        "coach_language": "Ukrainian",
        "fallback_reply": "Чудова робота. Говоріть щодня та намагайтеся будувати короткі й чіткі речення.",
    },
    "tatar": {
        "label": "Tatar",
        "coach_language": "Tatar",
        "fallback_reply": "Бик яхшы тырышлык. Көн саен сөйләш һәм фикерләреңне кыска, аңлаешлы җөмләләр белән әйт.",
    },
    "chechen": {
        "label": "Chechen",
        "coach_language": "Chechen",
        "fallback_reply": "Хьо дика къахьегам баьккхина. Къамел де кхета-кхета, кIезиг а, кхетамца а предложенеш ло.",
    },
    "nogai": {
        "label": "Nogai",
        "coach_language": "Nogai",
        "fallback_reply": "Сен тырысыуынъ бек яхшы. Ар куьн сойлеп тур да, ойлерини кыска да анык айт.",
    },
    "chinese": {"label": "Chinese (Mandarin)", "coach_language": "Mandarin Chinese", "fallback_reply": "你说得很好。我们继续慢慢聊。", "latin_reading": "pinyin with tone marks"},
    "japanese": {"label": "Japanese", "coach_language": "Japanese", "fallback_reply": "よく話せました。これからもゆっくり話しましょう。", "latin_reading": "Hepburn romaji"},
    "korean": {"label": "Korean", "coach_language": "Korean", "fallback_reply": "정말 잘했어요. 계속 편하게 이야기해 봐요.", "latin_reading": "Revised Romanization"},
    "indonesian": {"label": "Indonesian", "coach_language": "Indonesian", "fallback_reply": "Kamu sudah berbicara dengan baik. Mari kita lanjutkan percakapannya."},
    "arabic": {"label": "Arabic", "coach_language": "Modern Standard Arabic", "fallback_reply": "أحسنت. لنواصل الحديث بهدوء.", "latin_reading": "standard Arabic Latin transliteration"},
    "kurmanji": {"label": "Kurdish (Kurmanji)", "coach_language": "Kurdish Kurmanji", "fallback_reply": "Tu baş axivî. Bila em axaftinê bidomînin."},
    "german": {"label": "German", "coach_language": "German", "fallback_reply": "Das hast du gut gemacht. Lass uns weiterreden."},
    "french": {"label": "French", "coach_language": "French", "fallback_reply": "Tu t'exprimes bien. Continuons à discuter."},
    "spanish": {"label": "Spanish", "coach_language": "Spanish", "fallback_reply": "Lo hiciste muy bien. Sigamos hablando."},
    "italian": {"label": "Italian", "coach_language": "Italian", "fallback_reply": "Hai parlato bene. Continuiamo a parlare."},
    "portuguese": {"label": "Portuguese", "coach_language": "Portuguese", "fallback_reply": "Você falou muito bem. Vamos continuar conversando."},
    "polish": {"label": "Polish", "coach_language": "Polish", "fallback_reply": "Dobrze ci poszło. Rozmawiajmy dalej."},
    "dutch": {"label": "Dutch", "coach_language": "Dutch", "fallback_reply": "Je hebt het goed gedaan. Laten we verder praten."},
    "greek": {"label": "Greek", "coach_language": "Greek", "fallback_reply": "Τα πήγες πολύ καλά. Ας συνεχίσουμε τη συζήτηση.", "latin_reading": "standard Greek Latin transliteration"},
    "hindi": {"label": "Hindi", "coach_language": "Hindi", "fallback_reply": "आपने अच्छा बोला। चलिए बात जारी रखें।", "latin_reading": "standard Hindi Latin transliteration"},
    "persian": {"label": "Persian", "coach_language": "Persian", "fallback_reply": "خیلی خوب صحبت کردی. بیایید به گفتگو ادامه دهیم.", "latin_reading": "standard Persian Latin transliteration"},
    "swedish": {"label": "Swedish", "coach_language": "Swedish", "fallback_reply": "Du pratade bra. Låt oss fortsätta samtalet."},
}


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                level TEXT NOT NULL DEFAULT 'B1',
                study_language TEXT NOT NULL DEFAULT 'english',
                display_name TEXT,
                awaiting_name INTEGER NOT NULL DEFAULT 0,
                conversation_scenario TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                study_language TEXT NOT NULL DEFAULT 'english',
                transcript TEXT NOT NULL,
                analysis TEXT NOT NULL,
                overall_score INTEGER,
                grammar_score INTEGER,
                vocabulary_score INTEGER,
                clarity_score INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS vocabulary_words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                study_language TEXT NOT NULL,
                word TEXT NOT NULL,
                transliteration TEXT NOT NULL DEFAULT '',
                translation TEXT NOT NULL,
                example TEXT NOT NULL DEFAULT '',
                added_at TEXT NOT NULL,
                review_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(user_id, study_language, word)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_activity (
                user_id INTEGER NOT NULL,
                activity_date TEXT NOT NULL,
                xp INTEGER NOT NULL DEFAULT 0,
                voice_count INTEGER NOT NULL DEFAULT 0,
                words_learned INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, activity_date)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memories (
                user_id INTEGER NOT NULL,
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, memory_key)
            )
            """
        )
        try:
            cursor.execute("ALTER TABLE user_profiles ADD COLUMN study_language TEXT NOT NULL DEFAULT 'english'")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE sessions ADD COLUMN study_language TEXT NOT NULL DEFAULT 'english'")
        except sqlite3.OperationalError:
            pass
        for statement in (
            "ALTER TABLE user_profiles ADD COLUMN display_name TEXT",
            "ALTER TABLE user_profiles ADD COLUMN awaiting_name INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE user_profiles ADD COLUMN conversation_scenario TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE sessions ADD COLUMN overall_score INTEGER",
            "ALTER TABLE sessions ADD COLUMN grammar_score INTEGER",
            "ALTER TABLE sessions ADD COLUMN vocabulary_score INTEGER",
            "ALTER TABLE sessions ADD COLUMN clarity_score INTEGER",
            "ALTER TABLE vocabulary_words ADD COLUMN transliteration TEXT NOT NULL DEFAULT ''",
        ):
            try:
                cursor.execute(statement)
            except sqlite3.OperationalError:
                pass
        conn.commit()
    finally:
        conn.close()


def get_user_level(user_id: int) -> str:
    conn = sqlite3.connect(settings.db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT level FROM user_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return row[0]
        now = datetime.utcnow().isoformat()
        cursor.execute(
            "INSERT INTO user_profiles (user_id, level, updated_at) VALUES (?, ?, ?)",
            (user_id, "B1", now),
        )
        conn.commit()
        return "B1"
    finally:
        conn.close()


def set_user_level(user_id: int, level: str) -> None:
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(settings.db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_profiles (user_id, level, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                level = excluded.level,
                updated_at = excluded.updated_at
            """,
            (user_id, level, now),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_study_language(user_id: int) -> str:
    conn = sqlite3.connect(settings.db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT study_language FROM user_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row and row[0] in SUPPORTED_LANGUAGES:
            return row[0]
        now = datetime.utcnow().isoformat()
        cursor.execute(
            """
            INSERT INTO user_profiles (user_id, level, study_language, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                study_language = excluded.study_language,
                updated_at = excluded.updated_at
            """,
            (user_id, "B1", "english", now),
        )
        conn.commit()
        return "english"
    finally:
        conn.close()


def set_user_study_language(user_id: int, language: str) -> None:
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(settings.db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_profiles (user_id, level, study_language, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                study_language = excluded.study_language,
                updated_at = excluded.updated_at
            """,
            (user_id, "B1", language, now),
        )
        conn.commit()
    finally:
        conn.close()


def set_intro_waiting(user_id: int, waiting: bool) -> None:
    conn = sqlite3.connect(settings.db_path)
    try:
        conn.execute(
            "UPDATE user_profiles SET awaiting_name = ?, updated_at = ? WHERE user_id = ?",
            (int(waiting), datetime.utcnow().isoformat(), user_id),
        )
        conn.commit()
    finally:
        conn.close()


def save_display_name(user_id: int, name: str) -> None:
    conn = sqlite3.connect(settings.db_path)
    try:
        conn.execute(
            "UPDATE user_profiles SET display_name = ?, awaiting_name = 0, updated_at = ? WHERE user_id = ?",
            (name[:60], datetime.utcnow().isoformat(), user_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_dialogue_context(user_id: int, study_language: str) -> tuple[Optional[str], list[str]]:
    conn = sqlite3.connect(settings.db_path)
    try:
        profile = conn.execute("SELECT display_name FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
        rows = conn.execute(
            "SELECT transcript FROM sessions WHERE user_id = ? AND study_language = ? ORDER BY id DESC LIMIT 3",
            (user_id, study_language),
        ).fetchall()
    finally:
        conn.close()
    return (profile[0] if profile and profile[0] else None, [row[0][:700] for row in reversed(rows)])


def build_dialogue_context(name: Optional[str], recent_turns: list[str]) -> str:
    if not recent_turns:
        return (
            "Dialogue context: this is the learner's first spoken turn in this language. In VOICE_REPLY, "
            "briefly introduce yourself as their warm, curious speaking partner. React to what they actually "
            "said before asking anything. Naturally learn what they like to be called if it is unknown, and "
            "open one concrete everyday topic.\n\n"
        )
    history = "\n".join(f"- Learner: {turn}" for turn in recent_turns)
    name_hint = f"The learner's name is {name}. " if name else "The learner's name is not known yet. "
    return (
        "Dialogue context from previous spoken turns:\n"
        f"{history}\n\n{name_hint}"
        "For VOICE_REPLY, continue the same topic when it still has life: refer to a specific detail from "
        "the learner's last turn, answer it, and ask a related open question. Do not restart with a generic "
        "greeting. When the topic is naturally complete, introduce a fresh but connected topic using a bridge "
        "(for example, from travel to food, work to routines, or a feeling to a recent experience). State the "
        "bridge in one natural sentence and give the learner an easy way to choose the new direction.\n\n"
    )


SCENARIOS = {
    "restaurant": "a relaxed restaurant or café conversation: ordering, preferences and small talk",
    "airport": "an airport or travel situation: check-in, directions, delays and friendly small talk",
    "interview": "a realistic job interview: professional but supportive questions and follow-ups",
    "shopping": "a shopping situation: asking for items, comparing options and making decisions",
    "football": "a friendly football discussion: a recent match, teams, players and opinions",
    "date": "a respectful, light first-date conversation: interests, humor and getting to know each other",
}


def get_user_scenario(user_id: int) -> str:
    conn = sqlite3.connect(settings.db_path)
    try:
        row = conn.execute("SELECT conversation_scenario FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
        return row[0] if row and row[0] in SCENARIOS else ""
    finally:
        conn.close()


def set_user_scenario(user_id: int, scenario: str) -> None:
    conn = sqlite3.connect(settings.db_path)
    try:
        conn.execute(
            "UPDATE user_profiles SET conversation_scenario = ?, updated_at = ? WHERE user_id = ?",
            (scenario, datetime.utcnow().isoformat(), user_id),
        )
        conn.commit()
    finally:
        conn.close()


def extract_memory_updates(analysis: str) -> list[tuple[str, str]]:
    if "MEMORY_UPDATE:" not in analysis:
        return []
    raw = analysis.split("MEMORY_UPDATE:", 1)[1]
    for marker in ("NEW_WORDS:", "VOICE_REPLY:", "VOICE_REPLY_TRANSLATION:"):
        if "\n" + marker in raw:
            raw = raw.split("\n" + marker, 1)[0]
    allowed_keys = {"city", "work", "hobbies", "favorite_team", "music", "food", "travel", "language_goal"}
    updates: list[tuple[str, str]] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.strip().lstrip("-• ").split("|", 1)]
        if len(parts) == 2 and parts[0] in allowed_keys and 1 <= len(parts[1]) <= 240:
            updates.append((parts[0], parts[1]))
    return updates[:3]


def save_memory_updates(user_id: int, updates: list[tuple[str, str]]) -> None:
    if not updates:
        return
    conn = sqlite3.connect(settings.db_path)
    try:
        conn.executemany(
            """
            INSERT INTO user_memories (user_id, memory_key, memory_value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, memory_key) DO UPDATE SET
                memory_value = excluded.memory_value, updated_at = excluded.updated_at
            """,
            [(user_id, key, value, datetime.utcnow().isoformat()) for key, value in updates],
        )
        conn.commit()
    finally:
        conn.close()


def build_personal_memory_context(user_id: int, scenario: str) -> str:
    conn = sqlite3.connect(settings.db_path)
    try:
        rows = conn.execute(
            "SELECT memory_key, memory_value FROM user_memories WHERE user_id = ? ORDER BY updated_at DESC LIMIT 8",
            (user_id,),
        ).fetchall()
    finally:
        conn.close()
    memory = "; ".join(f"{key}: {value}" for key, value in rows)
    parts = []
    if memory:
        parts.append(
            "Remembered learner details (use one only when it is naturally relevant; never recite this list): "
            + memory
            + "."
        )
    if scenario:
        parts.append(
            f"Active role-play: {SCENARIOS[scenario]}. Stay in this scene until the learner changes it, "
            "but keep it playful and conversational rather than testing them."
        )
    return "\n".join(parts) + ("\n" if parts else "")


def extract_scores(analysis: str) -> dict[str, Optional[int]]:
    scores: dict[str, Optional[int]] = {}
    for key, label in (("overall", "Overall"), ("grammar", "Grammar"), ("vocabulary", "Vocabulary"), ("clarity", "Clarity")):
        match = re.search(rf"^{label}:\s*(\d{{1,3}})", analysis, re.IGNORECASE | re.MULTILINE)
        scores[key] = max(0, min(100, int(match.group(1)))) if match else None
    return scores


def save_session(user_id: int, study_language: str, transcript: str, analysis: str) -> int:
    scores = extract_scores(analysis)
    conn = sqlite3.connect(settings.db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (
                user_id, created_at, study_language, transcript, analysis,
                overall_score, grammar_score, vocabulary_score, clarity_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, datetime.utcnow().isoformat(), study_language, transcript, analysis,
                scores["overall"], scores["grammar"], scores["vocabulary"], scores["clarity"],
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def extract_new_words(analysis: str) -> list[tuple[str, str, str, str]]:
    if "NEW_WORDS:" not in analysis:
        return []
    raw = analysis.split("NEW_WORDS:", 1)[1]
    for marker in ("VOICE_REPLY:", "VOICE_REPLY_TRANSLATION:"):
        if "\n" + marker in raw:
            raw = raw.split("\n" + marker, 1)[0]
    words: list[tuple[str, str, str, str]] = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-• ").strip()
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        word = parts[0][:80]
        if len(parts) >= 4:
            transliteration, translation, example = parts[1][:120], parts[2][:160], parts[3][:240]
        else:
            transliteration, translation, example = "", parts[1][:160], parts[2][:240] if len(parts) > 2 else ""
        words.append((word, transliteration, translation, example))
    return words[:5]


def save_new_words(user_id: int, study_language: str, words: list[tuple[str, str, str, str]]) -> int:
    if not words:
        return 0
    conn = sqlite3.connect(settings.db_path)
    added = 0
    try:
        cursor = conn.cursor()
        for word, transliteration, translation, example in words:
            cursor.execute(
                """
                INSERT OR IGNORE INTO vocabulary_words
                (user_id, study_language, word, transliteration, translation, example, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, study_language, word.lower(), transliteration, translation, example, datetime.utcnow().isoformat()),
            )
            added += max(0, cursor.rowcount)
        conn.commit()
    finally:
        conn.close()
    return added


def award_activity(user_id: int, new_words_count: int) -> tuple[int, int, int, str]:
    today = datetime.now().date()
    today_key = today.isoformat()
    conn = sqlite3.connect(settings.db_path)
    try:
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT xp FROM daily_activity WHERE user_id = ? AND activity_date = ?", (user_id, today_key)
        ).fetchone()
        base_xp = 10 if row is None else 0
        gained = base_xp + new_words_count * 3
        cursor.execute(
            """
            INSERT INTO daily_activity (user_id, activity_date, xp, voice_count, words_learned)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(user_id, activity_date) DO UPDATE SET
                xp = daily_activity.xp + excluded.xp,
                voice_count = daily_activity.voice_count + 1,
                words_learned = daily_activity.words_learned + excluded.words_learned
            """,
            (user_id, today_key, gained, new_words_count),
        )
        days = [
            datetime.fromisoformat(row[0]).date()
            for row in cursor.execute(
                "SELECT activity_date FROM daily_activity WHERE user_id = ? ORDER BY activity_date DESC", (user_id,)
            ).fetchall()
        ]
        streak = 0
        expected = today
        for day in days:
            if day == expected:
                streak += 1
                expected = expected.fromordinal(expected.toordinal() - 1)
            elif day < expected:
                break
        total_xp = cursor.execute(
            "SELECT COALESCE(SUM(xp), 0) FROM daily_activity WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return gained, streak, total_xp, get_reward_title(total_xp)


def build_words_report(user_id: int, study_language: str) -> str:
    conn = sqlite3.connect(settings.db_path)
    try:
        rows = conn.execute(
            """
            SELECT word, transliteration, translation, example FROM vocabulary_words
            WHERE user_id = ? AND study_language = ? ORDER BY id DESC LIMIT 12
            """,
            (user_id, study_language),
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM vocabulary_words WHERE user_id = ? AND study_language = ?",
            (user_id, study_language),
        ).fetchone()[0]
    finally:
        conn.close()
    if not rows:
        return "Словарь пока пуст. Отправьте голосовое: после диалога бот предложит полезные новые слова."
    lines = [f"📚 Ваш словарь ({SUPPORTED_LANGUAGES[study_language]['label']}): {total} слов\n"]
    for word, transliteration, translation, example in rows:
        reading = f" ({transliteration})" if transliteration else ""
        lines.append(f"• {word}{reading} — {translation}" + (f"\n  {example}" if example else ""))
    lines.append("\nПовторите эти слова вслух и попробуйте использовать 2–3 из них в следующем голосовом.")
    return "\n".join(lines)


def build_vocabulary_context(user_id: int, study_language: str) -> str:
    conn = sqlite3.connect(settings.db_path)
    try:
        rows = conn.execute(
            """
            SELECT word, translation FROM vocabulary_words
            WHERE user_id = ? AND study_language = ? ORDER BY id DESC LIMIT 6
            """,
            (user_id, study_language),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return ""
    items = ", ".join(f"{word} ({translation})" for word, translation in rows)
    return (
        "Personal vocabulary to reinforce naturally when relevant: " + items + ". "
        "Use at most one of these items in VOICE_REPLY, with enough context for its meaning; never turn the "
        "conversation into a word list.\n"
    )


def build_rewards_report(user_id: int) -> str:
    conn = sqlite3.connect(settings.db_path)
    try:
        total_xp, active_days, words = conn.execute(
            """
            SELECT COALESCE(SUM(xp), 0), COUNT(*), COALESCE(SUM(words_learned), 0)
            FROM daily_activity WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        rows = conn.execute(
            "SELECT activity_date FROM daily_activity WHERE user_id = ? ORDER BY activity_date DESC", (user_id,)
        ).fetchall()
    finally:
        conn.close()
    streak = 0
    expected = datetime.now().date()
    for (date_text,) in rows:
        day = datetime.fromisoformat(date_text).date()
        if day == expected:
            streak += 1
            expected = expected.fromordinal(expected.toordinal() - 1)
        elif day < expected:
            break
    title = get_reward_title(int(total_xp))
    return (
        f"🏆 Ваш прогресс: {title}\n"
        f"✨ XP: {total_xp}\n🔥 Текущий стрик: {streak} дн.\n"
        f"🎙️ Дней с практикой: {active_days}\n📚 Новых слов: {words}\n\n"
        "Награды: +10 XP за первую голосовую практику дня и +3 XP за каждое новое слово."
    )


def get_reward_title(total_xp: int) -> str:
    title = "Новичок"
    if total_xp >= 1000:
        title = "Легенда языка"
    elif total_xp >= 500:
        title = "Уверенный собеседник"
    elif total_xp >= 200:
        title = "Искатель слов"
    elif total_xp >= 50:
        title = "Регулярный практик"
    return title


def build_progress_report(user_id: int, study_language: str) -> str:
    conn = sqlite3.connect(settings.db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT analysis, created_at, overall_score, grammar_score, vocabulary_score, clarity_score
            FROM sessions
            WHERE user_id = ? AND study_language = ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (user_id, study_language),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        return "No history yet for this language. Send at least one voice message to build your progress report."

    categories = {"Grammar": 0, "Vocabulary": 0, "Clarity": 0, "Pronunciation": 0}
    for analysis, *_ in rows:
        lowered = analysis.lower()
        if "grammar" in lowered:
            categories["Grammar"] += 1
        if "vocabulary" in lowered:
            categories["Vocabulary"] += 1
        if "clarity" in lowered:
            categories["Clarity"] += 1
        if "pronunciation" in lowered:
            categories["Pronunciation"] += 1

    scored = [row for row in rows if row[2] is not None]
    score_line = "Scores will appear after the next checked voice message."
    if scored:
        latest = scored[0]
        averages = []
        for index in range(2, 6):
            values = [row[index] for row in scored if row[index] is not None]
            averages.append(round(sum(values) / len(values)) if values else 0)
        first_batch = scored[-min(5, len(scored)):]
        baseline = sum(row[2] for row in first_batch if row[2] is not None) / len(first_batch)
        change = round(latest[2] - baseline) if latest[2] is not None else 0
        sign = "+" if change > 0 else ""
        score_line = (
            f"Latest overall score: {latest[2]}/100 ({sign}{change} vs. first sessions)\n"
            f"Average — grammar {averages[1]}, vocabulary {averages[2]}, clarity {averages[3]}"
        )
    unique_days = sorted({datetime.fromisoformat(row[1]).date() for row in rows}, reverse=True)
    streak = 0
    if unique_days:
        day = unique_days[0]
        for practiced_day in unique_days:
            if practiced_day == day:
                streak += 1
                day = day.fromordinal(day.toordinal() - 1)
            elif practiced_day < day:
                break
    top = sorted(categories.items(), key=lambda item: item[1], reverse=True)
    top_text = ", ".join(f"{name}: {count}" for name, count in top)
    return (
        f"Progress for {SUPPORTED_LANGUAGES[study_language]['label']}: {len(rows)} recent voice sessions\n"
        f"Practice streak: {streak} day(s)\n"
        f"{score_line}\n"
        f"Detected issue frequency: {top_text}\n"
        "Focus suggestion: work first on the top 1-2 categories."
    )


def _voice_reply_instructions(target_language: str) -> str:
    return (
        "VOICE_REPLY:\n"
  f"Write a single spoken monologue for TEXT-TO-SPEECH in {target_language} only.\n"
"You are a real conversation partner, not a tutor reciting a script. Read the learner's "
"message closely and react to what they actually said — the content, the mood, a specific "
"detail or word choice — before anything else. Never open with generic praise like 'Great job!' "
"or 'Well done!'; if something genuinely impressed or amused you, say what and why in one natural line.\n"
"If the learner's message contains a question or a request to be taught, explained, or helped with "
"something — explicit (a question word, question mark, or phrases like 'teach me', 'explain', 'help me "
"understand', 'how do I') or implied (describing confusion, asking to practice a specific thing, asking "
"for a rule or example) — your first priority is to actually address that request: teach it, explain it, "
"or answer it properly. Do not pivot to small talk or change the topic before doing this.\n"
"If it's a learning request about the language itself (grammar, translation, a word, pronunciation, a "
"rule, the difference between two constructions), actually teach it: give a clear explanation and at "
"least one concrete example, in a way that would genuinely help someone understand and remember it. "
"Don't just acknowledge the request or gesture at the topic — deliver the explanation itself.\n"
"If the question is personal or about the current topic, answer it directly and in your own words, "
"like a real conversational partner — don't deflect with vague phrases like 'that's an interesting "
"question.'\n"
"Only after the request has been properly addressed may you naturally react to the rest of the message "
"or add a follow-up. Do not ask your own question back, and do not change the subject, before you've "
"actually taught or answered what was asked.\n"
"Bad example: the learner says 'научи меня разнице между ser и estar' and the bot replies with "
"encouragement or a related anecdote instead of the actual explanation. Bad example: the learner asks "
"'how do you say \"I'll be late\" in Spanish?' and the bot replies 'By the way, how was your day?' "
"Good example: give the explanation or translation immediately and properly, with an example, and only "
"then continue the conversation if it flows naturally.\n"
"React like a person would: ask a real follow-up if you're curious, push back gently if something "
"is surprising or funny, laugh it off if it's silly. Your response should change shape depending on "
"what they said — a sad story gets a different reaction than a joke, which gets a different reaction "
"than a flat 'I don't know.'\n"
"Correct at most one mistake, and only if it doesn't interrupt the emotional thread — never stop a "
"personal story to fix grammar. Slip the correction in naturally, the way a friend would, then move on.\n"
"Read emotional tone and match it: slow down and soften if they sound tired, stressed, or upset; bring "
"energy up if they're excited or playful. If they give a short or guarded answer, don't interrogate them "
"with more questions — offer a small thought or observation of your own and pivot to something low-pressure "
"and everyday (food, travel, music, films, games, sport, technology, work).\n"
"You can invent a small, clearly hypothetical example or anecdote to make a point, but never claim real "
"personal experiences, memories, or feelings as fact — you're a conversational presence, not a person with "
"a biography.\n"
"Ask a follow-up question only when it actually moves the conversation forward — not as a default habit "
"at the end of every turn.\n"
"Light humor is welcome when it fits the moment, but don't force a joke into a serious exchange.\n"
"Write in flowing, natural spoken paragraphs. No bullet points, no headers, no canned sign-offs, no "
"formulaic 'Keep practicing!' type closers. End naturally, the way a real conversation trails into the "
"next moment.\n\n"
    )


def _voice_reply_translation_instructions(study_language: str) -> str:
    if study_language == "russian":
        return (
            "VOICE_REPLY_TRANSLATION:\n"
            "<faithful Turkish translation of the entire VOICE_REPLY monologue; plain paragraphs only>\n\n"
        )
    return (
        "VOICE_REPLY_TRANSLATION:\n"
        "<faithful Russian translation of the entire VOICE_REPLY monologue; plain paragraphs only>\n\n"
    )


def build_analysis_prompt(transcript: str, level: str, progress_summary: str, study_language: str) -> str:
    if study_language == "turkish":
        return _build_analysis_prompt_turkish(transcript, level, progress_summary)
    if study_language == "russian":
        return _build_analysis_prompt_russian(transcript, level, progress_summary)
    if study_language != "english":
        return _build_analysis_prompt_generic(
            transcript=transcript,
            level=level,
            progress_summary=progress_summary,
            language_name=SUPPORTED_LANGUAGES[study_language]["coach_language"],
            translation_hint=f"{SUPPORTED_LANGUAGES[study_language]['label']}→RU",
        )
    return _build_analysis_prompt_english(transcript, level, progress_summary)


def _build_analysis_prompt_generic(
    transcript: str, level: str, progress_summary: str, language_name: str, translation_hint: str
) -> str:
    return (
        f"You are a {language_name} pronunciation and speaking coach.\n"
        f"The user sent a spoken message in {language_name}. Analyze it as a teacher.\n\n"
        f"Student CEFR level: {level}\n"
        "Tailor feedback complexity to this level.\n"
        "Use simple explanations for A1-A2, more nuanced for B2-C2.\n\n"
        f"All textual sections and examples MUST be in {language_name}.\n\n"
        "Recent progress summary:\n"
        f"{progress_summary}\n\n"
        "Return strictly in this format:\n"
        "SCORE:\n"
        "Overall: <0-100>\n"
        "Grammar: <0-100>\n"
        "Vocabulary: <0-100>\n"
        "Clarity: <0-100>\n\n"
        "TRANSCRIPT:\n"
        "<cleaned transcript>\n\n"
        "MISTAKES:\n"
        "- Grammar: ...\n"
        "- Vocabulary: ...\n"
        "- Clarity: ...\n"
        "- Pronunciation guesses (if obvious from transcript only): ...\n\n"
        "MEMO:\n"
        "- Up to 5 short bullets with concrete fixes and examples.\n"
        "- Keep memo concise and practical.\n\n"
        "CORRECTED_VERSION:\n"
        "<one natural corrected variant of user text>\n\n"
        "ONE_MINUTE_HOMEWORK:\n"
        "<short speaking task for 1 minute adapted to level>\n\n"
        + _voice_reply_instructions(language_name)
        + _voice_reply_translation_instructions(study_language=language_name.lower())
        + "User transcript:\n"
        f"{transcript}\n\n"
        f"Translation context for VOICE_REPLY_TRANSLATION label: {translation_hint}."
    )


def _build_analysis_prompt_english(transcript: str, level: str, progress_summary: str) -> str:
    return (
        "You are an English pronunciation and speaking coach.\n"
        "The user sent a spoken message in English. Analyze it as a teacher.\n\n"
        f"Student CEFR level: {level}\n"
        "Tailor feedback complexity to this level.\n"
        "Use simple explanations for A1-A2, more nuanced for B2-C2.\n\n"
        "All textual sections and examples MUST be in English.\n\n"
        "Recent progress summary:\n"
        f"{progress_summary}\n\n"
        "Return strictly in this format:\n"
        "SCORE:\n"
        "Overall: <0-100>\n"
        "Grammar: <0-100>\n"
        "Vocabulary: <0-100>\n"
        "Clarity: <0-100>\n\n"
        "TRANSCRIPT:\n"
        "<cleaned transcript>\n\n"
        "MISTAKES:\n"
        "- Grammar: ...\n"
        "- Vocabulary: ...\n"
        "- Clarity: ...\n"
        "- Pronunciation guesses (if obvious from transcript only): ...\n\n"
        "MEMO:\n"
        "- Up to 5 short bullets with concrete fixes and examples.\n"
        "- Keep memo concise and practical.\n\n"
        "CORRECTED_VERSION:\n"
        "<one natural corrected variant of user text>\n\n"
        "ONE_MINUTE_HOMEWORK:\n"
        "<short speaking task for 1 minute adapted to level>\n\n"
        + _voice_reply_instructions("English")
        + _voice_reply_translation_instructions("english")
        + "User transcript:\n"
        f"{transcript}"
    )


def _build_analysis_prompt_turkish(transcript: str, level: str, progress_summary: str) -> str:
    return (
        "You are a Turkish pronunciation and speaking coach.\n"
        "The user sent a spoken message in Turkish. Analyze it as a teacher.\n\n"
        f"Student CEFR level: {level}\n"
        "Tailor feedback complexity to this level.\n"
        "Use simple explanations for A1-A2, more nuanced for B2-C2.\n\n"
        "Sections SCORE through ONE_MINUTE_HOMEWORK MUST be written in Turkish.\n"
        "TEXT_FEEDBACK_RU MUST be a faithful Russian translation of the Turkish feedback "
        "(scores summary, mistakes, memo, corrected version, homework) so the user can read it in Russian. "
        "Inside TEXT_FEEDBACK_RU, preserve the headings SCORE, TRANSCRIPT, MISTAKES, MEMO, "
        "CORRECTED_VERSION and ONE_MINUTE_HOMEWORK exactly.\n\n"
        "Recent progress summary:\n"
        f"{progress_summary}\n\n"
        "Return strictly in this format:\n"
        "SCORE:\n"
        "Overall: <0-100>\n"
        "Grammar: <0-100>\n"
        "Vocabulary: <0-100>\n"
        "Clarity: <0-100>\n\n"
        "TRANSCRIPT:\n"
        "<cleaned transcript>\n\n"
        "MISTAKES:\n"
        "- Grammar: ...\n"
        "- Vocabulary: ...\n"
        "- Clarity: ...\n"
        "- Pronunciation guesses (if obvious from transcript only): ...\n\n"
        "MEMO:\n"
        "- Up to 5 short bullets with concrete fixes and examples.\n"
        "- Keep memo concise and practical.\n\n"
        "CORRECTED_VERSION:\n"
        "<one natural corrected variant of user text>\n\n"
        "ONE_MINUTE_HOMEWORK:\n"
        "<short speaking task for 1 minute adapted to level>\n\n"
        "TEXT_FEEDBACK_RU:\n"
        "<full Russian translation of the Turkish feedback above>\n\n"
        + _voice_reply_instructions("Turkish")
        + _voice_reply_translation_instructions("turkish")
        + "User transcript:\n"
        f"{transcript}"
    )


def _build_analysis_prompt_russian(transcript: str, level: str, progress_summary: str) -> str:
    return (
        "You are a Russian pronunciation and speaking coach.\n"
        "The user sent a spoken message in Russian. Analyze it as a teacher.\n\n"
        f"Student CEFR level: {level}\n"
        "Tailor feedback complexity to this level.\n"
        "Use simple explanations for A1-A2, more nuanced for B2-C2.\n\n"
        "Sections SCORE through ONE_MINUTE_HOMEWORK MUST be written in Russian.\n"
        "TEXT_FEEDBACK_TR MUST be a faithful Turkish translation of the Russian feedback "
        "(scores summary, mistakes, memo, corrected version, homework).\n\n"
        "ACCENT_AND_ENDINGS: use the transcript and your corrections. "
        "Full audio-based accent diagnosis is not available; infer likely issues from word forms, "
        "missing letters, wrong endings, and typical L2 patterns.\n"
        "Include:\n"
        "- Word stress (ударение): list 5-12 words where stress matters; give correct spelling with stress mark "
        "using combining acute on the stressed vowel (e.g. здра\u0301вствуйте) or explicit description.\n"
        "- Word endings: noun cases, adjective agreement, verb tense/person/aspect; "
        "note wrong endings in the transcript and show correct forms.\n\n"
        "Recent progress summary:\n"
        f"{progress_summary}\n\n"
        "Return strictly in this format:\n"
        "SCORE:\n"
        "Overall: <0-100>\n"
        "Grammar: <0-100>\n"
        "Vocabulary: <0-100>\n"
        "Clarity: <0-100>\n\n"
        "TRANSCRIPT:\n"
        "<cleaned transcript>\n\n"
        "MISTAKES:\n"
        "- Grammar: ...\n"
        "- Vocabulary: ...\n"
        "- Clarity: ...\n"
        "- Pronunciation guesses (if obvious from transcript only): ...\n\n"
        "MEMO:\n"
        "- Up to 5 short bullets with concrete fixes and examples.\n"
        "- Keep memo concise and practical.\n\n"
        "CORRECTED_VERSION:\n"
        "<one natural corrected variant of user text>\n\n"
        "ONE_MINUTE_HOMEWORK:\n"
        "<short speaking task for 1 minute adapted to level>\n\n"
        "TEXT_FEEDBACK_TR:\n"
        "<full Turkish translation of the Russian feedback above>\n\n"
        "ACCENT_AND_ENDINGS:\n"
        "- Word stress: ...\n"
        "- Endings and morphology: ...\n\n"
        + _voice_reply_instructions("Russian")
        + _voice_reply_translation_instructions("russian")
        + "User transcript:\n"
        f"{transcript}"
    )


async def transcribe_voice(openai_client: AsyncOpenAI, audio_bytes: bytes) -> str:
    file_like = io.BytesIO(audio_bytes)
    file_like.name = "voice.ogg"
    result = await openai_client.audio.transcriptions.create(
        model=settings.transcribe_model,
        file=file_like,
    )
    return result.text.strip()


async def analyze_transcript(
    openai_client: AsyncOpenAI,
    transcript: str,
    level: str,
    progress_summary: str,
    study_language: str,
    dialogue_context: str,
    user_voice_seconds: int,
) -> str:
    target_seconds = max(5, min(180, user_voice_seconds))
    latin_reading = SUPPORTED_LANGUAGES[study_language].get("latin_reading", "")
    latin_reading_instruction = ""
    if latin_reading:
        latin_reading_instruction = (
            f"This language uses a non-Latin/non-Cyrillic script. In every written learner-facing section "
            f"(TRANSCRIPT, MISTAKES, MEMO, CORRECTED_VERSION and ONE_MINUTE_HOMEWORK), put a {latin_reading} "
            "in parentheses immediately after each target-language word, phrase or example. In TEXT_FEEDBACK_RU, "
            "also preserve the original expression followed by its Latin reading. Do NOT add readings inside "
            "VOICE_REPLY, because it is sent to speech synthesis.\n"
        )
    russian_feedback_instruction = ""
    if study_language not in ("russian", "turkish"):
        russian_feedback_instruction = (
            "Before MEMORY_UPDATE, NEW_WORDS and VOICE_REPLY, add this exact section:\n"
            "TEXT_FEEDBACK_RU:\n"
            "<a faithful, natural Russian translation of SCORE, TRANSCRIPT, MISTAKES, MEMO, "
            "CORRECTED_VERSION and ONE_MINUTE_HOMEWORK; preserve these section headings so each part can "
            "be opened separately in the app>\n"
        )
    response = await openai_client.responses.create(
        model=settings.analysis_model,
        input=(
            build_analysis_prompt(transcript, level, progress_summary, study_language)
            + "\n\n"
            + dialogue_context
            + russian_feedback_instruction
            + latin_reading_instruction
            + (
                "Voice reply timing: the learner's voice message lasted about "
                f"{target_seconds} seconds. Make VOICE_REPLY take approximately the same amount of time "
                "when spoken aloud (aim for 80–110% of that duration). Match its pace and substance; do not "
                "pad a short message or compress a substantial one.\n"
            )
            + (
                "At the end, before NEW_WORDS, add this exact section:\n"
                "MEMORY_UPDATE:\n"
                "- <city|work|hobbies|favorite_team|music|food|travel|language_goal> | <fact explicitly stated by the learner>\n"
                "Add 0–3 stable, non-sensitive details only when the learner clearly states or corrects them. "
                "Never infer facts, store temporary mood, health, politics, relationship identities, contact "
                "details, or anything the learner did not volunteer.\n"
            )
            + (
                "At the end, before VOICE_REPLY, add this exact section:\n"
                "NEW_WORDS:\n"
                + (
                    "- <word or phrase> | <Latin transcription> | <short Russian translation> | <short natural example>\n"
                    if latin_reading else
                    "- <useful word or phrase from the dialogue> | <short Russian translation> | <short natural example>\n"
                )
                + "Choose 0–3 genuinely useful, level-appropriate items that the learner did not already use "
                "confidently. The translation or explanation must be in Russian, so the learner can use it "
                "immediately. Do not force words when none are useful.\n"
            )
        ),
    )
    return response.output_text.strip()


def extract_voice_reply(analysis_text: str, study_language: str) -> str:
    marker = "VOICE_REPLY:"
    if marker not in analysis_text:
        return SUPPORTED_LANGUAGES[study_language]["fallback_reply"]
    raw = analysis_text.split(marker, maxsplit=1)[1].strip()
    for stop in (
        "\nVOICE_REPLY_TRANSLATION:",
        "\nUser transcript:",
        "\nTRANSCRIPT:",
        "\nSCORE:",
        "\nMISTAKES:",
        "\nMEMO:",
        "\nTEXT_FEEDBACK_TR:",
        "\nTEXT_FEEDBACK_RU:",
        "\nACCENT_AND_ENDINGS:",
        "\nNEW_WORDS:",
        "\nMEMORY_UPDATE:",
    ):
        if stop in raw:
            raw = raw.split(stop, 1)[0].strip()
    return raw or SUPPORTED_LANGUAGES[study_language]["fallback_reply"]


def extract_voice_reply_translation(analysis_text: str) -> str:
    marker = "VOICE_REPLY_TRANSLATION:"
    if marker not in analysis_text:
        return ""
    raw = analysis_text.split(marker, maxsplit=1)[1].strip()
    for stop in (
        "\nUser transcript:",
        "\nTRANSCRIPT:",
        "\nSCORE:",
        "\nVOICE_REPLY:",
        "\nNEW_WORDS:",
        "\nMEMORY_UPDATE:",
    ):
        if stop in raw:
            raw = raw.split(stop, 1)[0].strip()
    return raw


def voice_reply_translation_header(study_language: str) -> str:
    if study_language == "russian":
        return "🇹🇷 Перевод голосового ответа (RU→TR)\n\n"
    if study_language == "turkish":
        return "🇷🇺 Перевод голосового ответа (TR→RU)\n\n"
    label = SUPPORTED_LANGUAGES[study_language]["label"]
    return f"🇷🇺 Перевод голосового ответа ({label}→RU)\n\n"


async def translate_voice_reply_fallback(
    client: AsyncOpenAI, text: str, study_language: str
) -> str:
    if study_language == "russian":
        instruction = (
            "Translate the following Russian text into natural Turkish. "
            "Output only the translation, no headings or quotes."
        )
    else:
        instruction = (
            "Translate the following text into natural Russian. "
            "Output only the translation, no headings or quotes."
        )
    response = await client.responses.create(
        model=settings.analysis_model,
        input=f"{instruction}\n\n{text}",
    )
    return response.output_text.strip()


def build_text_qa_prompt(question: str, study_language: str, level: str) -> str:
    if study_language == "turkish":
        return (
            "You are an experienced Turkish language teacher.\n"
            f"The student is learning Turkish at about CEFR {level}.\n"
            "They may write their question in English, Russian, or Turkish.\n\n"
            "Rules:\n"
            "- Detect the language of the question. Answer mainly in that language "
            "(English or Russian if the question is in English or Russian; Turkish if they wrote in Turkish).\n"
            "- You may add short Turkish examples with glosses or translations.\n"
            "- Focus on Turkish: grammar, vocabulary, usage, common mistakes, study tips.\n"
            "- Be clear and structured; use short examples. No meta-commentary about being an AI.\n"
            "- If the question is off-topic, answer briefly and gently steer back to Turkish learning.\n\n"
            f"Question:\n{question}"
        )
    if study_language == "english":
        return (
            "You are a language tutor.\n"
            f"The student's main focus is English (about CEFR {level}), but they want to ask about Turkish: "
            "words, grammar, or how the language works.\n\n"
            "Answer in English. Give accurate explanations and Turkish examples with glosses or translations "
            "where helpful. Be concise.\n\n"
            f"Question:\n{question}"
        )
    if study_language != "russian":
        return (
        "You are a language tutor.\n"
        f"The student is learning {SUPPORTED_LANGUAGES[study_language]['label']} (about CEFR {level}).\n\n"
        f"Answer in {SUPPORTED_LANGUAGES[study_language]['coach_language']}. "
        "Give accurate explanations, short examples, and practical tips for speaking progress. "
        "Be concise and clear.\n\n"
        f"Question:\n{question}"
        )
    return (
        "You are a language tutor.\n"
        f"The student's main focus is Russian (about CEFR {level}), but they want to ask about Turkish: "
        "words, grammar, or how the language works.\n\n"
        "Answer in Russian. Give accurate explanations and Turkish examples with glosses or translations "
        "where helpful. Be concise.\n\n"
        f"Question:\n{question}"
    )


async def ask_tutor_text(
    client: AsyncOpenAI, question: str, study_language: str, level: str
) -> str:
    response = await client.responses.create(
        model=settings.analysis_model,
        input=build_text_qa_prompt(question, study_language, level),
    )
    return response.output_text.strip()


def split_text_for_telegram(text: str, max_len: int = 4000) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    rest = text
    while len(rest) > max_len:
        window = rest[:max_len]
        cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
        if cut < max_len // 4:
            cut = max_len
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return [p for p in parts if p]


RESULT_SECTIONS = {
    "summary": ("📊 Summary", ("SCORE:", "TRANSCRIPT:")),
    "mistakes": ("🔎 Mistakes", ("MISTAKES:", "MEMO:")),
    "corrected": ("✍️ Corrected version", ("CORRECTED_VERSION:",)),
    "homework": ("🎯 1-minute task", ("ONE_MINUTE_HOMEWORK:",)),
    "words": ("📚 New words", ("NEW_WORDS:",)),
    "transcript": ("📝 Transcript", ("TRANSCRIPT:",)),
}
SECTION_MARKERS = (
    "SCORE:", "TRANSCRIPT:", "MISTAKES:", "MEMO:", "CORRECTED_VERSION:",
    "ONE_MINUTE_HOMEWORK:", "TEXT_FEEDBACK_RU:", "TEXT_FEEDBACK_TR:",
    "ACCENT_AND_ENDINGS:", "MEMORY_UPDATE:", "NEW_WORDS:", "VOICE_REPLY:", "VOICE_REPLY_TRANSLATION:",
)


def russian_feedback_source(analysis: str) -> str:
    if "TEXT_FEEDBACK_RU:" not in analysis:
        return ""
    source = analysis.split("TEXT_FEEDBACK_RU:", 1)[1]
    for marker in ("\nMEMORY_UPDATE:", "\nNEW_WORDS:", "\nVOICE_REPLY:", "\nVOICE_REPLY_TRANSLATION:"):
        if marker in source:
            source = source.split(marker, 1)[0]
    return source.strip()


def get_analysis_section(analysis: str, section: str, russian: bool = False) -> str:
    if russian and section != "words":
        translated = russian_feedback_source(analysis)
        if translated:
            analysis = translated
    title, markers = RESULT_SECTIONS.get(section, RESULT_SECTIONS["summary"])
    pieces: list[str] = []
    for marker in markers:
        if marker not in analysis:
            continue
        part = analysis.split(marker, 1)[1]
        stops = [part.find("\n" + candidate) for candidate in SECTION_MARKERS if candidate != marker]
        stops = [stop for stop in stops if stop >= 0]
        if stops:
            part = part[:min(stops)]
        pieces.append(part.strip())
    body = "\n\n".join(piece for piece in pieces if piece)
    if russian:
        title = {
            "📊 Summary": "📊 Обзор", "🔎 Mistakes": "🔎 Ошибки", "✍️ Corrected version": "✍️ Исправленный вариант",
            "🎯 1-minute task": "🎯 Задание на минуту", "📚 New words": "📚 Новые слова", "📝 Transcript": "📝 Текст",
        }.get(title, title)
    return f"{title}\n\n{body or 'Этот раздел пока не переведён. Откройте оригинал или отправьте следующее голосовое.'}"


def result_keyboard(session_id: int, active: str, russian: bool = False, has_russian: bool = False) -> InlineKeyboardMarkup:
    labels = [
        ("summary", "📊 Overview"), ("mistakes", "🔎 Mistakes"), ("corrected", "✍️ Version"),
        ("homework", "🎯 Task"), ("words", "📚 Words"), ("transcript", "📝 Text"),
    ]
    buttons = [
        InlineKeyboardButton(
            text=("• " if key == active else "") + label,
            callback_data=f"result:{session_id}:{key}:{'ru' if russian else 'orig'}",
        )
        for key, label in labels
    ]
    rows = [buttons[:2], buttons[2:4], buttons[4:]]
    if has_russian:
        rows.append([
            InlineKeyboardButton(
                text="🌐 Оригинал" if russian else "🇷🇺 На русском",
                callback_data=f"result:{session_id}:{active}:{'orig' if russian else 'ru'}",
            )
        ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def get_session_analysis(session_id: int, user_id: int) -> Optional[str]:
    conn = sqlite3.connect(settings.db_path)
    try:
        row = conn.execute("SELECT analysis FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


async def send_feedback_card(message: Message, session_id: int, analysis: str) -> None:
    await message.answer(
        get_analysis_section(analysis, "summary")[:4096],
        reply_markup=result_keyboard(session_id, "summary", has_russian="TEXT_FEEDBACK_RU:" in analysis),
    )


async def send_feedback_messages(message: Message, analysis: str, study_language: str) -> None:
    if study_language in ("english", "serbian", "ukrainian", "tatar", "chechen", "nogai"):
        body = analysis.split("VOICE_REPLY:", 1)[0].strip() if "VOICE_REPLY:" in analysis else analysis
        for chunk in split_text_for_telegram(body):
            await message.answer(chunk)
        return

    if study_language == "turkish":
        main = analysis
        if "TEXT_FEEDBACK_RU:" in analysis:
            main = analysis.split("TEXT_FEEDBACK_RU:", 1)[0].strip()
        elif "VOICE_REPLY:" in analysis:
            main = analysis.split("VOICE_REPLY:", 1)[0].strip()
        if "VOICE_REPLY:" in main:
            main = main.split("VOICE_REPLY:", 1)[0].strip()
        ru_text = ""
        if "TEXT_FEEDBACK_RU:" in analysis:
            tail = analysis.split("TEXT_FEEDBACK_RU:", 1)[1]
            if "VOICE_REPLY:" in tail:
                tail = tail.split("VOICE_REPLY:", 1)[0]
            ru_text = tail.strip()
        for chunk in split_text_for_telegram(main):
            await message.answer("🇹🇷 Türkçe\n\n" + chunk)
        if ru_text:
            for chunk in split_text_for_telegram(ru_text):
                await message.answer("🇷🇺 Русский\n\n" + chunk)
        if not main and not ru_text:
            for chunk in split_text_for_telegram(analysis.split("VOICE_REPLY:", 1)[0].strip()):
                await message.answer(chunk)
        return

    if study_language == "russian":
        main = analysis
        if "TEXT_FEEDBACK_TR:" in analysis:
            main = analysis.split("TEXT_FEEDBACK_TR:", 1)[0].strip()
        elif "ACCENT_AND_ENDINGS:" in analysis:
            main = analysis.split("ACCENT_AND_ENDINGS:", 1)[0].strip()
        elif "VOICE_REPLY:" in analysis:
            main = analysis.split("VOICE_REPLY:", 1)[0].strip()
        if "VOICE_REPLY:" in main:
            main = main.split("VOICE_REPLY:", 1)[0].strip()

        tr_text = ""
        if "TEXT_FEEDBACK_TR:" in analysis:
            tail = analysis.split("TEXT_FEEDBACK_TR:", 1)[1]
            if "ACCENT_AND_ENDINGS:" in tail:
                tail = tail.split("ACCENT_AND_ENDINGS:", 1)[0]
            elif "VOICE_REPLY:" in tail:
                tail = tail.split("VOICE_REPLY:", 1)[0]
            tr_text = tail.strip()

        accent = ""
        if "ACCENT_AND_ENDINGS:" in analysis:
            tail = analysis.split("ACCENT_AND_ENDINGS:", 1)[1]
            if "VOICE_REPLY:" in tail:
                tail = tail.split("VOICE_REPLY:", 1)[0]
            accent = tail.strip()

        for chunk in split_text_for_telegram(main):
            await message.answer("🇷🇺 Русский\n\n" + chunk)
        if tr_text:
            for chunk in split_text_for_telegram(tr_text):
                await message.answer("🇹🇷 Türkçe\n\n" + chunk)
        if accent:
            for chunk in split_text_for_telegram(accent):
                await message.answer("📌 Ударение и окончания\n\n" + chunk)
        if not main and not tr_text and not accent:
            for chunk in split_text_for_telegram(analysis.split("VOICE_REPLY:", 1)[0].strip()):
                await message.answer(chunk)
        return

    for chunk in split_text_for_telegram(analysis):
        await message.answer(chunk)


def split_text_for_tts(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    rest = text
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max(
            window.rfind("\n\n"),
            window.rfind(". "),
            window.rfind("! "),
            window.rfind("? "),
            window.rfind("… "),
            window.rfind(" "),
        )
        if cut < max_chars // 3:
            cut = max_chars
        piece = rest[:cut].strip()
        rest = rest[cut:].strip()
        if piece:
            chunks.append(piece)
    if rest:
        chunks.append(rest)
    return [c for c in chunks if c]


async def synthesize_voice_chunk(openai_client: AsyncOpenAI, text: str) -> bytes:
    try:
        speech_response = await openai_client.audio.speech.create(
            model=settings.tts_model,
            voice=settings.tts_voice,
            response_format="opus",
            input=text,
            speed=settings.tts_speed,
        )
    except TypeError:
        speech_response = await openai_client.audio.speech.create(
            model=settings.tts_model,
            voice=settings.tts_voice,
            response_format="opus",
            input=text,
        )
    return speech_response.read()


async def synthesize_voice_segments(openai_client: AsyncOpenAI, text: str) -> list[bytes]:
    pieces = split_text_for_tts(text, settings.tts_max_chars_per_chunk)
    return [await synthesize_voice_chunk(openai_client, p) for p in pieces]


settings = load_settings()
openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
bot = Bot(token=settings.telegram_token)
dp = Dispatcher()


MENU_PROGRESS = "📊 Прогресс"
MENU_WORDS = "📚 Мои слова"
MENU_REWARDS = "🏆 Награды"
MENU_LEVEL = "🎚 Уровень"
MENU_LANGUAGE = "🌐 Язык"
MENU_SCENARIO = "🎭 Сцена"


def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=MENU_PROGRESS), KeyboardButton(text=MENU_WORDS)],
            [KeyboardButton(text=MENU_REWARDS), KeyboardButton(text=MENU_SCENARIO)],
            [KeyboardButton(text=MENU_LEVEL), KeyboardButton(text=MENU_LANGUAGE)],
        ],
        resize_keyboard=True,
        input_field_placeholder="Запишите голосовое или выберите действие…",
    )


def level_keyboard() -> InlineKeyboardMarkup:
    levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    buttons = [InlineKeyboardButton(text=level, callback_data=f"settings:level:{level}") for level in levels]
    return InlineKeyboardMarkup(inline_keyboard=[buttons[:3], buttons[3:]])


LANGUAGES_PER_PAGE = 8


def language_keyboard(page: int = 0) -> InlineKeyboardMarkup:
    all_buttons = [
        InlineKeyboardButton(text=data["label"], callback_data=f"settings:lang:{key}")
        for key, data in SUPPORTED_LANGUAGES.items()
    ]
    page_count = max(1, (len(all_buttons) + LANGUAGES_PER_PAGE - 1) // LANGUAGES_PER_PAGE)
    page = max(0, min(page, page_count - 1))
    buttons = all_buttons[page * LANGUAGES_PER_PAGE:(page + 1) * LANGUAGES_PER_PAGE]
    rows = [buttons[index:index + 2] for index in range(0, len(buttons), 2)]
    if page_count > 1:
        navigation = []
        if page > 0:
            navigation.append(InlineKeyboardButton(text="◀️", callback_data=f"settings:langpage:{page - 1}"))
        navigation.append(InlineKeyboardButton(text=f"{page + 1}/{page_count}", callback_data="settings:noop:0"))
        if page < page_count - 1:
            navigation.append(InlineKeyboardButton(text="▶️", callback_data=f"settings:langpage:{page + 1}"))
        rows.append(navigation)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def scenario_keyboard() -> InlineKeyboardMarkup:
    labels = {
        "restaurant": "🍽 Ресторан", "airport": "✈️ Аэропорт", "interview": "💼 Собеседование",
        "shopping": "🛍 Покупки", "football": "⚽ Футбол", "date": "💬 Знакомство", "off": "Обычный разговор",
    }
    buttons = [InlineKeyboardButton(text=label, callback_data=f"settings:scenario:{key}") for key, label in labels.items()]
    return InlineKeyboardMarkup(inline_keyboard=[buttons[:2], buttons[2:4], buttons[4:6], buttons[6:]])


@dp.callback_query(F.data.startswith("result:"))
async def result_navigation_handler(callback: CallbackQuery) -> None:
    if not callback.data or not callback.from_user or not callback.message:
        return
    parts = callback.data.split(":")
    if len(parts) not in (3, 4) or not parts[1].isdigit() or parts[2] not in RESULT_SECTIONS:
        await callback.answer("This result is unavailable.", show_alert=True)
        return
    analysis = get_session_analysis(int(parts[1]), callback.from_user.id)
    if not analysis:
        await callback.answer("This result is no longer available.", show_alert=True)
        return
    section = parts[2]
    russian = len(parts) == 4 and parts[3] == "ru"
    text = get_analysis_section(analysis, section, russian=russian)[:4096]
    try:
        await callback.message.edit_text(
            text,
            reply_markup=result_keyboard(
                int(parts[1]), section, russian=russian, has_russian="TEXT_FEEDBACK_RU:" in analysis
            ),
        )
    except Exception:
        pass  # The user may press the already selected button.
    await callback.answer()


@dp.callback_query(F.data.startswith("settings:"))
async def settings_navigation_handler(callback: CallbackQuery) -> None:
    if not callback.data or not callback.from_user or not callback.message:
        return
    parts = callback.data.split(":")
    if len(parts) != 3:
        await callback.answer("Не удалось применить настройку.", show_alert=True)
        return
    _, kind, value = parts
    user_id = callback.from_user.id
    if kind == "level" and value in {"A1", "A2", "B1", "B2", "C1", "C2"}:
        set_user_level(user_id, value)
        await callback.message.edit_text(f"🎚 Уровень установлен: {value}. Следующие ответы подстроятся под него.")
    elif kind == "lang" and value in SUPPORTED_LANGUAGES:
        set_user_study_language(user_id, value)
        await callback.message.edit_text(f"🌐 Язык обучения: {SUPPORTED_LANGUAGES[value]['label']}.")
    elif kind == "langpage" and value.isdigit():
        await callback.message.edit_text("Выберите язык для практики:", reply_markup=language_keyboard(int(value)))
    elif kind == "noop":
        await callback.answer()
        return
    elif kind == "scenario" and (value in SCENARIOS or value == "off"):
        get_user_study_language(user_id)
        set_user_scenario(user_id, "" if value == "off" else value)
        text = "🎭 Обычный разговор включён." if value == "off" else f"🎭 Сцена: {SCENARIOS[value]}."
        await callback.message.edit_text(text + " Пришлите голосовое — начнём.")
    else:
        await callback.answer("Неизвестная настройка.", show_alert=True)
        return
    await callback.answer("Готово")


@dp.message(CommandStart())
async def start_handler(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    level = get_user_level(user_id)
    study_language = get_user_study_language(user_id)
    language_label = SUPPORTED_LANGUAGES[study_language]["label"]
    set_intro_waiting(user_id, True)
    await message.answer(
        "Привет! Я твой разговорный напарник и тренер. Как мне к тебе обращаться?\n\n"
        + f"Текущий уровень: {level}\n"
        + f"Язык: {language_label}\n\n"
        + INSTRUCTION_TEXT,
        reply_markup=main_menu(),
    )


@dp.message(F.text.regexp(r"^/setlevel\s+(A1|A2|B1|B2|C1|C2)$"))
async def set_level_handler(message: Message) -> None:
    if not message.text:
        return
    match = re.match(r"^/setlevel\s+(A1|A2|B1|B2|C1|C2)$", message.text.strip(), re.IGNORECASE)
    if not match:
        await message.answer("Use format: /setlevel A2")
        return

    level = match.group(1).upper()
    user_id = message.from_user.id if message.from_user else 0
    set_user_level(user_id, level)
    await message.answer(f"Level updated to {level}. Next feedback will be adapted.")


@dp.message(F.text.startswith("/setlang"))
async def set_language_handler(message: Message) -> None:
    if not message.text:
        return
    parts = message.text.strip().split(maxsplit=1)
    language = parts[1].lower() if len(parts) == 2 else ""
    if language not in SUPPORTED_LANGUAGES:
        await message.answer(
            "Используйте /setlang <язык> или кнопку «🌐 Язык» в меню."
        )
        return

    user_id = message.from_user.id if message.from_user else 0
    set_user_study_language(user_id, language)
    language_label = SUPPORTED_LANGUAGES[language]["label"]
    await message.answer(f"Study language updated to {language_label}. Next feedback will use this language.")


@dp.message(F.text == "/progress")
async def progress_handler(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    study_language = get_user_study_language(user_id)
    report = build_progress_report(user_id, study_language)
    await message.answer(report)


@dp.message(F.text == "/words")
async def words_handler(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    study_language = get_user_study_language(user_id)
    await message.answer(build_words_report(user_id, study_language))


@dp.message(F.text == "/rewards")
async def rewards_handler(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    await message.answer(build_rewards_report(user_id))


@dp.message(F.text.regexp(r"^/scenario\s+(restaurant|airport|interview|shopping|football|date|off)$"))
async def scenario_handler(message: Message) -> None:
    if not message.text:
        return
    scenario = message.text.split(maxsplit=1)[1].strip().lower()
    user_id = message.from_user.id if message.from_user else 0
    get_user_study_language(user_id)  # Ensures the profile exists for a brand-new user.
    set_user_scenario(user_id, "" if scenario == "off" else scenario)
    if scenario == "off":
        await message.answer("Режим сцен выключен. Продолжим обычный живой разговор.")
    else:
        await message.answer(f"Сцена включена: {SCENARIOS[scenario]}. Пришлите голосовое — начнём.")


@dp.message(F.text == MENU_PROGRESS)
async def menu_progress_handler(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    await message.answer(build_progress_report(user_id, get_user_study_language(user_id)))


@dp.message(F.text == MENU_WORDS)
async def menu_words_handler(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    await message.answer(build_words_report(user_id, get_user_study_language(user_id)))


@dp.message(F.text == MENU_REWARDS)
async def menu_rewards_handler(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    await message.answer(build_rewards_report(user_id))


@dp.message(F.text == MENU_LEVEL)
async def menu_level_handler(message: Message) -> None:
    await message.answer("Выберите текущий уровень:", reply_markup=level_keyboard())


@dp.message(F.text == MENU_LANGUAGE)
async def menu_language_handler(message: Message) -> None:
    await message.answer("Выберите язык для практики:", reply_markup=language_keyboard())


@dp.message(F.text == MENU_SCENARIO)
async def menu_scenario_handler(message: Message) -> None:
    await message.answer("Выберите жизненную сцену или вернитесь к обычному разговору:", reply_markup=scenario_keyboard())


@dp.message(F.text, ~F.text.startswith("/"))
async def text_question_handler(message: Message) -> None:
    raw = (message.text or "").strip()
    if not raw:
        return

    user_id = message.from_user.id if message.from_user else 0
    conn = sqlite3.connect(settings.db_path)
    try:
        row = conn.execute("SELECT awaiting_name FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    finally:
        conn.close()
    if row and row[0]:
        name = re.sub(r"\s+", " ", raw).strip()
        if len(name) <= 60:
            save_display_name(user_id, name)
            await message.answer(
                f"Очень приятно, {name}! Пришли первое голосовое — я начну знакомство на выбранном языке, "
                "подхвачу твою тему и буду помнить ход разговора."
            )
            return
    level = get_user_level(user_id)
    study_language = get_user_study_language(user_id)
    status = await message.answer("⌛ …")
    try:
        answer = await ask_tutor_text(openai_client, raw, study_language, level)
        await status.delete()
        if not answer:
            await message.answer("No answer generated. Try rephrasing your question.")
            return
        for chunk in split_text_for_telegram(answer):
            await message.answer(chunk)
    except Exception as exc:
        logging.exception("Text Q&A error")
        try:
            await status.edit_text(f"Error: {exc}")
        except Exception:
            await message.answer(f"Error: {exc}")
 

@dp.message(F.voice)
async def voice_handler(message: Message) -> None:
    if not message.voice:
        await message.answer("I did not get a voice file. Please send a voice message.")
        return

    progress = await message.answer("Processing your voice... Please wait.")

    voice_paths: list[str] = []
    try:
        file = await bot.get_file(message.voice.file_id)
        voice_bytes = await bot.download_file(file.file_path)
        audio_bytes = voice_bytes.read()

        transcript = await transcribe_voice(openai_client, audio_bytes)
        if not transcript:
            await progress.edit_text("I could not recognize speech. Please try again more clearly.")
            return

        user_id = message.from_user.id if message.from_user else 0
        level = get_user_level(user_id)
        study_language = get_user_study_language(user_id)
        progress_summary = build_progress_report(user_id, study_language)
        name, recent_turns = get_dialogue_context(user_id, study_language)
        scenario = get_user_scenario(user_id)
        analysis = await analyze_transcript(
            openai_client,
            transcript,
            level,
            progress_summary,
            study_language,
            (
                build_dialogue_context(name, recent_turns)
                + build_vocabulary_context(user_id, study_language)
                + build_personal_memory_context(user_id, scenario)
            ),
            message.voice.duration,
        )
        voice_reply_text = extract_voice_reply(analysis, study_language)
        session_id = save_session(user_id, study_language, transcript, analysis)
        save_memory_updates(user_id, extract_memory_updates(analysis))
        added_words = save_new_words(user_id, study_language, extract_new_words(analysis))
        gained_xp, streak, total_xp, title = award_activity(user_id, added_words)

        segments = await synthesize_voice_segments(openai_client, voice_reply_text)
        for seg in segments:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
                tmp.write(seg)
                voice_paths.append(tmp.name)

        reward = f" + {added_words} слов" if added_words else ""
        await progress.edit_text(
            f"Готово! +{gained_xp} XP{reward} · 🔥 {streak} дн. · {title} ({total_xp} XP)"
        )
        await send_feedback_card(message, session_id, analysis)
        lang_label = SUPPORTED_LANGUAGES[study_language]["label"]
        total = len(voice_paths)
        for idx, vp in enumerate(voice_paths):
            cap = (
                f"Voice {idx + 1}/{total} — {lang_label}"
                if total > 1
                else f"Voice reply in {lang_label}"
            )
            await message.answer_voice(voice=FSInputFile(vp), caption=cap)

        voice_translation = extract_voice_reply_translation(analysis)
        if not voice_translation and voice_reply_text:
            try:
                voice_translation = await translate_voice_reply_fallback(
                    openai_client, voice_reply_text, study_language
                )
            except Exception:
                logging.exception("Voice reply translation fallback failed")
        if voice_translation:
            header = voice_reply_translation_header(study_language)
            for idx, chunk in enumerate(split_text_for_telegram(voice_translation)):
                await message.answer((header if idx == 0 else "") + chunk)
    except Exception as exc:
        logging.exception("Error while processing voice")
        await progress.edit_text(f"Error: {exc}")
    finally:
        for vp in voice_paths:
            if vp and os.path.exists(vp):
                os.remove(vp)


@dp.message()
async def fallback_handler(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    study_language = get_user_study_language(user_id)
    language_label = SUPPORTED_LANGUAGES[study_language]["label"]
    await message.answer(
        f"Please send a *voice message* in {language_label}.\n\n" + INSTRUCTION_TEXT,
        parse_mode="Markdown",
        reply_markup=main_menu(),
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    init_db(settings.db_path)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
