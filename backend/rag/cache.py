"""
Translation Cache — SQLite-based caching for translation results.

Every unique (text, source_lang, target_lang) combination is stored
so repeated requests are served instantly from local disk, bypassing
the Gemini API entirely.
"""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "translation_cache.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS translation_cache (
            cache_key   TEXT PRIMARY KEY,
            text        TEXT NOT NULL,
            source_lang TEXT NOT NULL,
            target_lang TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            hit_count   INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def _make_key(text: str, source_lang: str, target_lang: str) -> str:
    raw = f"{text.strip().lower()}|{source_lang.lower()}|{target_lang.lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_get(text: str, source_lang: str, target_lang: str) -> dict | None:
    """Return cached translation dict, or None if not found."""
    key = _make_key(text, source_lang, target_lang)
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT result_json FROM translation_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE translation_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                (key,),
            )
            conn.commit()
            conn.close()
            result = json.loads(row["result_json"])
            result["from_cache"] = True
            logger.info("Cache HIT for: '%s'", text[:50])
            return result
        conn.close()
    except Exception as exc:
        logger.warning("Cache read error: %s", exc)
    return None


def cache_set(text: str, source_lang: str, target_lang: str, result: dict) -> None:
    """Store a translation result in the cache."""
    key = _make_key(text, source_lang, target_lang)
    # Don't cache errors
    if result.get("error"):
        return
    try:
        conn = _get_conn()
        serializable = {k: v for k, v in result.items() if k != "from_cache"}
        conn.execute(
            """INSERT OR REPLACE INTO translation_cache
               (cache_key, text, source_lang, target_lang, result_json)
               VALUES (?, ?, ?, ?, ?)""",
            (key, text, source_lang, target_lang, json.dumps(serializable)),
        )
        conn.commit()
        conn.close()
        logger.info("Cache SET for: '%s'", text[:50])
    except Exception as exc:
        logger.warning("Cache write error: %s", exc)


def cache_stats() -> dict:
    """Return cache statistics."""
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as total, SUM(hit_count) as hits FROM translation_cache"
        ).fetchone()
        conn.close()
        return {"total_entries": row["total"] or 0, "total_hits": row["hits"] or 0}
    except Exception:
        return {"total_entries": 0, "total_hits": 0}
