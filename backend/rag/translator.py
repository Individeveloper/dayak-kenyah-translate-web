"""
RAG Translator — Orchestrates Retrieval-Augmented Generation for Translation

Optimized pipeline:
  0. Check SQLite cache — return instantly if hit (0ms)
  1. Single-word lookup — bypass Gemini, use ChromaDB directly (< 200ms)
  2. Search ChromaDB for top-5 relevant entries (reduced from 15)
  3. Call Gemini gemini-2.5-flash with lean prompt + streaming
  4. Parse, validate, and cache the response
"""

import json
import logging
import os
import re

from google import genai
from google.genai import types

from .cache import cache_get, cache_set
from .prompts import build_translation_prompt
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

_EXPECTED_KEYS = {
    "translation",
    "word_breakdown",
    "grammar_explanation",
    "similar_examples",
    "fun_fact",
}

# A single word: no spaces, no punctuation beyond hyphens
_SINGLE_WORD_RE = re.compile(r"^[\w\-']+$", re.UNICODE)


def _extract_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    if text.endswith("```"):
        text = re.sub(r"\n?\s*```$", "", text)
    text = text.strip()
    # Clean trailing commas
    text = re.sub(r",\s*([\]}])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start: brace_end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(
        f"Could not extract valid JSON from model response: {text[:500]}... (length: {len(text)})"
    )


def _validate_response(data: dict) -> dict:
    validated = {
        "translation": data.get("translation", ""),
        "word_breakdown": data.get("word_breakdown", []),
        "grammar_explanation": data.get("grammar_explanation", ""),
        "similar_examples": data.get("similar_examples", []),
        "fun_fact": data.get("fun_fact", ""),
    }
    clean_breakdown = []
    for item in validated["word_breakdown"]:
        if isinstance(item, dict):
            clean_breakdown.append({
                "original": item.get("original", ""),
                "translated": item.get("translated", ""),
                "word_class": item.get("word_class", "other"),
                "explanation": item.get("explanation", ""),
            })
    validated["word_breakdown"] = clean_breakdown

    clean_examples = []
    for item in validated["similar_examples"]:
        if isinstance(item, dict):
            clean_examples.append({
                "source": item.get("source", ""),
                "translated": item.get("translated", ""),
            })
    validated["similar_examples"] = clean_examples
    return validated


def _format_context(search_results: list[dict]) -> str:
    if not search_results:
        return "(No relevant dictionary entries found.)"
    lines = []
    for i, result in enumerate(search_results, 1):
        meta = result.get("metadata", {})
        word = meta.get("word", "N/A")
        translation = meta.get("translation", "N/A")
        examples = meta.get("examples", "")
        entry_text = f"{i}. {word} → {translation}"
        if examples:
            entry_text += f" | Ex: {examples[:80]}"
        lines.append(entry_text)
    return "\n".join(lines)


def _direct_lookup(
    word: str, source_lang: str, target_lang: str, vector_store: VectorStore
) -> dict | None:
    """
    Bypass Gemini for single-word queries. Search ChromaDB directly and
    return a structured result if the top match is a close enough hit.
    """
    try:
        results = vector_store.search(query=word, n_results=3)
        if not results:
            return None

        top = results[0]
        meta = top.get("metadata", {})
        db_word = meta.get("word", "").strip().lower()
        query_word = word.strip().lower()

        # Only use direct lookup if it's a near-exact match
        if db_word != query_word and not db_word.startswith(query_word):
            return None

        original_word = meta.get("word", word)
        translation = meta.get("translation", "")
        examples_raw = meta.get("examples", "")
        examples = [e.strip() for e in examples_raw.split("|") if e.strip()] if examples_raw else []

        result = {
            "translation": translation,
            "word_breakdown": [
                {
                    "original": original_word,
                    "translated": translation,
                    "word_class": "other",
                    "explanation": f"Ditemukan langsung dari kamus (halaman {meta.get('page_number', '?')}).",
                }
            ],
            "grammar_explanation": "",
            "similar_examples": [
                {"source": ex, "translated": ""} for ex in examples[:3]
            ],
            "fun_fact": "",
            "error": False,
            "dictionary_entries_used": 1,
            "from_direct_lookup": True,
        }
        logger.info("Direct lookup HIT for word: '%s' → '%s'", word, translation)
        return result
    except Exception as exc:
        logger.warning("Direct lookup failed: %s", exc)
        return None


async def translate(
    input_text: str,
    source_lang: str,
    target_lang: str,
    vector_store: VectorStore,
    api_key: str | None = None,
) -> dict:
    """
    Execute the optimized RAG translation pipeline.

    Priority chain:
      0. SQLite cache      → 0ms
      1. Direct DB lookup  → <200ms (single words only)
      2. Gemini API + RAG  → 1-3s
    """
    text = input_text.strip()

    # ── Optimization 0: Cache lookup ──────────────────────────────────────────
    cached = cache_get(text, source_lang, target_lang)
    if cached:
        return cached

    # ── Optimization 1: Direct single-word lookup (no Gemini needed) ──────────
    if _SINGLE_WORD_RE.match(text) and len(text.split()) == 1:
        direct = _direct_lookup(text, source_lang, target_lang, vector_store)
        if direct:
            cache_set(text, source_lang, target_lang, direct)
            return direct

    # ── Resolve API key ───────────────────────────────────────────────────────
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if resolved_api_key == "your_api_key_here":
        resolved_api_key = None
    if not resolved_api_key:
        return {
            "error": True,
            "message": "No valid Gemini API key. Please set GEMINI_API_KEY in .env file.",
        }

    # ── Step 1: Retrieve top-5 dictionary entries (reduced from 15) ──────────
    try:
        logger.info("Searching ChromaDB for: '%s'", text[:80])
        search_results = vector_store.search(query=text, n_results=5)
        dictionary_context = _format_context(search_results)
    except Exception as exc:
        logger.error("Vector search failed: %s", exc, exc_info=True)
        dictionary_context = "(Dictionary search failed.)"
        search_results = []

    # ── Step 2: Build lean prompt ─────────────────────────────────────────────
    system_prompt, user_prompt = build_translation_prompt(
        input_text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        dictionary_context=dictionary_context,
    )

    # ── Step 3: Call Gemini with streaming ───────────────────────────────────
    try:
        logger.info("Calling Gemini API (gemini-2.5-flash) with streaming...")
        client = genai.Client(api_key=resolved_api_key)

        # Use streaming to collect chunks faster
        full_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                max_output_tokens=4096,
                response_mime_type="application/json",
            ),
        ):
            if chunk.text:
                full_text += chunk.text

        raw_response = full_text
        logger.debug("Raw Gemini response: %s", raw_response[:300])

    except Exception as exc:
        logger.error("Gemini API call failed: %s", exc, exc_info=True)
        return {
            "error": True,
            "message": f"Translation API call failed: {str(exc)}",
        }

    # ── Step 4: Parse, validate, and cache ───────────────────────────────────
    try:
        parsed = _extract_json(raw_response)
        result = _validate_response(parsed)
        result["error"] = False
        result["dictionary_entries_used"] = len(search_results)
        result["from_cache"] = False

        # Store in cache for next time
        cache_set(text, source_lang, target_lang, result)
        return result

    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("Failed to parse Gemini response: %s", exc, exc_info=True)
        return {
            "error": True,
            "message": f"Failed to parse translation response: {str(exc)}",
            "raw_response": raw_response[:500],
        }
