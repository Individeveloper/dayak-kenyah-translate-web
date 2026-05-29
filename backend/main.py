"""
Dayak Kenyah Translation API — FastAPI Application

This is the main entry point for the backend server. It exposes REST endpoints
for translating text, uploading dictionary PDFs, and checking system status.

Endpoints:
  POST /api/translate          — Translate text between Dayak Kenyah and other languages
  POST /api/upload-dictionary  — Upload a PDF dictionary to build the RAG knowledge base
  GET  /api/dictionary-status  — Check if a dictionary is loaded and how many entries exist
  GET  /api/health             — Simple health check

Run with:
  uvicorn main:app --reload --port 8000
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.document_processor import process_file
from rag.prompts import build_translation_prompt
from rag.translator import translate
from rag.vector_store import VectorStore
from rag.cache import cache_stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load environment variables from .env file (if it exists)
load_dotenv()

# Configure logging — INFO for production, DEBUG during development
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifespan — initialize shared resources on startup
# ---------------------------------------------------------------------------

# Global reference to the vector store (initialized during lifespan startup)
vector_store: VectorStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    On startup:
      - Validate that GEMINI_API_KEY is set (for translation only, NOT for embedding)
      - Initialize the ChromaDB vector store (uses local embedding model)

    On shutdown:
      - Clean up resources (currently no-op; ChromaDB handles its own cleanup)
    """
    global vector_store

    # Resolve API key: check GEMINI_API_KEY first, then GOOGLE_API_KEY
    # NOTE: This key is ONLY used for the translation feature (Gemini LLM),
    # NOT for embedding. Embedding is 100% local and free.
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    # Filter out placeholder value
    if api_key == "your_api_key_here":
        api_key = None

    if not api_key:
        logger.warning(
            "[WARNING] No valid Gemini/Google API key found in environment. "
            "Translation feature will not work (embedding is local and unaffected). "
            "Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file."
        )

    # Initialize the vector store (uses LOCAL model for embedding — no API key needed)
    try:
        vector_store = VectorStore()
        logger.info(
            "[OK] Vector store ready - %d dictionary entries loaded.",
            vector_store.get_entry_count(),
        )
    except Exception as exc:
        logger.error("[ERROR] Failed to initialize vector store: %s", exc, exc_info=True)
        vector_store = None

    yield  # App is running

    # Shutdown cleanup
    logger.info("[INFO] Shutting down Dayak Kenyah Translation API.")


# ---------------------------------------------------------------------------
# FastAPI app instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Dayak Kenyah Translation API",
    description=(
        "A RAG-powered translation API for the Dayak Kenyah language. "
        "Upload a PDF dictionary, then translate text with contextual learning data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class TranslateRequest(BaseModel):
    """Schema for the translation request body."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The text to translate.",
        examples=["Alo kuman nah?"],
    )
    source_lang: str = Field(
        default="Dayak Kenyah",
        description="Source language name.",
        examples=["Dayak Kenyah", "Indonesian", "English"],
    )
    target_lang: str = Field(
        default="Indonesian",
        description="Target language name.",
        examples=["Indonesian", "English", "Dayak Kenyah"],
    )


class WordBreakdown(BaseModel):
    """A single word in the breakdown analysis."""

    original: str
    translated: str
    word_class: str
    explanation: str


class ExampleSentence(BaseModel):
    """A source–translated example sentence pair."""

    source: str
    translated: str


class TranslateResponse(BaseModel):
    """Schema for a successful translation response."""

    error: bool = False
    translation: str
    word_breakdown: list[WordBreakdown] = []
    grammar_explanation: str = ""
    similar_examples: list[ExampleSentence] = []
    fun_fact: str = ""
    dictionary_entries_used: int = 0
    from_cache: bool = False
    from_direct_lookup: bool = False


class ErrorResponse(BaseModel):
    """Schema for error responses."""

    error: bool = True
    message: str
    raw_response: str | None = None


class DictionaryStatusResponse(BaseModel):
    """Schema for the dictionary status endpoint."""

    loaded: bool
    entry_count: int
    message: str


class UploadResponse(BaseModel):
    """Schema for the dictionary upload endpoint response."""

    success: bool
    message: str
    entries_added: int
    total_entries: int


class HealthResponse(BaseModel):
    """Schema for the health check endpoint."""

    status: str
    timestamp: str
    vector_store_ready: bool
    dictionary_entries: int


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/api/translate",
    response_model=TranslateResponse | ErrorResponse,
    summary="Translate text",
    description="Translate text between Dayak Kenyah and other languages using RAG.",
)
async def api_translate(request: TranslateRequest):
    """
    Main translation endpoint.

    Accepts a text string with source and target language, retrieves relevant
    dictionary entries from ChromaDB, and uses Gemini to produce a grounded
    translation with educational metadata.
    """
    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not initialized. Please check the server logs.",
        )

    logger.info(
        "Translation request: '%s' [%s → %s]",
        request.text[:80],
        request.source_lang,
        request.target_lang,
    )

    result = await translate(
        input_text=request.text,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        vector_store=vector_store,
    )

    if result.get("error"):
        logger.warning("Translation failed: %s", result.get("message"))
        return ErrorResponse(
            message=result.get("message", "Unknown error"),
            raw_response=result.get("raw_response"),
        )

    return TranslateResponse(**result)


@app.post(
    "/api/upload-dictionary",
    response_model=UploadResponse,
    summary="Upload a dictionary PDF",
    description="Upload a Dayak Kenyah dictionary PDF to build the translation knowledge base.",
)
async def api_upload_dictionary(file: UploadFile = File(...)):
    """
    Upload and process a PDF dictionary file.

    The PDF is parsed into individual dictionary entries, which are then
    embedded using Gemini and stored in ChromaDB for semantic retrieval.
    """
    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not initialized. Please check the server logs.",
        )

    # Validate file type
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    allowed_exts = [".pdf", ".docx", ".csv"]
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_exts)}",
        )

    # Validate file size (max 50 MB)
    max_size = 50 * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is 50 MB, got {len(contents) / 1024 / 1024:.1f} MB.",
        )

    logger.info("Processing uploaded dictionary: %s (%d bytes)", file.filename, len(contents))

    try:
        # Step 1: Extract text and chunk into dictionary entries
        import io

        file_stream = io.BytesIO(contents)
        entries = process_file(file_stream, ext)

        if not entries:
            raise HTTPException(
                status_code=400,
                detail=f"No dictionary entries could be extracted from the {ext.upper()} file.",
            )

        logger.info("Extracted %d dictionary entries from %s", len(entries), file.filename)

        # Step 2: Store entries in ChromaDB with embeddings
        added_count = vector_store.add_entries(entries)
        total_count = vector_store.get_entry_count()

        logger.info(
            "Dictionary upload complete: %d entries added, %d total.",
            added_count,
            total_count,
        )

        return UploadResponse(
            success=True,
            message=f"Successfully processed '{file.filename}'. "
            f"{added_count} entries added to the dictionary.",
            entries_added=added_count,
            total_entries=total_count,
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except ValueError as exc:
        logger.error("PDF processing failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Dictionary upload failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the dictionary: {str(exc)}",
        ) from exc


@app.get(
    "/api/dictionary-status",
    response_model=DictionaryStatusResponse,
    summary="Dictionary status",
    description="Check whether a dictionary has been loaded and how many entries it contains.",
)
async def api_dictionary_status():
    """Return the current status of the dictionary knowledge base."""
    if vector_store is None:
        return DictionaryStatusResponse(
            loaded=False,
            entry_count=0,
            message="Vector store is not initialized.",
        )

    count = vector_store.get_entry_count()
    if count > 0:
        return DictionaryStatusResponse(
            loaded=True,
            entry_count=count,
            message=f"Dictionary loaded with {count} entries.",
        )
    else:
        return DictionaryStatusResponse(
            loaded=False,
            entry_count=0,
            message="No dictionary has been uploaded yet. "
            "Use POST /api/upload-dictionary to upload a PDF.",
        )


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API server is running and its dependencies are ready.",
)
async def api_health():
    """Simple health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        vector_store_ready=vector_store is not None,
        dictionary_entries=vector_store.get_entry_count() if vector_store else 0,
    )


@app.get(
    "/api/cache-stats",
    summary="Cache statistics",
    description="Return statistics about the translation cache.",
)
async def api_cache_stats():
    """Return SQLite cache statistics."""
    stats = cache_stats()
    return {
        "status": "ok",
        "cache_entries": stats["total_entries"],
        "cache_hits": stats["total_hits"],
        "message": f"{stats['total_entries']} translations cached, {stats['total_hits']} cache hits total.",
    }


@app.delete(
    "/api/cache",
    summary="Clear translation cache",
    description="Delete all cached translations.",
)
async def api_clear_cache():
    """Clear the entire translation cache."""
    from rag.cache import DB_PATH
    import sqlite3
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("DELETE FROM translation_cache")
        conn.commit()
        conn.close()
        return {"status": "ok", "message": "Cache cleared successfully."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
