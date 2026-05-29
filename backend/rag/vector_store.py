"""
Vector Store — ChromaDB Wrapper with Local Embeddings

This module manages the persistent ChromaDB vector database used to store
and retrieve Dayak Kenyah dictionary entries. It uses a LOCAL open-source
model (all-MiniLM-L6-v2) to generate embeddings — completely free, no API
key needed, and no rate limits.

Key design decisions:
- Persistent storage in backend/data/chroma_db/ so data survives restarts
- Local embedding model (no Google API calls for embedding)
- Deterministic document IDs based on content hash to avoid duplicates
"""

import hashlib
import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from .document_processor import DictionaryEntry

logger = logging.getLogger(__name__)

# Default ChromaDB collection name
_COLLECTION_NAME = "dayak_kenyah_dictionary"

# Persistent storage path (relative to the backend directory)
_DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "chroma_db")


class VectorStore:
    """
    Wrapper around ChromaDB for storing and searching dictionary entries.

    Uses a LOCAL open-source model (all-MiniLM-L6-v2) to generate dense
    vector embeddings. No Google API key is needed for this process.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize the vector store.

        Args:
            db_path: Path to the ChromaDB persistent storage directory.
                     Defaults to backend/data/chroma_db.
        """
        self.db_path = db_path or _DEFAULT_DB_PATH

        # Use a fully local embedding model — no API key, no rate limits
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        logger.info(
            "VectorStore initialized — db_path=%s, collection=%s, entries=%d",
            self.db_path,
            _COLLECTION_NAME,
            self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    # The _generate_embeddings function is removed because ChromaDB 
    # handles embedding automatically via DefaultEmbeddingFunction.

    @staticmethod
    def _content_hash(text: str) -> str:
        """Create a deterministic ID from text content to prevent duplicates."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_entries(self, entries: list[DictionaryEntry]) -> int:
        """
        Add dictionary entries to the ChromaDB collection.

        Each entry is embedded using a local model and stored with its metadata.
        Duplicate entries (same content hash) are silently skipped by ChromaDB
        via the upsert operation.

        Args:
            entries: A list of DictionaryEntry objects to store.

        Returns:
            The number of entries that were added/updated.
        """
        if not entries:
            logger.warning("No entries to add.")
            return 0

        # Prepare documents, metadata, and IDs
        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []
        seen_ids: set[str] = set()

        for entry in entries:
            doc_text = entry.to_document()
            doc_id = self._content_hash(doc_text)

            # Avoid inserting duplicate IDs in the same batch, which crashes ChromaDB
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            documents.append(doc_text)
            metadatas.append(entry.to_metadata())
            ids.append(doc_id)

        # Upsert into ChromaDB (it will automatically generate embeddings using the local model)
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Upserted %d entries into ChromaDB.", len(documents))
        return len(documents)

    def search(self, query: str, n_results: int = 10) -> list[dict]:
        """
        Search the vector store for dictionary entries relevant to the query.

        Args:
            query: The search text (e.g., a word or phrase to translate).
            n_results: Maximum number of results to return.

        Returns:
            A list of dicts, each containing:
                - 'document': The stored document text
                - 'metadata': The entry metadata (word, translation, etc.)
                - 'distance': The cosine distance (lower = more similar)
        """
        if self.collection.count() == 0:
            logger.warning("Search called on an empty collection.")
            return []

        # Query ChromaDB (it automatically embeds the query text)
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Flatten the results into a clean list
        search_results: list[dict] = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                search_results.append({
                    "document": doc,
                    "metadata": meta,
                    "distance": dist,
                })

        logger.info(
            "Search for '%s' returned %d results.",
            query[:50],
            len(search_results),
        )
        return search_results

    def get_entry_count(self) -> int:
        """Return the total number of entries in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """
        Delete the collection and recreate it (effectively clearing all data).
        Useful for re-uploading a dictionary from scratch.
        """
        self.chroma_client.delete_collection(name=_COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared all entries from the vector store.")
