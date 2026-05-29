"""
Document Processor for Dayak Kenyah Dictionary Files

This module handles:
1. Extracting raw text from each page of a PDF using PyMuPDF (fitz)
2. Chunking the extracted text into individual dictionary entries
3. Structuring each entry with word, translation, examples, and page metadata

The processor is designed to handle typical bilingual dictionary layouts where
entries are separated by newlines, and each entry contains a headword followed
by its translation and optional usage examples.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO
import io
import csv

import fitz  # PyMuPDF
import docx

logger = logging.getLogger(__name__)


@dataclass
class DictionaryEntry:
    """Represents a single dictionary entry extracted from the PDF."""

    word: str  # The headword or phrase
    translation: str  # The translation / definition text
    examples: list[str] = field(default_factory=list)  # Usage examples, if any
    page_number: int = 0  # 1-indexed page number from the source PDF
    raw_text: str = ""  # The original unprocessed text block

    def to_document(self) -> str:
        """
        Serialize the entry into a single text block suitable for embedding.
        This format ensures the vector search can match on all relevant parts.
        """
        parts = [f"Word: {self.word}", f"Translation: {self.translation}"]
        if self.examples:
            parts.append("Examples: " + " | ".join(self.examples))
        return "\n".join(parts)

    def to_metadata(self) -> dict:
        """Return metadata dict for ChromaDB storage."""
        return {
            "word": self.word,
            "translation": self.translation,
            "page_number": self.page_number,
            "examples": " | ".join(self.examples) if self.examples else "",
            "raw_text": self.raw_text,
        }


def extract_text_from_pdf(pdf_source: str | Path | BinaryIO) -> list[dict]:
    """
    Extract text from each page of a PDF file.

    Args:
        pdf_source: Either a file path (str/Path) or a file-like object (BinaryIO)
                    containing the PDF data.

    Returns:
        A list of dicts, each with keys:
            - 'page_number' (int): 1-indexed page number
            - 'text' (str): The raw text content of that page

    Raises:
        ValueError: If the PDF cannot be opened or contains no text.
    """
    pages: list[dict] = []

    try:
        # Open from path or from bytes/stream
        if isinstance(pdf_source, (str, Path)):
            doc = fitz.open(str(pdf_source))
        else:
            # Read bytes from the file-like object
            pdf_bytes = pdf_source.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if doc.page_count == 0:
            raise ValueError("The PDF file contains no pages.")

        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            text = page.get_text("text")  # Plain text extraction
            if text.strip():
                pages.append({
                    "page_number": page_idx + 1,  # 1-indexed
                    "text": text.strip(),
                })

        logger.info("Extracted text from %d pages (out of %d total).", len(pages), doc.page_count)
        doc.close()

    except fitz.FileDataError as exc:
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    if not pages:
        raise ValueError("No text could be extracted from the PDF. It may be image-based.")

    return pages


def chunk_into_entries(pages: list[dict]) -> list[DictionaryEntry]:
    """
    Parse extracted page texts into structured dictionary entries.

    Strategy:
    ---------
    1. Split each page's text by blank lines to get raw blocks.
    2. For each block, try to identify a headword (first significant line)
       and the remaining lines as translation / examples.
    3. Lines starting with common example markers (e.g., "~", "Ex:", "Cth:",
       "Contoh:", or lines in quotes) are classified as usage examples.
    4. Everything else in the block is treated as the translation/definition.

    This heuristic works well for most bilingual dictionary PDFs, but can be
    extended with more specific patterns if the PDF format is known.

    Args:
        pages: Output of `extract_text_from_pdf`.

    Returns:
        A list of DictionaryEntry objects.
    """
    entries: list[DictionaryEntry] = []

    # Patterns that indicate a line is a usage example
    example_pattern = re.compile(
        r"^(?:~|ex[.:]|cth[.:]|contoh[.:]|e\.g\.|mis[.:]|\"|\'|→)",
        re.IGNORECASE,
    )

    for page in pages:
        page_num = page["page_number"]
        text = page["text"]

        # Split the page into blocks separated by one or more blank lines
        blocks = re.split(r"\n\s*\n", text)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = [line.strip() for line in block.split("\n") if line.strip()]
            if not lines:
                continue

            # The first line is treated as the headword / phrase
            headword = lines[0]

            translation_parts: list[str] = []
            example_parts: list[str] = []

            for line in lines[1:]:
                if example_pattern.match(line):
                    example_parts.append(line)
                else:
                    translation_parts.append(line)

            # If we only have a headword with no translation, combine small
            # blocks or keep the headword as both word and translation
            translation = " ".join(translation_parts) if translation_parts else headword

            entry = DictionaryEntry(
                word=headword,
                translation=translation,
                examples=example_parts,
                page_number=page_num,
                raw_text=block,
            )
            entries.append(entry)

    logger.info("Chunked PDF into %d dictionary entries.", len(entries))
    return entries


def process_pdf(pdf_source: str | Path | BinaryIO) -> list[DictionaryEntry]:
    """
    High-level convenience function: extract text from a PDF and chunk it
    into dictionary entries in one call.
    """
    pages = extract_text_from_pdf(pdf_source)
    entries = chunk_into_entries(pages)
    return entries


def process_docx(docx_source: BinaryIO) -> list[DictionaryEntry]:
    """
    Extract text from a DOCX file and chunk it into dictionary entries.
    We process each paragraph as a line, group them into blocks by empty paragraphs,
    and then use the same chunking strategy as PDF.
    """
    doc = docx.Document(docx_source)
    text_blocks = []
    current_block = []

    for para in doc.paragraphs:
        line = para.text.strip()
        if not line:
            if current_block:
                text_blocks.append("\n".join(current_block))
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        text_blocks.append("\n".join(current_block))

    # Re-use the chunking strategy but pretend everything is on "Page 1"
    mock_pages = [{"page_number": 1, "text": "\n\n".join(text_blocks)}]
    return chunk_into_entries(mock_pages)


def process_csv(csv_source: BinaryIO) -> list[DictionaryEntry]:
    """
    Extract text from a CSV file.
    Expects columns like: Word, Translation, Examples (optional)
    """
    text_wrapper = io.TextIOWrapper(csv_source, encoding='utf-8')
    reader = csv.DictReader(text_wrapper)
    
    # Check if we have the expected columns (case-insensitive)
    if not reader.fieldnames:
        raise ValueError("The CSV file is empty or missing headers.")
        
    headers = {str(h).strip().lower(): str(h).strip() for h in reader.fieldnames if h}
    
    if not any(k in headers for k in ['word', 'kata', 'headword']):
        # If no clear headers, fallback to treating it as just plain columns (Word in Col 1, Translation in Col 2)
        text_wrapper.seek(0)
        simple_reader = csv.reader(text_wrapper)
        next(simple_reader, None)  # skip header
        
        entries = []
        for i, row in enumerate(simple_reader, 2):
            if not row or not row[0].strip():
                continue
            word = row[0].strip()
            translation = row[1].strip() if len(row) > 1 else word
            examples = [row[2].strip()] if len(row) > 2 and row[2].strip() else []
            entries.append(DictionaryEntry(
                word=word,
                translation=translation,
                examples=examples,
                page_number=i,
                raw_text=f"{word} - {translation}"
            ))
        return entries

    # Mapped headers
    word_col = headers.get('word', headers.get('kata', headers.get('headword')))
    trans_col = headers.get('translation', headers.get('terjemahan', headers.get('arti')))
    ex_col = headers.get('examples', headers.get('contoh', headers.get('example')))

    entries = []
    for i, row in enumerate(reader, 2):
        word = row.get(word_col, "").strip() if word_col else ""
        if not word:
            continue
            
        translation = row.get(trans_col, "").strip() if trans_col else word
        examples_str = row.get(ex_col, "").strip() if ex_col else ""
        examples = [examples_str] if examples_str else []
        
        entries.append(DictionaryEntry(
            word=word,
            translation=translation,
            examples=examples,
            page_number=i,
            raw_text=f"{word} - {translation}"
        ))
        
    logger.info("Extracted %d dictionary entries from CSV.", len(entries))
    return entries


def process_file(file_source: BinaryIO, extension: str) -> list[DictionaryEntry]:
    """
    Route the file to the correct processor based on extension.
    """
    ext = extension.lower()
    if ext == ".pdf":
        return process_pdf(file_source)
    elif ext == ".docx":
        return process_docx(file_source)
    elif ext == ".csv":
        return process_csv(file_source)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
