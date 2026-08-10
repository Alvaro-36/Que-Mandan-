import re
import logging
from extractor.models import Extraction

logger = logging.getLogger(__name__)

NEWLINE_SUBSTITUTES = re.compile(r'\s*/\s*|\s*\\\s*n\s*|\s*\\n\s*')
AUTHOR_PREFIX = re.compile(r'^[^:]+:\s*', re.MULTILINE)


def _normalize_for_comparison(text: str) -> str:
    """
    Normalizes text for evidence comparison by collapsing newlines, common LLM
    substitutions for newlines (like ' / ', '\\n'), and excess whitespace into
    single spaces, then lowercasing.
    """
    normalized = NEWLINE_SUBSTITUTES.sub(' ', text)
    normalized = re.sub(r'\s+', ' ', normalized).strip().lower()
    return normalized


def _strip_author_prefixes(text: str) -> str:
    """
    Removes 'Author: ' prefixes from each line of the chunk text,
    leaving only the raw message content. This allows matching evidence
    that spans multiple consecutive messages from the same author
    (e.g. 'me estoy tentando sola\\ndel fer' when the chunk text has
    'Ale Albornoz: me estoy tentando sola\\nAle Albornoz: del fer').
    """
    return AUTHOR_PREFIX.sub('', text)


def validate_evidence(chunk_text: str, extraction: Extraction) -> bool:
    """
    Validates whether the evidence string in an extraction exists inside the chunk text.
    Uses multi-level fallback:
      1. Exact literal match
      2. Case-insensitive / whitespace-normalized
      3. Newline-substitute-normalized (handles ' / ' and '\\n')
      4. Author-prefix-stripped (handles cross-message evidence from same author)

    Parameters:
        chunk_text (str): Complete original text of the chunk.
        extraction (Extraction): Extracted fact object.

    Returns:
        bool: True if evidence exists in chunk_text, False otherwise (hallucination).
    """
    evidencia = extraction.evidencia.strip()
    if not evidencia:
        logger.warning("Empty evidence string encountered for entity '%s'. Marking invalid.", extraction.entidad)
        return False

    # Level 1: Literal exact check
    if evidencia in chunk_text:
        return True

    # Level 2: Case-insensitive / whitespace-normalized check
    chunk_spaces = " ".join(chunk_text.split()).lower()
    evidencia_spaces = " ".join(evidencia.split()).lower()

    if evidencia_spaces in chunk_spaces:
        return True

    # Level 3: Normalize newline substitutes (LLM may use ' / ' or '\\n' instead of real newlines)
    chunk_norm = _normalize_for_comparison(chunk_text)
    evidencia_norm = _normalize_for_comparison(evidencia)

    if evidencia_norm in chunk_norm:
        return True

    # Level 4: Strip author prefixes and re-check with normalization.
    # Handles evidence spanning multiple messages from the same author that weren't burst-merged
    # because another author's message appeared in between.
    chunk_stripped = _strip_author_prefixes(chunk_text)

    if evidencia in chunk_stripped:
        return True

    chunk_stripped_norm = _normalize_for_comparison(chunk_stripped)
    if evidencia_norm in chunk_stripped_norm:
        return True

    logger.warning(
        "Hallucination detected! Evidence '%s' not found literally in chunk text.",
        evidencia[:80]
    )
    return False
