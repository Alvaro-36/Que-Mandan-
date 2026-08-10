import logging
from pydantic import ValidationError
from extractor.models import ChunkExtraction

logger = logging.getLogger(__name__)


def validate_chunk_extraction_schema(data: dict) -> ChunkExtraction:
    """
    Validates a parsed JSON dictionary against the ChunkExtraction Pydantic schema.

    Parameters:
        data (dict): Dictionary obtained from JSON parsing.

    Returns:
        ChunkExtraction: Validated Pydantic instance.

    Raises:
        ValidationError: If the structure does not match the Pydantic schema.
    """
    if isinstance(data, dict) and "chunk_id" in data:
        data["chunk_id"] = str(data["chunk_id"])

    try:
        return ChunkExtraction.model_validate(data)
    except ValidationError as e:
        chunk_id = data.get("chunk_id", "unknown") if isinstance(data, dict) else "unknown"
        logger.error("Schema validation failed for chunk_id '%s': %s", chunk_id, str(e))
        raise
