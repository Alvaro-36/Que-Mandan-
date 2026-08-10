import logging
from extractor.models import Extraction, ChunkExtraction

logger = logging.getLogger(__name__)


def normalize_extraction(extraction: Extraction) -> Extraction:
    """
    Normalizes extracted fields to standard values and clean formatting.

    Parameters:
        extraction (Extraction): Extracted fact object.

    Returns:
        Extraction: Normalized Extraction instance.
    """
    # Polarity normalization mapping
    polaridad_raw = extraction.polaridad.strip().lower()
    if any(p in polaridad_raw for p in ["pos", "buen", "afirma"]):
        polaridad = "positiva"
    elif any(p in polaridad_raw for p in ["neg", "mala", "niega"]):
        polaridad = "negativa"
    else:
        polaridad = "neutra"

    # Certainty normalization mapping
    certeza_raw = extraction.certeza.strip().lower()
    if "alt" in certeza_raw or "segur" in certeza_raw:
        certeza = "alta"
    elif "med" in certeza_raw or "probab" in certeza_raw:
        certeza = "media"
    elif "baj" in certeza_raw or "duda" in certeza_raw:
        certeza = "baja"
    elif "hipot" in certeza_raw or "quiz" in certeza_raw:
        certeza = "hipotetica"
    else:
        certeza = "alta"

    return Extraction(
        usuario=extraction.usuario.strip(),
        categoria=extraction.categoria.strip().title(),
        subcategoria=extraction.subcategoria.strip().title(),
        entidad=extraction.entidad.strip(),
        atributo_o_valor=extraction.atributo_o_valor.strip(),
        polaridad=polaridad,
        certeza=certeza,
        evidencia=extraction.evidencia.strip()
    )


def normalize_chunk_extraction(chunk_extraction: ChunkExtraction) -> ChunkExtraction:
    """
    Applies normalization to all extractions in a ChunkExtraction container.
    """
    normalized_extracciones = [
        normalize_extraction(ext) for ext in chunk_extraction.extracciones
    ]
    return ChunkExtraction(
        chunk_id=str(chunk_extraction.chunk_id),
        extracciones=normalized_extracciones
    )
