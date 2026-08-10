import os
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from extractor.models import ChunkExtraction, PipelineOutput
from extractor.prompt import build_extraction_prompt, SYSTEM_PROMPT
from extractor.llm import query_llm
from extractor.parser import parse_json_response
from extractor.validator import validate_chunk_extraction_schema
from extractor.evidence import validate_evidence
from extractor.normalizer import normalize_chunk_extraction

logger = logging.getLogger(__name__)


def prepare_chunk_text(chunk: Dict[str, Any]) -> str:
    """
    Extracts or builds the continuous text representation of a chunk.
    """
    if "text" in chunk and chunk["text"]:
        return str(chunk["text"])
    if "texto" in chunk and chunk["texto"]:
        return str(chunk["texto"])
    if "mensajes" in chunk and isinstance(chunk["mensajes"], list):
        lines = []
        for m in chunk["mensajes"]:
            author = m.get("autor") or m.get("author") or ""
            text = m.get("texto") or m.get("message") or ""
            if author:
                lines.append(f"{author}: {text}")
            else:
                lines.append(text)
        return "\n".join(lines)
    return ""


def analyze_chunk_with_llm(chunk: Dict[str, Any], model_name: str = "MiniMaxAI/MiniMax-M3") -> ChunkExtraction:
    """
    Fase 1: Analiza un chunk individual mediante MiniMax-M3 (Nebius API) + parseo JSON + validacion de esquema.
    No ejecuta validacion de evidencia ni alucinaciones en esta fase.
    """
    chunk_id = str(chunk.get("chunk_id", "unknown"))
    timestamp = str(chunk.get("ts_inicio", chunk.get("timestamp", "")))
    chunk_text = prepare_chunk_text(chunk)

    if not chunk_text.strip():
        logger.warning("Empty chunk text for chunk_id '%s'. Returning empty extracciones.", chunk_id)
        return ChunkExtraction(chunk_id=chunk_id, extracciones=[])

    prompt = build_extraction_prompt(chunk_id, timestamp, chunk_text)

    parsed_dict = None
    stage = "LLM_Query_and_Parsing"
    for attempt in range(1, 3):
        try:
            raw_response = query_llm(prompt, model_name=model_name, system_prompt=SYSTEM_PROMPT)
            parsed_dict = parse_json_response(raw_response)
            break
        except Exception as e:
            logger.warning(
                "Chunk_ID '%s' failed at stage '%s' (Attempt %d/2): %s",
                chunk_id, stage, attempt, str(e)
            )
            if attempt == 2:
                logger.error("Chunk_ID '%s' failed both LLM query/parsing attempts. Returning empty extracciones.", chunk_id)
                return ChunkExtraction(chunk_id=chunk_id, extracciones=[])

    if not parsed_dict:
        return ChunkExtraction(chunk_id=chunk_id, extracciones=[])

    stage = "Schema_Validation"
    try:
        chunk_extraction = validate_chunk_extraction_schema(parsed_dict)
        return chunk_extraction
    except Exception as e:
        logger.error("Chunk_ID '%s' failed at stage '%s': %s", chunk_id, stage, str(e))
        return ChunkExtraction(chunk_id=chunk_id, extracciones=[])


def validate_and_normalize_chunk(chunk: Dict[str, Any], chunk_ext: ChunkExtraction) -> ChunkExtraction:
    """
    Fase 2: Ejecuta la validacion de evidencia (anti-alucinaciones) y la normalizacion para un chunk.
    """
    chunk_id = str(chunk_ext.chunk_id)
    chunk_text = prepare_chunk_text(chunk)

    # Validacion de Evidencia (Anti-Alucinaciones)
    valid_extracciones = []
    for ext in chunk_ext.extracciones:
        try:
            if validate_evidence(chunk_text, ext):
                valid_extracciones.append(ext)
            else:
                logger.info("Chunk_ID '%s': Extraccion para entidad '%s' descartada por alucinacion de evidencia.", chunk_id, ext.entidad)
        except Exception as e:
            logger.error("Chunk_ID '%s' evidence validation failed for extraction: %s", chunk_id, str(e))

    validated_chunk_ext = ChunkExtraction(chunk_id=chunk_id, extracciones=valid_extracciones)

    # Normalizacion
    try:
        return normalize_chunk_extraction(validated_chunk_ext)
    except Exception as e:
        logger.error("Chunk_ID '%s' normalization failed: %s", chunk_id, str(e))
        return validated_chunk_ext


def process_all_chunks(
    chunks: List[Dict[str, Any]],
    model_name: str = "MiniMaxAI/MiniMax-M3",
    concurrency: int = 40
) -> PipelineOutput:
    """
    Processes all chunks in two parallelized phases.

    Fase 1: Concurrent LLM API analysis (MiniMax-M3 via Nebius API), JSON parsing, and schema validation across ALL chunks.
    Fase 2: Evidence validation (anti-hallucination) and normalization across ALL chunks.

    Parameters:
        chunks (List[Dict[str, Any]]): List of chunk dictionaries.
        model_name (str): Nebius model name. Defaults to 'MiniMaxAI/MiniMax-M3'.
        concurrency (int): Number of simultaneous HTTP API requests. Defaults to 15.

    Returns:
        PipelineOutput: Container with final validated chunk extractions.
    """
    n_chunks = len(chunks)
    if n_chunks > 0:
        effective_concurrency = min(concurrency, n_chunks)
    else:
        effective_concurrency = 1

    logger.info("=== FASE 1: Analisis LLM (%s) en %d chunks (Peticiones simultaneas: %d) ===", model_name, n_chunks, effective_concurrency)

    raw_extractions: List[ChunkExtraction] = [None] * n_chunks

    def _task_fase1(index_and_chunk):
        idx, chk = index_and_chunk
        chunk_id = str(chk.get("chunk_id", idx))
        try:
            res = analyze_chunk_with_llm(chk, model_name=model_name)
            return idx, res
        except Exception as e:
            logger.error("Fase 1 - Error inesperado en chunk_id '%s': %s", chunk_id, str(e))
            return idx, ChunkExtraction(chunk_id=chunk_id, extracciones=[])

    with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
        futures = [executor.submit(_task_fase1, (i, chk)) for i, chk in enumerate(chunks)]
        completed_count = 0
        for future in as_completed(futures):
            idx, res = future.result()
            raw_extractions[idx] = res
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == n_chunks:
                logger.info("Fase 1 - Progreso: %d/%d chunks procesados.", completed_count, n_chunks)

    logger.info("=== FASE 1 COMPLETADA (%d chunks analizados con %s) ===", n_chunks, model_name)

    logger.info("=== FASE 2: Validacion de Evidencia y Normalizacion ===")
    final_results: List[ChunkExtraction] = [None] * n_chunks

    def _task_fase2(index_chunk_and_ext):
        idx, chk, chk_ext = index_chunk_and_ext
        chunk_id = str(chk_ext.chunk_id)
        try:
            final_ext = validate_and_normalize_chunk(chk, chk_ext)
            return idx, final_ext
        except Exception as e:
            logger.error("Fase 2 - Error inesperado en chunk_id '%s': %s", chunk_id, str(e))
            return idx, ChunkExtraction(chunk_id=chunk_id, extracciones=[])

    with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
        futures = [
            executor.submit(_task_fase2, (i, chk, raw_extractions[i]))
            for i, chk in enumerate(chunks)
        ]
        for future in as_completed(futures):
            idx, res = future.result()
            final_results[idx] = res

    logger.info("=== FASE 2 COMPLETADA (Pipeline finalizado) ===")
    return PipelineOutput(chunks_procesados=final_results)
