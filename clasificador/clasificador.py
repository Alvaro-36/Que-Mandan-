"""
clasificador/clasificador.py - Core cluster classification logic.
"""

import json
import logging
import requests
from typing import Any

from clasificador.prompt import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "qwen2.5:3b"


def cargar_clusters(path: str = "clusters_temas.json") -> list[dict[str, Any]]:
    """Loads clusters_temas.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    clusters = data.get("clusters", [])
    logger.info("Loaded %d clusters from '%s'.", len(clusters), path)
    return clusters


def cargar_extracciones(path: str = "extracciones_salida.json") -> dict[str, list[dict[str, Any]]]:
    """
    Loads extracciones_salida.json and returns a dict mapping
    chunk_id -> list of extracciones.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks_map: dict[str, list[dict[str, Any]]] = {}
    for chunk in data.get("chunks_procesados", []):
        cid = str(chunk["chunk_id"])
        chunks_map[cid] = chunk.get("extracciones", [])

    logger.info("Loaded extractions for %d chunks from '%s'.", len(chunks_map), path)
    return chunks_map


def construir_texto_cluster(
    chunk_ids: list[str],
    extracciones_map: dict[str, list[dict[str, Any]]],
) -> str:
    """
    Builds a concatenated string of 'categoria subcategoria' for all
    extractions across all chunks in the cluster.
    """
    parts: list[str] = []
    for cid in chunk_ids:
        for ext in extracciones_map.get(cid, []):
            cat = ext.get("categoria", "")
            sub = ext.get("subcategoria", "")
            label = f"{cat} {sub}".strip()
            if label:
                parts.append(label)

    return ", ".join(parts)


def clasificar_con_ollama(texto_temas: str) -> str:
    """
    Sends the concatenated themes to qwen2.5:3b via Ollama and returns
    the single-word title.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": build_user_prompt(texto_temas),
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 20,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        raw = result.get("response", "").strip()
        titulo = raw.split()[0] if raw.split() else "Sin_Titulo"
        titulo = titulo.strip(".,;:!?\"'()[]{}").strip()
        return titulo if titulo else "Sin_Titulo"
    except requests.RequestException as e:
        logger.error("Ollama API error: %s", e)
        return "Error_LLM"


def clasificar_todos_los_clusters(
    clusters_path: str = "clusters_temas.json",
    extracciones_path: str = "extracciones_salida.json",
) -> list[dict[str, Any]]:
    """
    Processes all clusters and classifies each with a title.

    Returns:
        List of dicts with cluster_id, titulo, and n_chunks.
    """
    clusters = cargar_clusters(clusters_path)
    extracciones_map = cargar_extracciones(extracciones_path)

    resultados: list[dict[str, Any]] = []

    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        chunk_ids = cluster["chunk_ids"]
        n_chunks = len(chunk_ids)

        texto = construir_texto_cluster(chunk_ids, extracciones_map)

        if not texto:
            titulo = "Vacio"
            logger.info("Cluster %d: no themes found, marking as 'Vacio'.", cluster_id)
        else:
            logger.info(
                "Cluster %d (%d chunks): classifying %d theme tokens...",
                cluster_id, n_chunks, len(texto.split(",")),
            )
            titulo = clasificar_con_ollama(texto)
            logger.info("Cluster %d -> '%s'", cluster_id, titulo)

        resultados.append({
            "cluster_id": cluster_id,
            "titulo": titulo,
            "n_chunks": n_chunks,
        })

    return resultados
