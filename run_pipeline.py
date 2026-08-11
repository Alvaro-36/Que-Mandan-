"""
run_pipeline.py - Script ejecutor del pipeline completo de analisis de chats.

Pasos del pipeline:
1. main.py: Convierte chatWPP.txt a chatWPP.json aplicando burst merging.
2. chunker.py: Segmenta los mensajes en chunks (chunks_debug.json).
3. extractor.main: Extrae conocimientos de cada chunk mediante LLM (extracciones_salida.json).
4. agrupador.py: Genera embeddings de extracciones y agrupa mediante UMAP+HDBSCAN (clusters_temas.json).
5. clasificador/main.py: Clasifica cada cluster tematico usando LLM (clusters_clasificados.json).
"""

import sys
import os
import subprocess
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_step(step_number: int, step_name: str, command: list[str]) -> None:
    logger.info("=" * 60)
    logger.info("PASO %d: %s", step_number, step_name)
    logger.info("Comando: %s", " ".join(command))
    logger.info("=" * 60)

    start_time = time.time()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        logger.error("Error en el paso %d (%s). Codigo de salida: %d", step_number, step_name, result.returncode)
        sys.exit(result.returncode)

    logger.info("Paso %d completado exitosamente en %.2f segundos.", step_number, elapsed)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    start_total = time.time()

    python_executable = sys.executable

    logger.info("=== INICIANDO PIPELINE COMPLETO DE ANALISIS DE CHATS ===")

    # Paso 1: Preprocesamiento y Burst Merging (main.py)
    run_step(1, "Preprocesamiento y Burst Merging", [python_executable, "main.py"])

    # Paso 2: Chunking Hibrido (chunker.py)
    run_step(2, "Chunking Temporal + Semantico", [python_executable, "chunker.py"])

    # Paso 3: Extraccion de Conocimiento (extractor.main)
    run_step(3, "Extraccion de Conocimiento con LLM", [python_executable, "-m", "extractor.main"])

    # Paso 4: Agrupamiento Tematico (agrupador.py)
    run_step(4, "Agrupamiento Tematico UMAP + HDBSCAN", [python_executable, "agrupador.py"])

    # Paso 5: Clasificacion de Titulos de Clusters (clasificador/main.py)
    run_step(5, "Clasificacion de Titulos de Clusters", [python_executable, "-m", "clasificador.main"])

    total_elapsed = time.time() - start_total
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETO FINALIZADO EXITOSAMENTE en %.2f segundos.", total_elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
