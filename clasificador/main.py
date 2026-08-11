"""
clasificador/main.py - Entry point for cluster title classification.

Reads clusters_temas.json and extracciones_salida.json, classifies each cluster
using qwen2.5:3b local via Ollama, and exports the results to clusters_clasificados.json.
"""

import json
import logging
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clasificador.clasificador import clasificar_todos_los_clusters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CLUSTERS_PATH: str = "clusters_temas.json"
EXTRACCIONES_PATH: str = "extracciones_salida.json"
OUTPUT_PATH: str = "clusters_clasificados.json"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logger.info("=== Starting Cluster Title Classification Pipeline ===")

    resultados = clasificar_todos_los_clusters(CLUSTERS_PATH, EXTRACCIONES_PATH)

    output = {
        "n_clusters": len([r for r in resultados if r["cluster_id"] != -1]),
        "clasificaciones": resultados,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    logger.info("Output saved to '%s'.", OUTPUT_PATH)

    print("\n" + "=" * 55)
    print(f"RESULTADOS DE CLASIFICACION ({OUTPUT_PATH})")
    print("=" * 55)
    print(f"{'Cluster':>8} | {'Titulo':<28} | {'Chunks':>6}")
    print("-" * 55)
    for r in resultados:
        cid = "Ruido" if r["cluster_id"] == -1 else str(r["cluster_id"])
        print(f"{cid:>8} | {r['titulo']:<28} | {r['n_chunks']:>6}")
    print("=" * 55)


if __name__ == "__main__":
    main()
