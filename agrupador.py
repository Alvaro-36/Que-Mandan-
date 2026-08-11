"""
agrupador.py - Topic grouping of conversation chunks via UMAP + HDBSCAN.

Reads extracciones_salida.json, computes sentence embeddings from
the concatenation of categoria + subcategoria per chunk, reduces
dimensionality with UMAP (n_neighbors=15), clusters with HDBSCAN,
and outputs a JSON with cluster assignments plus a debug scatter plot.

Dependencies: numpy, torch, sentence-transformers, umap-learn, hdbscan, matplotlib
"""

import json
import logging
import sys
from typing import Any

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELO_EMBEDDING: str = "paraphrase-multilingual-MiniLM-L12-v2"
UMAP_N_NEIGHBORS: int = 15
UMAP_N_COMPONENTS: int = 2
UMAP_METRIC: str = "cosine"
RANDOM_SEED: int = 42

INPUT_PATH: str = "extracciones_salida.json"
OUTPUT_JSON_PATH: str = "clusters_temas.json"
OUTPUT_PLOT_PATH: str = "umap_clusters.png"

# ---------------------------------------------------------------------------
# Model singleton (same pattern as chunker.py)
# ---------------------------------------------------------------------------
_model_cache: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Returns the SentenceTransformer model, loading it once (singleton)."""
    global _model_cache
    if _model_cache is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading embedding model '%s' on device '%s'...", MODELO_EMBEDDING, device)
        _model_cache = SentenceTransformer(MODELO_EMBEDDING, device=device)
        logger.info("Model loaded successfully.")
    return _model_cache


# ---------------------------------------------------------------------------
# Data loading & text construction
# ---------------------------------------------------------------------------
def cargar_extracciones(path: str) -> list[dict[str, Any]]:
    """Loads extracciones_salida.json and returns the list of chunk dicts."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data.get("chunks_procesados", [])
    logger.info("Loaded %d chunks from '%s'.", len(chunks), path)
    return chunks


def construir_textos(
    chunks: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """
    Builds the embedding text per chunk by concatenating
    'categoria subcategoria' from each extraction, joined by ' | '.

    Returns:
        texts: list of concatenated strings (one per valid chunk).
        chunk_ids: list of chunk_id strings (matching texts).

    Chunks with empty extracciones are excluded.
    """
    texts: list[str] = []
    chunk_ids: list[str] = []

    for chunk in chunks:
        extracciones = chunk.get("extracciones", [])
        if not extracciones:
            logger.debug("Chunk '%s' has no extractions, skipping.", chunk.get("chunk_id"))
            continue

        parts = []
        for ext in extracciones:
            cat = ext.get("categoria", "")
            sub = ext.get("subcategoria", "")
            parts.append(f"{cat} {sub}".strip())

        text = " | ".join(parts)
        texts.append(text)
        chunk_ids.append(str(chunk["chunk_id"]))

    logger.info(
        "Built texts for %d chunks (%d excluded for having no extractions).",
        len(texts),
        len(chunks) - len(texts),
    )
    return texts, chunk_ids


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------
def calcular_embeddings(texts: list[str]) -> np.ndarray:
    """Computes normalized embeddings for the given texts."""
    model = _get_model()
    logger.info("Encoding %d chunk texts...", len(texts))
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    logger.info("Encoding done. Shape: %s", embeddings.shape)
    return embeddings


# ---------------------------------------------------------------------------
# UMAP reduction
# ---------------------------------------------------------------------------
def reducir_umap(embeddings: np.ndarray) -> np.ndarray:
    """
    Applies UMAP to reduce embeddings to 2 dimensions.

    Parameters:
        embeddings: array of shape (n_samples, embedding_dim).

    Returns:
        Array of shape (n_samples, 2).
    """
    logger.info(
        "Applying UMAP (n_neighbors=%d, n_components=%d, metric='%s')...",
        UMAP_N_NEIGHBORS,
        UMAP_N_COMPONENTS,
        UMAP_METRIC,
    )
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        metric=UMAP_METRIC,
        random_state=RANDOM_SEED,
    )
    coords_2d = reducer.fit_transform(embeddings)
    logger.info("UMAP reduction done. Output shape: %s", coords_2d.shape)
    return coords_2d


# ---------------------------------------------------------------------------
# HDBSCAN clustering
# ---------------------------------------------------------------------------
def clusterizar_hdbscan(coords_2d: np.ndarray) -> np.ndarray:
    """
    Applies HDBSCAN on the 2D coordinates to find clusters.

    Returns:
        Array of cluster labels (int). -1 = noise.
    """
    logger.info("Applying HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(coords_2d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info(
        "HDBSCAN found %d clusters and %d noise points.", n_clusters, n_noise
    )
    return labels


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------
def generar_json_clusters(
    chunk_ids: list[str],
    labels: np.ndarray,
    output_path: str,
) -> None:
    """Writes the cluster assignments to a JSON file."""
    cluster_map: dict[int, list[str]] = {}
    for cid, label in zip(chunk_ids, labels):
        label_int = int(label)
        cluster_map.setdefault(label_int, []).append(cid)

    clusters_list = [
        {"cluster_id": k, "chunk_ids": v}
        for k, v in sorted(cluster_map.items())
    ]

    n_clusters = len([c for c in clusters_list if c["cluster_id"] != -1])

    output = {
        "n_clusters": n_clusters,
        "clusters": clusters_list,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    logger.info("Cluster JSON written to '%s'.", output_path)


def generar_grafico(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    chunk_ids: list[str],
    output_path: str,
) -> None:
    """Generates a scatter plot of the UMAP 2D projection, colored by cluster."""
    unique_labels = sorted(set(labels))
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(len(unique_labels), 1))

    fig, ax = plt.subplots(figsize=(14, 10))

    for label in unique_labels:
        mask = labels == label
        color = "lightgray" if label == -1 else cmap(label)
        label_str = "Ruido" if label == -1 else f"Cluster {label}"
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[color],
            label=label_str,
            s=60,
            alpha=0.8,
            edgecolors="k",
            linewidths=0.5,
        )

    for i, cid in enumerate(chunk_ids):
        ax.annotate(
            cid,
            (coords_2d[i, 0], coords_2d[i, 1]),
            fontsize=7,
            ha="center",
            va="bottom",
            textcoords="offset points",
            xytext=(0, 5),
        )

    ax.set_title("UMAP + HDBSCAN - Agrupacion tematica de chunks", fontsize=14)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        title="Clusters",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Debug plot saved to '%s'.", output_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    chunks = cargar_extracciones(INPUT_PATH)
    texts, chunk_ids = construir_textos(chunks)

    if len(texts) < 2:
        logger.error("Need at least 2 chunks with extractions to run UMAP. Aborting.")
        sys.exit(1)

    embeddings = calcular_embeddings(texts)
    coords_2d = reducir_umap(embeddings)
    labels = clusterizar_hdbscan(coords_2d)

    generar_json_clusters(chunk_ids, labels, OUTPUT_JSON_PATH)
    generar_grafico(coords_2d, labels, chunk_ids, OUTPUT_PLOT_PATH)

    print(f"\nResults:")
    print(f"  Clusters JSON: {OUTPUT_JSON_PATH}")
    print(f"  Debug plot:    {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()
