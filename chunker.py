"""
chunker.py - Hybrid temporal + semantic chunking of group conversations.

Segments a pre-treated message history (JSON) into discrete conversations
("chunks") by combining an adaptive temporal criterion with a semantic
rescue check via cosine similarity on sentence embeddings.

Dependencies: pandas, numpy, torch, sentence-transformers
Existing deps: main.calculate_peak_hour_coefficient, main.calculate_activity_kde,
               main.calculate_median_message_distance
"""

import json
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Import existing functions from main.py
from main import (
    calculate_activity_kde,
    calculate_median_message_distance,
    calculate_peak_hour_coefficient,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants (provisional, subject to empirical tuning)
# ---------------------------------------------------------------------------
UMBRAL_SIMILITUD_COSENO: float = 0.65       # HARDCODED - provisional
VENTANA_CONTEXTO: int = 15                   # Last N messages of current block used as context
MODELO_EMBEDDING: str = "paraphrase-multilingual-MiniLM-L12-v2"

# Base temporal cut threshold (seconds) for conversation inactivity
T_CORTE_BASE_SEG: float = 45 * 60    # 45 minutes base

# T_corte clamping range (seconds)
T_CORTE_MIN_SEG: float = 15 * 60    # 15 minutes
T_CORTE_MAX_SEG: float = 2 * 3600   # 2 hours

# Maximum time gap allowed for semantic rescue (seconds)
DELTA_MAX_RESCATE_SEG: float = 24 * 3600  # 24 hours

# Context strategy: "centroide" (mean pooling) or "max_sim" (max individual similarity)
ESTRATEGIA_CONTEXTO: str = "centroide"

# Minimum dataset size for reliable median
MIN_MENSAJES_MEDIANA: int = 30

# Placeholder patterns to exclude from centroid computation
PATRONES_EXCLUIR_CENTROIDE = {"<Media omitted>"}

# Seed for determinism
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Column mapping: JSON field -> DataFrame column
# ---------------------------------------------------------------------------
# The pre-treated JSON uses: {"dateTime": str, "author": str, "message": str}
COLUMN_MAP = {
    "dateTime": "timestamp",
    "author": "autor",
    "message": "texto",
}

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_model_cache: Optional[SentenceTransformer] = None


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
# Data loading
# ---------------------------------------------------------------------------
def load_json_to_dataframe(json_path: str) -> pd.DataFrame:
    """
    Loads the pre-treated JSON into a pd.DataFrame, maps columns, sorts
    chronologically, and assigns sequential msg_id.

    Parameters:
        json_path: Path to the JSON file.

    Returns:
        DataFrame with columns: msg_id, timestamp, autor, texto.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df.rename(columns=COLUMN_MAP, inplace=True)

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce")

    # Log and drop rows with unparseable timestamps
    n_nat = df["timestamp"].isna().sum()
    if n_nat > 0:
        logger.warning("%d rows with unparseable timestamps dropped.", n_nat)
        df = df.dropna(subset=["timestamp"])

    # Sort chronologically and check for out-of-order timestamps
    df = df.sort_values("timestamp").reset_index(drop=True)
    diffs = df["timestamp"].diff()
    n_negative = (diffs < pd.Timedelta(0)).sum()
    if n_negative > 0:
        logger.warning(
            "%d negative time deltas found after sorting (duplicate timestamps are OK).",
            n_negative,
        )

    # Assign sequential msg_id
    df.insert(0, "msg_id", range(len(df)))

    # Fill empty author strings
    df["autor"] = df["autor"].fillna("").astype(str)
    df["texto"] = df["texto"].fillna("").astype(str)

    if len(df) < MIN_MENSAJES_MEDIANA:
        warnings.warn(
            f"Dataset has only {len(df)} messages (< {MIN_MENSAJES_MEDIANA}). "
            "Median-based thresholds may not be reliable.",
            stacklevel=2,
        )

    logger.info("Loaded %d messages from '%s'.", len(df), json_path)
    return df


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------
def _compute_embeddings(df: pd.DataFrame) -> torch.Tensor:
    """
    Pre-computes normalized embeddings for all messages in a single batched call.

    Parameters:
        df: DataFrame with 'texto' column.

    Returns:
        Tensor of shape (len(df), embedding_dim), L2-normalized.
    """
    model = _get_model()
    texts = df["texto"].tolist()

    logger.info("Encoding %d messages...", len(texts))
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    logger.info("Encoding done. Shape: %s", embeddings.shape)
    return embeddings


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------
def _mask_excluir_centroide(df: pd.DataFrame) -> np.ndarray:
    """
    Returns a boolean mask (True = include in centroid computation).
    Excludes: system messages (empty author), multimedia placeholders.
    """
    is_system = df["autor"] == ""
    is_placeholder = df["texto"].isin(PATRONES_EXCLUIR_CENTROIDE)
    return ~(is_system | is_placeholder)


def _similitud_contexto(
    embeddings: torch.Tensor,
    mask_centroide: np.ndarray,
    inicio_ventana: int,
    idx_candidato: int,
) -> float:
    """
    Computes cosine similarity between the context centroid and a candidate message.

    The context centroid is the mean of normalized embeddings of the last
    <=VENTANA_CONTEXTO messages in the current block (excluding multimedia
    and system messages). The centroid is re-normalized before computing
    the dot product.

    Why centroid and not string concatenation: the max_seq_length of
    paraphrase-multilingual-MiniLM-L12-v2 is 128 tokens; concatenating
    15 messages would silently truncate and lose almost all context.

    Parameters:
        embeddings: Pre-computed normalized embeddings tensor.
        mask_centroide: Boolean array indicating which rows to include in centroid.
        inicio_ventana: Start index of the context window (inclusive).
        idx_candidato: Index of the candidate message.

    Returns:
        Cosine similarity as a Python float.
    """
    # Get eligible indices in the context window
    window_mask = mask_centroide[inicio_ventana:idx_candidato]
    eligible_indices = np.where(window_mask)[0] + inicio_ventana

    if len(eligible_indices) == 0:
        # No eligible context messages; fall back to raw embedding comparison
        # Use the single closest message even if it's a placeholder
        eligible_indices = np.arange(inicio_ventana, idx_candidato)
        if len(eligible_indices) == 0:
            return 0.0

    context_embeddings = embeddings[eligible_indices]

    if ESTRATEGIA_CONTEXTO == "centroide":
        centroid = context_embeddings.mean(dim=0, keepdim=True)
        # Re-normalize the centroid
        centroid = torch.nn.functional.normalize(centroid, p=2, dim=1)
        # Dot product on normalized vectors = cosine similarity
        sim = torch.mm(centroid, embeddings[idx_candidato].unsqueeze(1)).item()
    elif ESTRATEGIA_CONTEXTO == "max_sim":
        # Maximum individual similarity against any message in the window
        sims = torch.mv(context_embeddings, embeddings[idx_candidato])
        sim = sims.max().item()
    else:
        raise ValueError(f"Unknown context strategy: {ESTRATEGIA_CONTEXTO}")

    return float(sim)


# ---------------------------------------------------------------------------
# Main chunking algorithm
# ---------------------------------------------------------------------------
def chunkear_conversacion(
    df: pd.DataFrame,
    density_vector: np.ndarray,
    mediana_delta_seg: float,
) -> pd.DataFrame:
    """
    Segments a chronologically ordered DataFrame of messages into conversation
    chunks using a hybrid temporal + semantic criterion.

    Algorithm (two passes):
    1. Vectorized pass: compute delta_t, T_corte per row, mark temporal cut candidates.
    2. Sequential pass (only over candidates): check semantic similarity against
       the context centroid; confirm or rescue cuts.

    T_corte(h) = mediana(delta_t) / coef_hora_pico(h)
    where coef_hora_pico is INVERTED (1/coef) so that peak hours yield a
    larger divisor and thus a smaller T_corte.

    Parameters:
        df: DataFrame with columns msg_id, timestamp, autor, texto.
            Must be sorted by timestamp ascending with reset index.
        density_vector: Pre-computed KDE density vector from calculate_activity_kde.
        mediana_delta_seg: Pre-computed median of inter-message distances in seconds.

    Returns:
        DataFrame with additional traceability columns:
        chunk_id, delta_seg, coef_hora_pico, t_corte_seg,
        es_candidato_corte, sim_coseno, corte_confirmado, motivo.
    """
    n = len(df)
    if n == 0:
        return df

    # ---- Pass 1: Vectorized computation ----
    logger.info("Pass 1: Vectorized temporal computation...")

    # Compute delta_t in seconds
    delta = df["timestamp"].diff()
    df["delta_seg"] = delta.dt.total_seconds().fillna(0.0)

    # Compute peak hour coefficient for each row based on PREVIOUS message's hour
    # First message has no previous, use its own hour (irrelevant since it's never a candidate)
    hours = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    prev_hours = hours.shift(1).fillna(hours.iloc[0])

    coefs_raw = np.array([
        calculate_peak_hour_coefficient(h, density_vector)
        for h in prev_hours
    ])

    # INVERT the coefficient: 1/coef
    # Original coef: peak_hour -> 0.5, valley -> 2.5
    # Inverted:      peak_hour -> 2.0, valley -> 0.4
    # T_corte = mediana / coef_inverted: peak -> small, valley -> large
    assert np.all(coefs_raw > 0), "coef_hora_pico must be > 0 for all hours"
    coefs_inverted = 1.0 / coefs_raw

    df["coef_hora_pico"] = coefs_inverted

    # T_corte = T_CORTE_BASE_SEG / coef_inverted, clamped to [T_CORTE_MIN, T_CORTE_MAX]
    t_corte = T_CORTE_BASE_SEG / coefs_inverted
    t_corte = np.clip(t_corte, T_CORTE_MIN_SEG, T_CORTE_MAX_SEG)
    df["t_corte_seg"] = t_corte

    # Mark temporal cut candidates
    df["es_candidato_corte"] = df["delta_seg"] > df["t_corte_seg"]
    # First message is never a candidate
    df.loc[0, "es_candidato_corte"] = False

    n_candidates = df["es_candidato_corte"].sum()
    logger.info(
        "Pass 1 done: %d temporal cut candidates out of %d messages.",
        n_candidates,
        n,
    )

    # ---- Pre-compute embeddings ----
    embeddings = _compute_embeddings(df)
    mask_centroide = _mask_excluir_centroide(df)

    # ---- Pass 2: Sequential semantic check (only on candidates) ----
    logger.info("Pass 2: Sequential semantic check on %d candidates...", n_candidates)

    cortes = np.zeros(n, dtype=bool)
    sim_coseno = np.full(n, np.nan)
    motivo = np.full(n, None, dtype=object)
    inicio_bloque = 0

    candidate_indices = df.index[df["es_candidato_corte"]].tolist()

    for i in candidate_indices:
        delta_val = float(df.loc[i, "delta_seg"])
        if delta_val > DELTA_MAX_RESCATE_SEG:
            # Exceeds maximum rescue window: confirmed cut due to excessive temporal gap
            cortes[i] = True
            inicio_bloque = i
            motivo[i] = "corte_temporal_excesivo"
            continue

        ini_ventana = max(inicio_bloque, i - VENTANA_CONTEXTO)
        sim = _similitud_contexto(embeddings, mask_centroide, ini_ventana, i)
        sim_coseno[i] = sim

        if sim < UMBRAL_SIMILITUD_COSENO:
            # Confirmed semantic cut
            cortes[i] = True
            inicio_bloque = i
            motivo[i] = "corte_semantico"
        else:
            # Rescued by similarity
            motivo[i] = "rescatado_por_similitud"

    df["sim_coseno"] = sim_coseno
    df["corte_confirmado"] = cortes
    df["motivo"] = motivo

    # chunk_id via cumsum
    df["chunk_id"] = df["corte_confirmado"].cumsum().astype(int)

    n_confirmed = cortes.sum()
    n_rescued = n_candidates - n_confirmed
    pct_rescued = (n_rescued / n_candidates * 100) if n_candidates > 0 else 0.0

    logger.info(
        "Pass 2 done: %d confirmed cuts, %d rescued (%.1f%% rescue rate).",
        n_confirmed,
        n_rescued,
        pct_rescued,
    )

    if n_candidates > 0:
        if pct_rescued < 1.0:
            logger.warning(
                "Rescue rate is ~0%%: semantic threshold %.2f may be too LOW.",
                UMBRAL_SIMILITUD_COSENO,
            )
        elif pct_rescued > 99.0:
            logger.warning(
                "Rescue rate is ~100%%: semantic threshold %.2f may be too HIGH.",
                UMBRAL_SIMILITUD_COSENO,
            )

    return df


# ---------------------------------------------------------------------------
# Chunk collapse
# ---------------------------------------------------------------------------
def colapsar_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses the row-level DataFrame into a chunk-level summary.

    Parameters:
        df: DataFrame produced by chunkear_conversacion (must have chunk_id).

    Returns:
        DataFrame with one row per chunk:
        chunk_id, n_mensajes, ts_inicio, ts_fin, duracion, n_autores, texto_concatenado.
    """
    def _agg(group: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_mensajes": len(group),
            "ts_inicio": group["timestamp"].iloc[0],
            "ts_fin": group["timestamp"].iloc[-1],
            "duracion": group["timestamp"].iloc[-1] - group["timestamp"].iloc[0],
            "n_autores": group.loc[group["autor"] != "", "autor"].nunique(),
            "texto_concatenado": "\n".join(group["texto"].tolist()),
        })

    chunks = df.groupby("chunk_id", sort=True).apply(_agg, include_groups=False).reset_index()
    return chunks


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------
def _report_metrics(df: pd.DataFrame, chunks: pd.DataFrame, mediana: float) -> None:
    """Prints acceptance metrics to the logger."""
    n_chunks = len(chunks)
    sizes = chunks["n_mensajes"]

    logger.info("=" * 60)
    logger.info("ACCEPTANCE METRICS")
    logger.info("=" * 60)
    logger.info("1. Chunks generated: %d", n_chunks)
    logger.info(
        "   Size distribution: min=%d, median=%.0f, p90=%.0f, max=%d",
        sizes.min(),
        sizes.median(),
        sizes.quantile(0.90),
        sizes.max(),
    )

    n_candidates = df["es_candidato_corte"].sum()
    n_confirmed = df["corte_confirmado"].sum()
    n_rescued = n_candidates - n_confirmed
    pct_rescued = (n_rescued / n_candidates * 100) if n_candidates > 0 else 0.0
    logger.info(
        "2. Temporal candidates: %d | Confirmed cuts: %d | Rescued: %d (%.1f%%)",
        n_candidates,
        n_confirmed,
        n_rescued,
        pct_rescued,
    )
    if n_candidates > 0 and (pct_rescued < 1.0 or pct_rescued > 99.0):
        logger.warning(
            "   Rescue rate is extreme (%.1f%%). Threshold %.2f may be miscalibrated.",
            pct_rescued,
            UMBRAL_SIMILITUD_COSENO,
        )

    t_corte_vals = df.loc[df["es_candidato_corte"], "t_corte_seg"]
    if len(t_corte_vals) > 0:
        logger.info(
            "3. median(delta_t) = %.1f sec | T_corte effective range: [%.1f, %.1f] sec",
            mediana,
            t_corte_vals.min(),
            t_corte_vals.max(),
        )
    else:
        logger.info("3. median(delta_t) = %.1f sec | No candidates, no T_corte range.", mediana)

    logger.info("4. Sample chunk boundaries:")
    boundaries = df.index[df["corte_confirmado"]].tolist()
    sample_boundaries = boundaries[:5]
    for b_idx in sample_boundaries:
        start = max(0, b_idx - 2)
        end = min(len(df), b_idx + 3)
        logger.info("   --- Boundary at msg_id=%d ---", df.loc[b_idx, "msg_id"])
        for j in range(start, end):
            row = df.iloc[j]
            marker = ">>>" if j == b_idx else "   "
            logger.info(
                "   %s [%d] %s | %s: %s",
                marker,
                row["msg_id"],
                str(row["timestamp"]),
                row["autor"][:15] if row["autor"] else "(system)",
                row["texto"][:60],
            )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Debugging / Export utilities
# ---------------------------------------------------------------------------
def exportar_chunks_a_json(df: pd.DataFrame, output_path: str = "chunks_debug.json") -> str:
    """
    Exports the generated chunks and their contained messages into a structured JSON file.
    Note: Created specifically for debugging and manual inspection purposes.

    Parameters:
        df: DataFrame produced by chunkear_conversacion (must contain 'chunk_id').
        output_path: Destination JSON file path. Defaults to 'chunks_debug.json'.

    Returns:
        str: Output JSON file path.
    """
    chunks_list = []

    for chunk_id, group in df.groupby("chunk_id", sort=True):
        messages = []
        for _, row in group.iterrows():
            msg_dict = {
                "msg_id": int(row["msg_id"]),
                "timestamp": row["timestamp"].isoformat() if pd.notna(row["timestamp"]) else None,
                "autor": str(row["autor"]),
                "texto": str(row["texto"]),
            }
            # Add traceability metadata for debugging if available
            if "delta_seg" in row and pd.notna(row["delta_seg"]):
                msg_dict["delta_seg"] = float(row["delta_seg"])
            if "t_corte_seg" in row and pd.notna(row["t_corte_seg"]):
                msg_dict["t_corte_seg"] = float(row["t_corte_seg"])
            if "sim_coseno" in row and pd.notna(row["sim_coseno"]):
                msg_dict["sim_coseno"] = float(row["sim_coseno"])
            if "motivo" in row and row["motivo"] is not None:
                msg_dict["motivo"] = str(row["motivo"])

            messages.append(msg_dict)

        chunk_data = {
            "chunk_id": int(chunk_id),
            "n_mensajes": len(group),
            "ts_inicio": group["timestamp"].iloc[0].isoformat() if pd.notna(group["timestamp"].iloc[0]) else None,
            "ts_fin": group["timestamp"].iloc[-1].isoformat() if pd.notna(group["timestamp"].iloc[-1]) else None,
            "mensajes": messages,
        }
        chunks_list.append(chunk_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_list, f, ensure_ascii=False, indent=4)

    logger.info("Exported %d chunks to '%s' (for debugging).", len(chunks_list), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    JSON_PATH = "chatWPP.json"

    # Step 1: Load data
    df = load_json_to_dataframe(JSON_PATH)

    # Step 2: Pre-compute KDE density vector and median
    density_vector = calculate_activity_kde(JSON_PATH, show_plot=False)
    mediana = calculate_median_message_distance(JSON_PATH, ignore_zeros=False)
    logger.info("Global median(delta_t) = %.1f seconds", mediana)

    # Step 3: Chunk the conversation
    df_chunked = chunkear_conversacion(df, density_vector, mediana)

    # Step 4: Collapse to chunk-level summary
    chunks = colapsar_chunks(df_chunked)

    # Step 5: Report metrics
    _report_metrics(df_chunked, chunks, mediana)

    # Export chunks to JSON for debugging purposes
    debug_json_path = exportar_chunks_a_json(df_chunked, "chunks_debug.json")
    print(f"\nDebug JSON file exported: {debug_json_path}")

    # Print first 10 chunks summary
    print("\n--- First 10 chunks ---")
    for _, row in chunks.head(10).iterrows():
        print(
            f"  Chunk {row['chunk_id']:3d} | "
            f"{row['n_mensajes']:3d} msgs | "
            f"{row['ts_inicio']} -> {row['ts_fin']} | "
            f"{row['n_autores']} authors | "
            f"dur={row['duracion']}"
        )

