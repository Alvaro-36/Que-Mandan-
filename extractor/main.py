import os
import sys
import json
import time
import logging
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extractor.extractor import process_all_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_input_chunks(input_path: str = "chunks_debug.json") -> List[Dict[str, Any]]:
    """
    Loads chunks from the specified JSON file. If not found, generates them using chunker.py.
    """
    if not os.path.exists(input_path):
        logger.info("'%s' not found. Generating chunks via chunker.py...", input_path)
        from chunker import load_json_to_dataframe, chunkear_conversacion, exportar_chunks_a_json
        from main import calculate_activity_kde, calculate_median_message_distance

        json_path = "chatWPP.json"
        df = load_json_to_dataframe(json_path)
        density_vector = calculate_activity_kde(json_path, show_plot=False)
        mediana = calculate_median_message_distance(json_path, ignore_zeros=False)
        df_chunked = chunkear_conversacion(df, density_vector, mediana)
        input_path = exportar_chunks_a_json(df_chunked, input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info("Loaded %d chunks from '%s'.", len(chunks), input_path)
    return chunks


def save_output_json(pipeline_output: dict, output_path: str = "extracciones_salida.json") -> str:
    """
    Saves the final pipeline output dictionary to JSON.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_output, f, ensure_ascii=False, indent=4)
    logger.info("Saved pipeline extractions output to '%s'.", output_path)
    return output_path


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    logger.info("=== Starting Knowledge Extraction Pipeline (Step 1) ===")

    # 1. Load input chunks
    chunks = load_input_chunks("chunks_debug.json")

    # 2. Process all chunks through the pipeline using MiniMax-M3 (Nebius API)
    start_time = time.time()
    output = process_all_chunks(chunks, model_name="MiniMaxAI/MiniMax-M3")
    elapsed_seconds = time.time() - start_time

    # 3. Convert Pydantic output model to dict
    output_dict = output.model_dump()

    # 4. Save final output JSON
    output_file = save_output_json(output_dict, "extracciones_salida.json")

    # 5. Display summary
    total_chunks = len(output.chunks_procesados)
    total_extracciones = sum(len(c.extracciones) for c in output.chunks_procesados)
    chunks_con_extracciones = [c for c in output.chunks_procesados if len(c.extracciones) > 0]
    chunks_sin_extracciones = [c for c in output.chunks_procesados if len(c.extracciones) == 0]

    # Format time display
    if elapsed_seconds < 60:
        time_str = f"{elapsed_seconds:.2f} segundos"
    else:
        mins = int(elapsed_seconds // 60)
        secs = elapsed_seconds % 60
        time_str = f"{mins} min {secs:.2f} seg ({elapsed_seconds:.2f} segundos)"

    # Map input chunks by chunk_id for text display
    chunk_dict_map = {str(chk.get("chunk_id", i)): chk for i, chk in enumerate(chunks)}

    print("\n" + "=" * 60)
    print("RESUMEN DE EXTRACCION DE CONOCIMIENTO (PASO 1)")
    print("=" * 60)
    print(f"Tiempo total de ejecucion: {time_str}")
    print(f"Total de chunks procesados: {total_chunks}")
    print(f"Chunks con extracciones validas: {len(chunks_con_extracciones)}")
    print(f"Chunks sin extracciones validas: {len(chunks_sin_extracciones)}")
    print(f"Total de hechos/extracciones extraidos: {total_extracciones}")
    print(f"Archivo de salida generado: {output_file}")
    print("=" * 60)

    if chunks_sin_extracciones:
        print(f"\nCHUNKS SIN EXTRACCIONES VALIDAS ({len(chunks_sin_extracciones)}):")
        print("-" * 60)
        for c in chunks_sin_extracciones:
            cid = c.chunk_id
            orig_chunk = chunk_dict_map.get(cid, {})
            mensajes = orig_chunk.get("mensajes", [])
            print(f"\n--- Chunk ID: {cid} ({len(mensajes)} mensajes) ---")
            for m in mensajes:
                autor = m.get("autor", "")
                texto = m.get("texto", "")
                ts = m.get("timestamp", "")
                print(f"  [{ts}] {autor}: {texto}")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
