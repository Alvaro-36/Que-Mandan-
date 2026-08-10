from typing import List
from pydantic import BaseModel, Field, field_validator


class Extraction(BaseModel):
    """
    Represents a single structured fact extracted from a conversation chunk.
    """
    usuario: str = Field(default="", description="Nombre del usuario que emite o sobre el cual trata el hecho")
    categoria: str = Field(default="General", description="Categoria del hecho extraido (ej: Vehiculos, Gustos, Eventos, Trabajo)")
    subcategoria: str = Field(default="", description="Subcategoria especifica del hecho")
    iniciador: str = Field(default="", description="Nombre del usuario que inicia la conversacion o tema")
    entidad: str = Field(default="", description="Entidad o concepto principal involucrado")
    atributo_o_valor: str = Field(default="", description="Atributo, estado, preferencia o valor asociado a la entidad")
    polaridad: str = Field(default="neutra", description="Polaridad del hecho: positiva, negativa, neutra")
    certeza: str = Field(default="alta", description="Grado de certeza: alta, media, baja, hipotetica")
    evidencia: str = Field(default="", description="Cita textual exacta que respalda el hecho dentro del texto del chunk")


class ChunkExtraction(BaseModel):
    """
    Container for all extractions associated with a specific chunk.
    """
    chunk_id: str
    extracciones: List[Extraction] = Field(default_factory=list)

    @field_validator("chunk_id", mode="before")
    @classmethod
    def coerce_chunk_id_to_str(cls, v):
        return str(v) if v is not None else ""


class PipelineOutput(BaseModel):
    """
    Final output structure of the extraction pipeline.
    """
    chunks_procesados: List[ChunkExtraction] = Field(default_factory=list)
