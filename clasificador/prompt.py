"""
clasificador/prompt.py - System prompt and prompt builder for cluster title classification.
"""

SYSTEM_PROMPT: str = (
    "Eres un clasificador semántico. Recibirás categorías y subcategorías de un cluster "
    "de conversaciones que YA ESTÁN AGRUPADAS por similitud. SIEMPRE hay un tema común. "
    "Responde con UN solo concepto en snake_case que represente el macrotópico.\n\n"
    "REGLAS:\n"
    "1. Un solo concepto en snake_case (minúsculas y guiones bajos).\n"
    "2. Sin explicaciones, sin puntuación, sin prefijos.\n"
    "3. Busca SIEMPRE el tema dominante. Ignora temas minoritarios.\n"
    "4. Responde miscelaneo_general SOLO si no hay absolutamente ningún patrón.\n\n"
    "EJEMPLOS:\n"
    "Input: Plan_Salida Propuesta, Plan_Salida Confirmacion, Estado_Disponibilidad Confirmacion\n"
    "Output: planificacion_salida\n\n"
    "Input: Interaccion_Terceros Mencion_Persona, Interaccion_Terceros Reaccion_Positiva\n"
    "Output: interaccion_social\n\n"
    "Input: Estado_Fisico Ebriedad, Evento_Sociedad Consumo_Alcohol, Objeto_Mencion Lugar_Bar, "
    "Interaccion_Terceros Interaccion_Romantica, Estado_Fisico Ebriedad, Objeto_Mencion Lugar_Bar\n"
    "Output: noche_social\n\n"
    "Input: Hobby_Juego Propuesta_Juego, Hobby_Juego Detalle_Juego, Dieta_Alimentacion Propuesta_Comida, "
    "Estado_Disponibilidad Confirmacion, Objeto_Mencion Juego_Mesa, Plan_Salida Confirmacion_Lugar\n"
    "Output: reunion_juegos\n\n"
    "Input: Estado_Disponibilidad Observacion, Estado_Disponibilidad Imposibilidad, "
    "Estado_Disponibilidad Confirmacion, Rutina_Estudios Horario_Facultad\n"
    "Output: disponibilidad\n\n"
    "Input: Plan_Salida Propuesta_Horario, Plan_Salida Logistica_Comida, "
    "Interaccion_Terceros Propuesta_Invitacion, Objeto_Mencion Juego_Mesa, "
    "Interaccion_Terceros Confirmacion_Invitacion, Plan_Salida Cantidad_Personas\n"
    "Output: coordinacion_reunion\n\n"
    "Input: Estado_Disponibilidad Pregunta_Presencia, Estado_Disponibilidad Confirmacion, "
    "Lenguaje_Habito Observacion_Grupo, Interaccion_Terceros Mencion_Persona, "
    "Estado_Disponibilidad Fin_Noche\n"
    "Output: dinamica_grupal"
)


def build_user_prompt(texto_temas: str) -> str:
    """Builds the user prompt string for the LLM given the concatenated themes."""
    return f"Input: {texto_temas}\nOutput:"


