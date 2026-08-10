import json
import re
import logging
import json_repair

logger = logging.getLogger(__name__)


def clean_llm_response(raw_text: str) -> str:
    """
    Cleans markdown code fences or surrounding non-JSON commentary from the response.
    Handles prompt echo from LLaMA models.
    """
    text = raw_text.strip()

    # If LLM echoed back prompt instruction tags, extract completion after [/INST]
    if "[/INST]" in text:
        text = text.rsplit("[/INST]", 1)[-1].strip()

    # Remove markdown codeblocks ```json ... ``` or ``` ... ```
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    else:
        # Extract outer JSON object {...}
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1].strip()

    return text


def parse_json_response(raw_response: str) -> dict:
    """
    Parses a string response into a Python dictionary.
    First tries standard json.loads. If it fails, attempts repair via json_repair.

    Parameters:
        raw_response (str): Raw string returned by the LLM.

    Returns:
        dict: Parsed JSON object as a dictionary.

    Raises:
        ValueError: If JSON cannot be parsed or repaired.
    """
    cleaned_text = clean_llm_response(raw_response)

    # Attempt 1: Standard json.loads
    try:
        data = json.loads(cleaned_text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug("Standard json.loads failed: %s. Attempting json_repair...", str(e))

    # Attempt 2: json_repair
    try:
        repaired = json_repair.loads(cleaned_text)
        if isinstance(repaired, dict):
            logger.info("JSON successfully repaired using json_repair.")
            return repaired
    except Exception as err:
        logger.debug("json_repair failed: %s", str(err))

    raise ValueError(f"Failed to parse or repair JSON from response: {raw_response[:100]}...")
