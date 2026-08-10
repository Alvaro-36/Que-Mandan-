import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

NEBIUS_BASE_URL = "https://api.tokenfactory.us-central1.nebius.com/v1/"
DEFAULT_MODEL = "MiniMaxAI/MiniMax-M3"


_CLIENT = None


def _get_client() -> OpenAI:
    """
    Creates and returns a singleton OpenAI-compatible client configured for Nebius API.
    Reuses connection pool across parallel threads.
    """
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NEBIUS_API_KEY not found in environment. "
                "Make sure your .env file contains: NEBIUS_API_KEY=your_key_here"
            )
        _CLIENT = OpenAI(base_url=NEBIUS_BASE_URL, api_key=api_key)
    return _CLIENT


def query_llm(prompt: str, model_name: str = DEFAULT_MODEL, system_prompt: str = "") -> str:
    """
    Sends a chat completion request to the Nebius API (MiniMax-M3) using the OpenAI-compatible endpoint.

    Parameters:
        prompt (str): User prompt content.
        model_name (str): Model identifier on Nebius. Defaults to 'MiniMaxAI/MiniMax-M3'.
        system_prompt (str): Optional system prompt instruction.

    Returns:
        str: Raw text completion from the assistant.
    """
    try:
        client = _get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        })

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=4000,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error("Nebius API call failed for model '%s': %s", model_name, str(e))
        raise RuntimeError(f"Nebius API error: {e}") from e
