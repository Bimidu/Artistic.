import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

_client = None


def get_openai_client():
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. "
            "Add it to your .env file."
        )

    _client = OpenAI(api_key=api_key)
    return _client