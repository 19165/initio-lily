import os
from dotenv import load_dotenv

load_dotenv()

TODOIST_API_KEY = os.getenv("TODOIST_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:cloud")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = "You are Lily, a helpful and cheerful assistant girl."
