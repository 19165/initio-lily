import os

from dotenv import load_dotenv

load_dotenv()

# Todoist & External APIs Configuration
TODOIST_API_KEY = os.getenv("TODOIST_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Ollama LLM & Vision Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:cloud")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", OLLAMA_MODEL)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# PostgreSQL Database Configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "lily_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "lily_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lily_secret")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

# Memory, Session & Execution Settings
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "5"))
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "12"))
DEFAULT_SESSION_ID = os.getenv("DEFAULT_SESSION_ID", "default")

# Tool Execution & Context Safety Limits
TOOL_DOC_MAX_CHARS = int(os.getenv("TOOL_DOC_MAX_CHARS", "10000"))
TOOL_EXEC_MAX_CHARS = int(os.getenv("TOOL_EXEC_MAX_CHARS", "5000"))
TOOL_EXEC_TIMEOUT_SEC = int(os.getenv("TOOL_EXEC_TIMEOUT_SEC", "15"))
TOOL_IMAGE_MAX_DIMENSION = int(os.getenv("TOOL_IMAGE_MAX_DIMENSION", "1120"))

# Local Notes Knowledge Base
NOTES_DIR = os.getenv("NOTES_DIR", "notes")


BASE_SYSTEM_PROMPT = (
    "You are Lily, a helpful, sharp, and cheerful assistant girl. "
    "Format your responses using standard clean Markdown and Unicode characters (e.g. use '→', '•', '✓', '≈') "
    "instead of raw LaTeX math syntax (avoid '$\\rightarrow$', '$\\approx$', etc.). "
    "Always use native structured tool calling when executing tools. Never output raw pseudo-syntax like 'call:tool_name{...}' in conversation text."
)
