import os
from dotenv import load_dotenv

load_dotenv()

# Todoist & External APIs Configuration
TODOIST_API_KEY = os.getenv("TODOIST_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Ollama LLM Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:cloud")
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

# Memory & Session Settings
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "12"))
DEFAULT_SESSION_ID = os.getenv("DEFAULT_SESSION_ID", "default")

BASE_SYSTEM_PROMPT = "You are Lily, a helpful and cheerful assistant girl."
