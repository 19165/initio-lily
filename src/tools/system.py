from datetime import datetime
from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Get the current date and time formatted as a string."""
    return datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
