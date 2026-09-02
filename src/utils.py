import functools
import json
import re
import time
from typing import Any

import requests

try:
    import httpx
    HTTPX_EXCEPTIONS = (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError)
except ImportError:
    HTTPX_EXCEPTIONS = ()

try:
    import httpcore
    HTTPCORE_EXCEPTIONS = (httpcore.ConnectError, httpcore.TimeoutException, httpcore.NetworkError)
except ImportError:
    HTTPCORE_EXCEPTIONS = ()

CANDIDATE_EXCEPTIONS = [
    ConnectionResetError,
    TimeoutError,
    getattr(requests.exceptions, "ConnectionError", Exception),
    getattr(requests.exceptions, "Timeout", Exception),
    *HTTPX_EXCEPTIONS,
    *HTTPCORE_EXCEPTIONS,
]

RETRIABLE_EXCEPTIONS = tuple(
    exc
    for exc in CANDIDATE_EXCEPTIONS
    if isinstance(exc, type) and issubclass(exc, BaseException)
)


def retry_on_network_error(max_retries: int = 5, initial_delay: float = 0.5):
    """Decorator สำหรับ retry เงียบๆ อัตโนมัติเมื่อเกิด Transient Network Glitch หรือ WinError 10054"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RETRIABLE_EXCEPTIONS as e:
                    if attempt == max_retries:
                        return (
                            f"Network Error: Failed after {max_retries} attempts ({e})"
                        )
                    time.sleep(delay)
                    delay *= 2
                except Exception as e:
                    if "10054" in str(e) or "forcibly closed" in str(e).lower():
                        if attempt == max_retries:
                            return f"Network Error: Failed after {max_retries} attempts ({e})"
                        time.sleep(delay)
                        delay *= 2
                        continue
                    fn_name = getattr(func, "__name__", str(func))
                    return f"Error executing {fn_name}: {str(e)}"

        return wrapper

    return decorator


def format_tool_result(result: Any, max_chars: int = 4000) -> str:
    """Safely format and normalize any tool execution result for the LLM context.

    Features:
        - Converts dict/list to clean formatted JSON strings.
        - Supports Pydantic models.
        - Handles exceptions and None gracefully.
        - Enforces max_chars token guard to prevent context overflow.
    """
    if result is None:
        return "Success (No output returned)."

    # 1. String output
    if isinstance(result, str):
        formatted = result.strip()

    # 2. Pydantic BaseModel (if present)
    elif hasattr(result, "model_dump_json") and callable(result.model_dump_json):
        formatted = result.model_dump_json(indent=2)

    # 3. Dict or List -> Valid JSON
    elif isinstance(result, (dict, list)):
        try:
            formatted = json.dumps(result, ensure_ascii=False, indent=2)
        except Exception:
            formatted = str(result)

    # 4. Exception / Error
    elif isinstance(result, Exception):
        formatted = f"Tool Error: {str(result)}"

    # 5. Fallback for other data types
    else:
        formatted = str(result)

    # 🛡️ Token Guard: Truncate if output exceeds safety limit
    if len(formatted) > max_chars:
        truncated_len = len(formatted) - max_chars
        formatted = (
            formatted[:max_chars]
            + f"\n\n...[Truncated {truncated_len} characters to protect context limit]..."
        )

    return formatted


LATEX_REPLACEMENTS = [
    # Arrows
    (r"\$*\\rightarrow\$*|\$*\\to\$*", "→"),
    (r"\$*\\leftarrow\$*", "←"),
    (r"\$*\\Rightarrow\$*", "⇒"),
    (r"\$*\\Leftarrow\$*", "⇐"),
    (r"\$*\\leftrightarrow\$*", "↔"),
    (r"\$*\\Leftrightarrow\$*", "⇔"),
    (r"\$*\\uparrow\$*", "↑"),
    (r"\$*\\downarrow\$*", "↓"),
    # Math & Common Symbols
    (r"\$*\\times\$*", "×"),
    (r"\$*\\div\$*", "÷"),
    (r"\$*\\approx\$*", "≈"),
    (r"\$*\\neq\$*|\$*\\ne\$*", "≠"),
    (r"\$*\\leq\$*|\$*\\le\$*", "≤"),
    (r"\$*\\geq\$*|\$*\\ge\$*", "≥"),
    (r"\$*\\pm\$*", "±"),
    (r"\$*\\cdot\$*", "·"),
    (r"\$*\\bullet\$*", "•"),
    (r"\$*\\degree\$*|\$*\\circ\$*", "°"),
    (r"\$*\\infty\$*", "∞"),
    (r"\$*\\sqrt\{([^}]+)\}\$*", r"√(\1)"),
    (r"\$*\\frac\{([^}]+)\}\{([^}]+)\}\$*", r"(\1/\2)"),
    # Standalone math tags
    (r"\$\s*\$", ""),
]


def clean_markdown_text(text: str) -> str:
    """Clean and sanitize LLM text output by converting LaTeX math symbols into clean Unicode.

    Prevents ugly raw strings like '$\\rightarrow$' on terminal and markdown viewers.
    """
    if not text:
        return ""

    cleaned = text
    for pattern, replacement in LATEX_REPLACEMENTS:
        cleaned = re.sub(pattern, replacement, cleaned)

    return cleaned

