import base64
import functools
import io
import json
import os
import re
import time
from typing import Any

import requests
from langchain_core.messages import HumanMessage
from PIL import Image, ImageOps

from src.config import TOOL_IMAGE_MAX_DIMENSION

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
                    return f"Error executing {fn_name}: {e!s}"

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

    formatted: str

    # 1. String output
    if isinstance(result, str):
        formatted = result.strip()

    # 2. Pydantic BaseModel (if present)
    elif hasattr(result, "model_dump_json") and callable(result.model_dump_json):
        formatted = str(result.model_dump_json(indent=2))

    # 3. Dict or List -> Valid JSON
    elif isinstance(result, (dict, list)):
        try:
            formatted = json.dumps(result, ensure_ascii=False, indent=2)
        except Exception:
            formatted = str(result)

    # 4. Exception / Error
    elif isinstance(result, Exception):
        formatted = f"Tool Error: {result}"

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


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard token similarity between two text queries.

    Returns float from 0.0 (completely distinct) to 1.0 (identical set of tokens).
    Normalizes by lowercasing and splitting into alphanumeric word sets.
    """
    if not text1 or not text2:
        return 0.0

    tokens1 = set(re.findall(r"\w+", text1.lower()))
    tokens2 = set(re.findall(r"\w+", text2.lower()))

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    return len(intersection) / len(union)


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
}


def is_image_file(file_path: str) -> bool:
    """Check if the provided file path has a supported image extension."""
    if not file_path or not isinstance(file_path, str):
        return False
    _, ext = os.path.splitext(file_path.strip().lower())
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def encode_image_to_base64(
    image_source: str | bytes,
    max_dimension: int = TOOL_IMAGE_MAX_DIMENSION,
    target_format: str = "PNG",
) -> str:
    """Safely load, downscale proportionally, and encode an image to Base64 string.

    Features:
    - Accepts local file path or raw bytes.
    - Resolves EXIF orientation (e.g. rotated mobile photos).
    - Proportional downscaling if width or height exceeds max_dimension.
    - Defensive: raises descriptive ValueError or FileNotFoundError if unreadable.
    """
    if not image_source:
        raise ValueError("Image source cannot be empty.")

    try:
        if isinstance(image_source, (str, os.PathLike)):
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image file not found: {image_source}")
            img = Image.open(image_source)
        elif isinstance(image_source, bytes):
            img = Image.open(io.BytesIO(image_source))
        else:
            raise TypeError(f"Unsupported image_source type: {type(image_source)}")

        # Handle EXIF orientation
        img = ImageOps.exif_transpose(img)

        # Proportional resize guard
        width, height = img.size
        if width > max_dimension or height > max_dimension:
            scale = min(max_dimension / width, max_dimension / height)
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Mode normalization
        fmt = target_format.upper()
        if fmt in ["JPEG", "JPG"]:
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
        elif img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        if isinstance(e, (FileNotFoundError, TypeError)):
            raise
        raise ValueError(f"Failed to process and encode image: {e}") from e


def build_multimodal_human_message(
    text: str,
    images: list[str | bytes] | None = None,
    max_dimension: int = TOOL_IMAGE_MAX_DIMENSION,
) -> HumanMessage:
    """Construct a LangChain HumanMessage compliant with Ollama multimodal input format.

    - If no images provided, returns simple text HumanMessage.
    - If images provided, returns content blocks: text + image_url blocks.
    - Accepts either raw file paths, base64 strings, or raw image bytes.
    """
    clean_text = (text or "").strip()
    if not images:
        return HumanMessage(content=clean_text)

    content_blocks: list[dict[str, Any]] = []
    if clean_text:
        content_blocks.append({"type": "text", "text": clean_text})

    for img in images:
        if not img:
            continue
        # If already a raw base64 string
        if isinstance(img, str) and not os.path.exists(img) and len(img) > 100:
            b64_str = img
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
        else:
            b64_str = encode_image_to_base64(img, max_dimension=max_dimension)

        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_str}"},
        })

    return HumanMessage(content=content_blocks)



