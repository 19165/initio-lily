import io
import os
import subprocess
import sys
import tempfile

import ollama
import pandas as pd
from PIL import Image
from langchain_core.tools import tool
from pypdf import PdfReader

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_VISION_MODEL,
    TOOL_DOC_MAX_CHARS,
    TOOL_EXEC_MAX_CHARS,
    TOOL_EXEC_TIMEOUT_SEC,
    TOOL_IMAGE_MAX_DIMENSION,
)
from src.utils import retry_on_network_error


@tool
def read_document(file_path: str) -> str:
    """Read and extract text from a local document (PDF, CSV, Excel, TXT, MD, JSON).

    Args:
        file_path: Absolute or relative path to the file.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    ext = file_path.lower().split(".")[-1]
    try:
        if ext == "pdf":
            reader = PdfReader(file_path)
            text = "\n".join(
                [page.extract_text() for page in reader.pages if page.extract_text()]
            )
            return text[:TOOL_DOC_MAX_CHARS] + (
                "\n...[TRUNCATED]" if len(text) > TOOL_DOC_MAX_CHARS else ""
            )

        elif ext in ["csv", "txt", "md", "json", "py"]:
            if ext == "csv":
                df = pd.read_csv(file_path)
                text = df.head(100).to_markdown()  # Show schema and top 100 rows
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            return text[:TOOL_DOC_MAX_CHARS] + (
                "\n...[TRUNCATED]" if len(text) > TOOL_DOC_MAX_CHARS else ""
            )

        elif ext in ["xls", "xlsx"]:
            df = pd.read_excel(file_path)
            text = df.head(100).to_markdown()
            return text[:TOOL_DOC_MAX_CHARS] + (
                "\n...[TRUNCATED]" if len(text) > TOOL_DOC_MAX_CHARS else ""
            )
        else:
            return f"Error: Unsupported file format '.{ext}'"
    except Exception as e:
        return f"Failed to read file: {str(e)}"


@tool
def execute_python(code: str) -> str:
    """Execute Python code in a local subprocess and return the standard output.
    Use this for math calculations, data analysis, or complex logic.
    Use `print()` to output the result you want to see.

    Args:
        code: The Python script to execute.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=TOOL_EXEC_TIMEOUT_SEC,
        )

        output = result.stdout
        if result.stderr:
            output += f"\n--- STDERR ---\n{result.stderr}"

        if not output.strip():
            return "Execution completed successfully, but no output was printed. Did you forget to use print()?"

        return output[:TOOL_EXEC_MAX_CHARS] + (
            "\n...[TRUNCATED]" if len(output) > TOOL_EXEC_MAX_CHARS else ""
        )
    except subprocess.TimeoutExpired:
        return f"Error: Python execution timed out after {TOOL_EXEC_TIMEOUT_SEC} seconds."
    except Exception as e:
        return f"Error executing Python: {str(e)}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _preprocess_image(
    image_path: str, max_dimension: int = TOOL_IMAGE_MAX_DIMENSION
) -> bytes:
    """Resize image if dimensions exceed max_dimension and return JPEG/PNG bytes."""
    with Image.open(image_path) as img:
        img_format = img.format if img.format in ["JPEG", "PNG", "WEBP"] else "JPEG"

        if img.mode in ("RGBA", "LA", "P") and img_format == "JPEG":
            img = img.convert("RGB")

        width, height = img.size
        if max(width, height) > max_dimension:
            scale = max_dimension / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format=img_format, quality=85)
        return buffer.getvalue()


@tool
@retry_on_network_error(max_retries=3)
def inspect_image(
    image_path: str,
    query: str = "Describe this image in detail and transcribe any visible text or data.",
) -> str:
    """Inspect and analyze a local image file using the Multimodal Vision Model.
    Use this tool whenever you need to see, read, or reason about an image (e.g. diagrams, charts, chessboards, photos).

    Args:
        image_path: Absolute or relative path to the image file (.png, .jpg, .jpeg, .webp, .bmp, .gif).
        query: Specific question or instruction regarding what to look for or calculate from the image.
    """
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"

    try:
        image_bytes = _preprocess_image(image_path)

        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=OLLAMA_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": query,
                    "images": [image_bytes],
                }
            ],
        )
        return response.get("message", {}).get(
            "content", "No response returned from vision model."
        )
    except Exception as e:
        return f"Failed to inspect image with model '{OLLAMA_VISION_MODEL}': {str(e)}"
