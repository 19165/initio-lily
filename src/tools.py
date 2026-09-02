from datetime import datetime
from typing import Optional
from langchain_core.tools import tool
from todoist_api_python.api import TodoistAPI
from tavily import TavilyClient
from src.config import TODOIST_API_KEY, TAVILY_API_KEY
from src.utils import retry_on_network_error
from src.db import upsert_fact, get_all_facts, delete_fact

import os
import subprocess
import tempfile
import pandas as pd
from pypdf import PdfReader


todoist = TodoistAPI(token=TODOIST_API_KEY) if TODOIST_API_KEY else None
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


# -------------------------------------------------------------
# 1. System / Time Tools
# -------------------------------------------------------------
@tool
def get_current_time() -> str:
    """Get the current date and time formatted as a string."""
    return datetime.now().strftime("%d-%m-%Y, %H:%M:%S")


# -------------------------------------------------------------
# 2. Todoist Integration Tools
# -------------------------------------------------------------
@tool
@retry_on_network_error(max_retries=5)
def add_task(
    content: str,
    due_string: str = "",
    description: str = "",
) -> str:
    """Create a new task in Todoist.

    Args:
        content: the title/name of the task (e.g. 'Buy Milk').
        due_string: Natural language due date (e.g. 'tomorrow at 5pm', 'next Monday', 'today').
        description: Optional detailed description for the task.
    """
    if not todoist:
        return "Todoist API key is missing. Please configure TODOIST_API_KEY in .env."

    kwargs = {"content": content}
    if due_string:
        kwargs["due_string"] = due_string
    if description:
        kwargs["description"] = description

    task = todoist.add_task(**kwargs)
    due_info = f", Due: {task.due.string}" if task.due else ""
    return f"Successfully created task '{task.content}' (ID: {task.id}{due_info})"


@tool
@retry_on_network_error(max_retries=5)
def get_tasks(filter_query: str = "today | overdue") -> str:
    """Get active tasks from Todoist based on a filter or query.

    Args:
        filter_query: Todoist filter string (e.g. 'today', 'today | overdue', 'view all', '7 days').
    """
    if not todoist:
        return "Todoist API key is missing. Please configure TODOIST_API_KEY in .env."

    clean_query = filter_query.strip() if filter_query else ""
    tasks = []

    if clean_query and clean_query.lower() != "view all":
        for page in todoist.filter_tasks(query=clean_query):
            tasks.extend(page)
    else:
        for page in todoist.get_tasks():
            tasks.extend(page)

    if not tasks:
        return "No tasks found."

    result_lines = []
    for t in tasks:
        due = f" (Due: {t.due.string})" if t.due else ""
        result_lines.append(f"- [ID: {t.id}] {t.content}{due}")
    return "\n".join(result_lines)


@tool
@retry_on_network_error(max_retries=5)
def complete_task(task_id_or_title: str) -> str:
    """Mark a Todoist task as completed / done.

    Args:
        task_id_or_title: The unique task ID (e.g. '6hQ3Rf87jqRCFJMw')
                          OR the title/name of the task (e.g. 'จัดการราชาแห่งอินชอน').
    """
    if not todoist:
        return "Todoist API key is missing. Please configure TODOIST_API_KEY in .env."

    target_id = task_id_or_title.strip()
    target_title = task_id_or_title

    # Search in active tasks if task_id_or_title is a name
    all_tasks = []
    for page in todoist.get_tasks():
        all_tasks.extend(page)

    for t in all_tasks:
        if t.id == target_id or target_title.lower() in t.content.lower():
            target_id = t.id
            target_title = t.content
            break

    is_success = todoist.complete_task(task_id=target_id)
    if is_success:
        return f"Successfully completed task '{target_title}' (ID: {target_id})"
    return f"Failed to complete task '{target_title}' (ID: {target_id})"


# -------------------------------------------------------------
# 3. Persistent Memory / User Profile Tools
# -------------------------------------------------------------
@tool
def save_fact(key: str, value: str, category: str = "general") -> str:
    """Save or update a personal fact or preference about the user into persistent memory.

    Args:
        key: A short identifier for the fact (e.g. 'user_name', 'favorite_drink', 'job_title').
        value: The detailed fact or preference (e.g. 'Bas', 'Black coffee without sugar', 'Software Engineer').
        category: Optional category for grouping (e.g. 'personal', 'preferences', 'work').
    """
    try:
        upsert_fact(key=key, value=value, category=category)
        return f"Successfully remembered: [{category}] {key} = '{value}'"
    except Exception as e:
        return f"Failed to save fact: {str(e)}"


@tool
def get_facts(category: str = "") -> str:
    """Retrieve saved facts about the user from persistent memory.

    Args:
        category: Optional category filter (e.g. 'personal', 'preferences', 'work'). Leave empty to get all facts.
    """
    try:
        facts = get_all_facts(category=category if category.strip() else None)
        if not facts:
            return "No saved facts found."

        result_lines = []
        for f in facts:
            result_lines.append(f"- [{f['category']}] {f['key']}: {f['value']}")
        return "\n".join(result_lines)
    except Exception as e:
        return f"Failed to fetch facts: {str(e)}"


@tool
def delete_fact_by_key(key: str) -> str:
    """Remove a fact about the user from persistent memory.

    Args:
        key: The key of the fact to remove (e.g. 'favorite_drink').
    """
    try:
        deleted = delete_fact(key=key)
        if deleted:
            return f"Successfully forgot fact '{key}'."
        return f"Fact '{key}' was not found in memory."
    except Exception as e:
        return f"Failed to delete fact: {str(e)}"


# -------------------------------------------------------------
# 4. Real-time Web Search Tools (Tavily API)
# -------------------------------------------------------------
@tool
@retry_on_network_error(max_retries=3)
def search_web(
    query: str,
    topic: str = "general",
    max_results: int = 4,
) -> str:
    """Search the web for real-time information, news, facts, weather, or technical documentation.

    Args:
        query: Specific search keywords (e.g. 'Bitcoin price today', 'Bangkok weather', 'Python 3.13 changelog').
        topic: Choose 'general' for general queries/facts, or 'news' for recent breaking news.
        max_results: Number of search results to fetch (default: 4).
    """
    if not tavily_client:
        return "Tavily API key is missing. Please configure TAVILY_API_KEY in .env."

    try:
        clean_topic = (
            topic.strip().lower() if topic in ["general", "news"] else "general"
        )
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            topic=clean_topic,
            max_results=max_results,
            include_answer=True,
            include_raw_content=False,
        )

        results = []

        # 1. Direct answer summary from Tavily
        if response.get("answer"):
            results.append(f"💡 Quick Summary: {response['answer']}\n")

        # 2. Detailed search result snippets with URLs
        raw_results = response.get("results", [])
        for item in raw_results:
            title = item.get("title", "No Title")
            url = item.get("url", "")
            content = item.get("content", "").strip()
            results.append(f"🔗 [{title}]({url})\n{content}\n")

        if not results:
            return "No relevant search results found."

        return "\n".join(results)
    except Exception as e:
        return f"Web search failed: {str(e)}"


# Centralized Tools List for Agent Binding
ALL_TOOLS = [
    get_current_time,
    add_task,
    get_tasks,
    complete_task,
    save_fact,
    get_facts,
    delete_fact_by_key,
    search_web,
]


# -------------------------------------------------------------
# 5. GAIA Benchmark Tier 1 Tools
# -------------------------------------------------------------
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
            return text[:10000] + ("\n...[TRUNCATED]" if len(text) > 10000 else "")

        elif ext in ["csv", "txt", "md", "json"]:
            if ext == "csv":
                df = pd.read_csv(file_path)
                text = df.head(100).to_markdown()  # Show schema and top 100 rows
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            return text[:10000] + ("\n...[TRUNCATED]" if len(text) > 10000 else "")

        elif ext in ["xls", "xlsx"]:
            df = pd.read_excel(file_path)
            text = df.head(100).to_markdown()
            return text[:10000] + ("\n...[TRUNCATED]" if len(text) > 10000 else "")
        else:
            return f"Error: Unsupported file format '.{ext}'"
    except Exception as e:
        return f"Failed to read file: {str(e)}"


@tool
@retry_on_network_error(max_retries=3)
def fetch_web_page(url: str) -> str:
    """Fetch and extract clean readable text from a specific web page URL using Tavily Extract.

    Args:
        url: The full HTTP/HTTPS URL of the web page.
    """
    if not tavily_client:
        return "Tavily API key is missing. Please configure TAVILY_API_KEY in .env."

    try:
        response = tavily_client.extract(urls=[url])
        results = response.get("results", [])
        if not results:
            return "Failed to extract content from URL."

        text = results[0].get("raw_content", "")
        if not text:
            return "No content could be extracted from this URL."

        return text[:10000] + ("\n...[TRUNCATED]" if len(text) > 10000 else "")
    except Exception as e:
        return f"Failed to fetch URL: {str(e)}"


@tool
def execute_python(code: str) -> str:
    """Execute Python code in a local subprocess and return the standard output.
    Use this for math calculations, data analysis, or complex logic.
    Use `print()` to output the result you want to see.

    Args:
        code: The Python script to execute.
    """
    # Create a temporary file to hold the code
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        # Run the script with a 15-second timeout
        result = subprocess.run(
            ["python", temp_path], capture_output=True, text=True, timeout=15
        )
        output = result.stdout
        if result.stderr:
            output += f"\n--- STDERR ---\n{result.stderr}"

        if not output.strip():
            return "Execution completed successfully, but no output was printed. Did you forget to use print()?"

        return output[:5000] + ("\n...[TRUNCATED]" if len(output) > 5000 else "")
    except subprocess.TimeoutExpired:
        return "Error: Python execution timed out after 15 seconds."
    except Exception as e:
        return f"Error executing Python: {str(e)}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


import io
import ollama
from PIL import Image
from src.config import OLLAMA_VISION_MODEL, OLLAMA_BASE_URL


def _preprocess_image(image_path: str, max_dimension: int = 1120) -> bytes:
    """Resize image if dimensions exceed max_dimension and return JPEG/PNG bytes."""
    with Image.open(image_path) as img:
        img_format = img.format if img.format in ["JPEG", "PNG", "WEBP"] else "JPEG"

        # Convert RGBA to RGB if saving as JPEG
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
        # Preprocess and resize image to reduce token burn & upload payload
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


ALL_TOOLS.extend([read_document, fetch_web_page, execute_python, inspect_image])
