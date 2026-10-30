"""Centralized Tools Registry for Lily AI Assistant.

Exposes only the 3 Super Tools for the autonomous agent:
- `web_search`: Real-time web search and direct URL content extraction.
- `python_interpreter`: Sandboxed Python execution with workspace file access.
- `workspace_notes`: Consolidated Markdown knowledge base and notes manager.
"""

from src.tools.interpreter import execute_python, python_interpreter
from src.tools.notes import (
    append_to_note,
    create_or_update_note,
    list_notes,
    read_note,
    search_notes,
    workspace_notes,
)
from src.tools.schemas import (
    PythonInterpreterInput,
    WebSearchInput,
    WorkspaceNotesInput,
)
from src.tools.search import fetch_web_page, search_web, web_search

# The 3 Super Tools bound to the LLM agent
ALL_TOOLS = [
    web_search,
    python_interpreter,
    workspace_notes,
]

__all__ = [
    "ALL_TOOLS",
    "PythonInterpreterInput",
    "WebSearchInput",
    "WorkspaceNotesInput",
    "append_to_note",
    "create_or_update_note",
    "execute_python",
    "fetch_web_page",
    "list_notes",
    "python_interpreter",
    "read_note",
    "search_notes",
    "search_web",
    "web_search",
    "workspace_notes",
]
