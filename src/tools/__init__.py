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


# Compatibility pass-through for agent before LangGraph migration
def filter_tools_by_intent(user_input: str, tools: list | None = None) -> list:
    """Pass-through stub: all 3 Super Tools are always directly available to the LLM."""
    return ALL_TOOLS if tools is None else tools


# Compatibility stubs for legacy CLI commands before Task 5.0 cleanup
def get_facts(category: str | None = None) -> str:
    """Compatibility stub reading facts from db."""
    from src.db import get_all_facts

    facts = get_all_facts(category)
    if not facts:
        return "No facts recorded yet."
    return "\n".join([f"- [{f['category']}] {f['key']}: {f['value']}" for f in facts])


def get_tasks(filter_by: str = "all") -> str:
    """Compatibility stub for deprecated Todoist integration."""
    return "Notice: Todoist task integration has been retired. Please use `workspace_notes` to manage your tasks in Markdown format."


def complete_task(task_identifier: str) -> str:
    """Compatibility stub for deprecated Todoist integration."""
    return f"Notice: Todoist is retired. Task '{task_identifier}' was not modified. Please use `workspace_notes`."


def add_task(content: str, due_string: str = "today", priority: int = 1) -> str:
    """Compatibility stub for deprecated Todoist integration."""
    return "Notice: Todoist is retired. Please use `workspace_notes(action='append', title='tasks.md', content=...)` instead."


__all__ = [
    "ALL_TOOLS",
    "PythonInterpreterInput",
    "WebSearchInput",
    "WorkspaceNotesInput",
    "add_task",
    "append_to_note",
    "complete_task",
    "create_or_update_note",
    "execute_python",
    "fetch_web_page",
    "filter_tools_by_intent",
    "get_facts",
    "get_tasks",
    "list_notes",
    "python_interpreter",
    "read_note",
    "search_notes",
    "search_web",
    "web_search",
    "workspace_notes",
]
