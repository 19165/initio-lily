"""Tools registry and capabilities catalog for Lily AI Assistant."""

from src.tools.benchmark import (
    _preprocess_image,
    execute_python,
    inspect_image,
    read_document,
)
from src.tools.filter import filter_tools_by_intent
from src.tools.memory import delete_fact_by_key, get_facts, save_fact
from src.tools.notes import (
    append_to_note,
    create_or_update_note,
    list_notes,
    read_note,
    search_notes,
)
from src.tools.search import fetch_web_page, search_web
from src.tools.system import get_current_time
from src.tools.todoist import add_task, complete_task, get_tasks

# Standard ReAct agent tools
BASE_TOOLS = [
    get_current_time,
    add_task,
    get_tasks,
    complete_task,
    save_fact,
    get_facts,
    delete_fact_by_key,
    search_web,
]

# Local Knowledge Base & Notes tools
NOTES_TOOLS = [
    create_or_update_note,
    append_to_note,
    read_note,
    list_notes,
    search_notes,
]

# Benchmark & Execution tools
BENCHMARK_TOOLS = [
    read_document,
    fetch_web_page,
    execute_python,
    inspect_image,
]

# Centralized tools list bound to LLM
ALL_TOOLS = BASE_TOOLS + NOTES_TOOLS + BENCHMARK_TOOLS

__all__ = [
    "ALL_TOOLS",
    "BASE_TOOLS",
    "BENCHMARK_TOOLS",
    "NOTES_TOOLS",
    "_preprocess_image",
    "add_task",
    "append_to_note",
    "complete_task",
    "create_or_update_note",
    "delete_fact_by_key",
    "execute_python",
    "fetch_web_page",
    "filter_tools_by_intent",
    "get_current_time",
    "get_facts",
    "get_tasks",
    "inspect_image",
    "list_notes",
    "read_document",
    "read_note",
    "save_fact",
    "search_notes",
    "search_web",
]
