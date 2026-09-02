"""Tools registry and capabilities catalog for Lily AI Assistant."""

from src.tools.benchmark import (
    _preprocess_image,
    execute_python,
    inspect_image,
    read_document,
)
from src.tools.memory import delete_fact_by_key, get_facts, save_fact
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

# Benchmark & Execution tools
BENCHMARK_TOOLS = [
    read_document,
    fetch_web_page,
    execute_python,
    inspect_image,
]

# Centralized tools list bound to LLM
ALL_TOOLS = BASE_TOOLS + BENCHMARK_TOOLS

__all__ = [
    "ALL_TOOLS",
    "BASE_TOOLS",
    "BENCHMARK_TOOLS",
    "_preprocess_image",
    "add_task",
    "complete_task",
    "delete_fact_by_key",
    "execute_python",
    "fetch_web_page",
    "get_current_time",
    "get_facts",
    "get_tasks",
    "inspect_image",
    "read_document",
    "save_fact",
    "search_web",
]
