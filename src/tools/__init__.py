"""Tools registry and capabilities catalog for Lily AI Assistant."""

from src.tools.system import get_current_time
from src.tools.todoist import add_task, get_tasks, complete_task
from src.tools.memory import save_fact, get_facts, delete_fact_by_key
from src.tools.search import search_web, fetch_web_page
from src.tools.benchmark import (
    read_document,
    execute_python,
    inspect_image,
    _preprocess_image,
)

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
    "get_current_time",
    "add_task",
    "get_tasks",
    "complete_task",
    "save_fact",
    "get_facts",
    "delete_fact_by_key",
    "search_web",
    "fetch_web_page",
    "read_document",
    "execute_python",
    "inspect_image",
    "_preprocess_image",
]
