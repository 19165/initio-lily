import re
from collections.abc import Sequence

from langchain_core.tools import BaseTool

from src.tools.benchmark import execute_python, inspect_image, read_document
from src.tools.memory import get_facts, save_fact
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

# Core tools always available in baseline context
CORE_BASE_TOOLS: list[BaseTool] = [
    get_current_time,
    save_fact,
    get_facts,
]

# Domain-specific tool groups
SEARCH_TOOLS: list[BaseTool] = [
    search_web,
    fetch_web_page,
]

NOTES_TOOLS_GROUP: list[BaseTool] = [
    create_or_update_note,
    append_to_note,
    read_note,
    list_notes,
    search_notes,
]

TASK_TOOLS: list[BaseTool] = [
    add_task,
    get_tasks,
    complete_task,
]

BENCHMARK_TOOLS_GROUP: list[BaseTool] = [
    read_document,
    execute_python,
    inspect_image,
]

# Intent Detection Regex Patterns
SEARCH_PATTERN = re.compile(
    r"(search|ค้น|หา|เว็บ|ข่าว|web|google|tavily|info|ราคา|vs|คืออะไร|เปรียบเทียบ|แนะนำ|วิจัย|biomechanics|exercise|technique|form|check)",
    re.IGNORECASE,
)

NOTES_PATTERN = re.compile(
    r"(note|โน้ต|จด|บันทึก|สรุป|เขียนลง|อ่านโน้ต|สร้างโน้ต|list\s*notes|knowledge)",
    re.IGNORECASE,
)

TASK_PATTERN = re.compile(
    r"(task|งาน|todo|สิ่งที่ต้องทำ|เตือน|กำหนด|due|done|overdue|เสร็จ)",
    re.IGNORECASE,
)

FILE_CODE_PATTERN = re.compile(
    r"(\.(pdf|csv|xlsx|xls|png|jpg|jpeg|webp|py|txt|json|md)\b|code|python|คำนวณ|รัน|ภาพ|รูป|เอกสาร|diagram|chart|calculate|script)",
    re.IGNORECASE,
)


def filter_tools_by_intent(
    user_input: str, all_tools: Sequence[BaseTool] | None = None
) -> list[BaseTool]:
    """Dynamically select relevant tools (4-7 max) based on user query intent.

    Prevents LLM attention overload when bound to 17+ tool schemas on local Ollama models.
    """
    query = user_input.strip()
    active_tools: list[BaseTool] = list(CORE_BASE_TOOLS)
    intent_detected = False

    # 1. Search Intent Detection
    if SEARCH_PATTERN.search(query):
        active_tools.extend(SEARCH_TOOLS)
        intent_detected = True

    # 2. Notes Knowledge Base Intent Detection
    if NOTES_PATTERN.search(query):
        active_tools.extend(NOTES_TOOLS_GROUP)
        intent_detected = True

    # 3. Tasks / Todoist Intent Detection
    if TASK_PATTERN.search(query):
        active_tools.extend(TASK_TOOLS)
        intent_detected = True

    # 4. Files, Document Analysis, Python Execution, Multimodal Vision
    if FILE_CODE_PATTERN.search(query):
        active_tools.extend(BENCHMARK_TOOLS_GROUP)
        intent_detected = True

    # 5. Fallback for General / Casual Chat (provide lightweight common capabilities)
    if not intent_detected:
        active_tools.extend([search_web, read_note, add_task])

    # Deduplicate while preserving insertion order
    seen_names = set()
    filtered: list[BaseTool] = []
    for tool in active_tools:
        if tool.name not in seen_names:
            seen_names.add(tool.name)
            filtered.append(tool)

    # If an external all_tools whitelist is provided, filter by it
    if all_tools:
        available_names = {t.name for t in all_tools}
        filtered = [t for t in filtered if t.name in available_names]

    return filtered
