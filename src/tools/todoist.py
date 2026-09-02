from langchain_core.tools import tool
from todoist_api_python.api import TodoistAPI

from src.config import TODOIST_API_KEY
from src.utils import retry_on_network_error

todoist = TodoistAPI(token=TODOIST_API_KEY) if TODOIST_API_KEY else None


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
