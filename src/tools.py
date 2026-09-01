import time
from datetime import datetime
from langchain_core.tools import tool
from todoist_api_python.api import TodoistAPI
from src.config import TODOIST_API_KEY

todoist = TodoistAPI(token=TODOIST_API_KEY) if TODOIST_API_KEY else None


@tool
def get_current_time() -> str:
    """Get the current date and time formatted as a string."""
    return datetime.now().strftime("%d-%m-%Y, %H:%M:%S")


@tool
def add_task(
    content: str,
    due_string: str = "",
    description: str = "",
    max_retries: int = 5,
) -> str:
    """Create a new task in Todoist.

    Args:
        content: the title/name of the task (e.g. 'Buy Milk').
        due_string: Natural language due date (e.g. 'tomorrow at 5pm', 'next Monday', 'today').
        description: Optional detailed description for the task.
        max_retries: Maximum number of retry attempts for network failures.
    """
    if not todoist:
        return "Todoist API key is missing. Please configure TODOIST_API_KEY in .env."

    kwargs = {"content": content}
    if due_string:
        kwargs["due_string"] = due_string
    if description:
        kwargs["description"] = description

    for attempt in range(1, max_retries + 1):
        try:
            task = todoist.add_task(**kwargs)
            due_info = f", Due: {task.due.string}" if task.due else ""
            return f"Successfully created task '{task.content}' (ID: {task.id}{due_info})"
        except Exception as e:
            if attempt == max_retries:
                return f"Failed to create task after {max_retries} attempts: {str(e)}"
            print(f"Connection error on attempt {attempt}, retrying in 3s...")
            time.sleep(3)


@tool
def get_tasks(filter_query: str = "today | overdue") -> str:
    """Get active tasks from Todoist based on a filter.

    Args:
        filter_query: Todoist filter string. Common examples:
                        - 'today' (tasks due today)
                        - 'today | overdue' (tasks due today or past due)
                        - 'view all' or '' (all active tasks)
                        - '7 days' (tasks due in next 7 days)
    """
    if not todoist:
        return "Todoist API key is missing. Please configure TODOIST_API_KEY in .env."

    try:
        active_filter = (
            filter_query.strip()
            if filter_query and filter_query.strip() != "view all"
            else None
        )
        tasks = (
            todoist.get_tasks(filter=active_filter)
            if active_filter
            else todoist.get_tasks()
        )
        task_list = list(tasks)
        if not task_list:
            return "No tasks found."

        result_lines = []
        for t in task_list:
            due = f" (Due: {t.due.string})" if t.due else ""
            result_lines.append(f"- [ID: {t.id}] {t.content}{due}")
        return "\n".join(result_lines)
    except Exception as e:
        return f"Failed to fetch tasks: {str(e)}"


# Centralized Tools List for Agent Binding
ALL_TOOLS = [get_current_time, add_task, get_tasks]
