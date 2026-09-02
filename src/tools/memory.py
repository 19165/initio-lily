from langchain_core.tools import tool
from src.db import upsert_fact, get_all_facts, delete_fact


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
