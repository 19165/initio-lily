"""Local markdown knowledge base and workspace notes tool for Lily AI Assistant."""

import os
import re
from datetime import datetime
from typing import Literal

from langchain_core.tools import tool

from src.config import NOTES_DIR, TOOL_DOC_MAX_CHARS
from src.tools.schemas import WorkspaceNotesInput


def _get_safe_note_path(title: str) -> str:
    """Sanitize title and resolve safe absolute file path strictly inside NOTES_DIR."""
    target_dir = os.path.abspath(NOTES_DIR)
    os.makedirs(target_dir, exist_ok=True)

    base_title = os.path.basename(title.strip())
    clean_title = re.sub(r'[\\/*?:"<>|]', "", base_title).strip()
    if not clean_title:
        clean_title = "untitled"

    if not clean_title.lower().endswith((".md", ".txt")):
        clean_title += ".md"

    full_path = os.path.abspath(os.path.join(target_dir, clean_title))

    # Security check: Ensure file path does not escape NOTES_DIR
    if os.path.commonpath([target_dir, full_path]) != target_dir:
        raise ValueError(f"Path traversal detected for note title: '{title}'")

    return full_path


@tool(args_schema=WorkspaceNotesInput)
def workspace_notes(
    action: Literal["read", "write", "append", "list", "search"],
    title: str | None = None,
    content: str | None = None,
    query: str | None = None,
) -> str:
    """Manage local Markdown notes and knowledge base in the 'notes/' directory.

    Supported Actions:
    - 'read': Read note content by title. Requires: title
    - 'write': Create new or overwrite existing note. Requires: title, content
    - 'append': Add text to an existing note (creates note if missing). Requires: title, content
    - 'list': List all notes with filenames, modification times, and preview.
    - 'search': Search for text across all saved notes. Requires: query
    """
    target_dir = os.path.abspath(NOTES_DIR)
    os.makedirs(target_dir, exist_ok=True)

    try:
        # 1. READ ACTION
        if action == "read":
            if not title or not title.strip():
                return "Error: 'title' is required for the 'read' action."
            file_path = _get_safe_note_path(title)
            filename = os.path.basename(file_path)

            if not os.path.exists(file_path):
                return f"Error: Note '{filename}' not found."

            with open(file_path, encoding="utf-8") as f:
                note_text = f.read()

            truncated = note_text[:TOOL_DOC_MAX_CHARS]
            if len(note_text) > TOOL_DOC_MAX_CHARS:
                truncated += "\n...[TRUNCATED TO PREVENT CONTEXT OVERFLOW]"
            return f"📝 Note '{filename}':\n\n{truncated}"

        # 2. WRITE ACTION
        elif action == "write":
            if not title or not title.strip():
                return "Error: 'title' is required for the 'write' action."
            if content is None:
                return "Error: 'content' is required for the 'write' action."

            file_path = _get_safe_note_path(title)
            filename = os.path.basename(file_path)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content.strip())

            return f"Successfully saved note '{filename}' ({len(content)} characters)."

        # 3. APPEND ACTION
        elif action == "append":
            if not title or not title.strip():
                return "Error: 'title' is required for the 'append' action."
            if content is None:
                return "Error: 'content' is required for the 'append' action."

            file_path = _get_safe_note_path(title)
            filename = os.path.basename(file_path)

            existing = ""
            if os.path.exists(file_path):
                with open(file_path, encoding="utf-8") as f:
                    existing = f.read().rstrip()

            separator = "\n\n" if existing else ""
            new_content = existing + separator + content.strip()

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return f"Successfully appended to note '{filename}'. Total length: {len(new_content)} characters."

        # 4. LIST ACTION
        elif action == "list":
            files = [
                f
                for f in os.listdir(target_dir)
                if f.lower().endswith((".md", ".txt")) and os.path.isfile(os.path.join(target_dir, f))
            ]
            if not files:
                return "No saved notes found in 'notes/' directory."

            lines = ["### 📝 Local Notes Knowledge Base:"]
            for f in sorted(files):
                full = os.path.join(target_dir, f)
                stat = os.stat(full)
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                size = stat.st_size
                lines.append(f"- 📄 `{f}` ({size} bytes, modified: {mtime})")

            return "\n".join(lines)

        # 5. SEARCH ACTION
        elif action == "search":
            if not query or not query.strip():
                return "Error: 'query' is required for the 'search' action."

            search_term = query.strip().lower()
            files = [
                f
                for f in os.listdir(target_dir)
                if f.lower().endswith((".md", ".txt")) and os.path.isfile(os.path.join(target_dir, f))
            ]

            matches: list[str] = []
            for f in sorted(files):
                full = os.path.join(target_dir, f)
                with open(full, encoding="utf-8") as file_obj:
                    text = file_obj.read()
                if search_term in text.lower():
                    # Extract matching snippet lines
                    matched_lines = [
                        line.strip()
                        for line in text.splitlines()
                        if search_term in line.lower()
                    ][:3]
                    snippet = " | ".join(matched_lines)
                    matches.append(f"🔗 `{f}`: {snippet}")

            if not matches:
                return f"No notes found matching '{query}'."

            return f"### 🔍 Search Results for '{query}':\n" + "\n".join(matches)

        else:
            return f"Error: Unknown action '{action}'. Supported actions: read, write, append, list, search."

    except ValueError as ve:
        return f"Security Error: {ve!s}"
    except Exception as e:
        return f"Failed to execute workspace note action '{action}': {e!s}"


# Backwards-compatibility legacy wrappers for current consumers
@tool
def create_or_update_note(title: str, content: str) -> str:
    """Legacy alias for workspace_notes(action='write')."""
    return workspace_notes.invoke({"action": "write", "title": title, "content": content})


@tool
def append_to_note(title: str, content: str) -> str:
    """Legacy alias for workspace_notes(action='append')."""
    return workspace_notes.invoke({"action": "append", "title": title, "content": content})


@tool
def read_note(title: str) -> str:
    """Legacy alias for workspace_notes(action='read')."""
    return workspace_notes.invoke({"action": "read", "title": title})


@tool
def list_notes() -> str:
    """Legacy alias for workspace_notes(action='list')."""
    return workspace_notes.invoke({"action": "list"})


@tool
def search_notes(query: str) -> str:
    """Legacy alias for workspace_notes(action='search')."""
    return workspace_notes.invoke({"action": "search", "query": query})
