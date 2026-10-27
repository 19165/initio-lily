import os
import re
from datetime import datetime

from langchain_core.tools import tool

from src.config import NOTES_DIR, TOOL_DOC_MAX_CHARS


def _get_safe_note_path(title: str) -> str:
    """Sanitize title and resolve safe absolute file path inside NOTES_DIR."""
    target_dir = os.path.abspath(NOTES_DIR)
    os.makedirs(target_dir, exist_ok=True)

    # Sanitize title to prevent path traversal and illegal file characters
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


@tool
def create_or_update_note(title: str, content: str) -> str:
    """Create a new local Markdown note in the 'notes/' directory or overwrite an existing one.

    Args:
        title: Title or filename of the note (e.g. 'Project Ideas', 'Weekly Review'). The note will be saved in the 'notes/' directory.
        content: Full Markdown formatted text content to save.
    """
    try:
        file_path = _get_safe_note_path(title)
        filename = os.path.basename(file_path)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip())

        try:
            rel_path = os.path.relpath(file_path, os.getcwd()).replace("\\", "/")
        except Exception:
            rel_path = f"notes/{filename}"

        return f"Successfully saved note '{filename}' to '{rel_path}' ({len(content)} characters)."
    except Exception as e:
        return f"Failed to save note: {e}"


@tool
def append_to_note(title: str, content: str) -> str:
    """Append new text/entry to an existing Markdown note in the 'notes/' directory (or create one if it doesn't exist).

    Args:
        title: Title or filename of the note in the 'notes/' directory.
        content: Text content to append to the note.
    """
    try:
        file_path = _get_safe_note_path(title)
        filename = os.path.basename(file_path)

        existing_content = ""
        if os.path.exists(file_path):
            with open(file_path, encoding="utf-8") as f:
                existing_content = f.read().rstrip()

        separator = "\n\n" if existing_content else ""
        new_content = existing_content + separator + content.strip()

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        try:
            rel_path = os.path.relpath(file_path, os.getcwd()).replace("\\", "/")
        except Exception:
            rel_path = f"notes/{filename}"

        return f"Successfully appended to note '{filename}' at '{rel_path}'. Total length: {len(new_content)} characters."
    except Exception as e:
        return f"Failed to append to note: {e}"


@tool
def read_note(title: str) -> str:
    """Read the full content of a local Markdown note from the 'notes/' directory by title.

    Args:
        title: Title or filename of the note to read.
    """
    try:
        file_path = _get_safe_note_path(title)
        filename = os.path.basename(file_path)

        if not os.path.exists(file_path):
            return f"Error: Note '{filename}' not found."

        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        return text[:TOOL_DOC_MAX_CHARS] + (
            "\n...[TRUNCATED]" if len(text) > TOOL_DOC_MAX_CHARS else ""
        )
    except Exception as e:
        return f"Failed to read note: {e}"


@tool
def list_notes() -> str:
    """List all saved local Markdown notes in the knowledge base."""
    try:
        target_dir = os.path.abspath(NOTES_DIR)
        if not os.path.exists(target_dir):
            return "No local notes found in 'notes/' directory."

        entries = []
        for fname in sorted(os.listdir(target_dir)):
            if fname.lower().endswith((".md", ".txt")):
                fpath = os.path.join(target_dir, fname)
                if os.path.isfile(fpath):
                    size_kb = os.path.getsize(fpath) / 1024
                    mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                    entries.append(f"- 📄 **{fname}** ({size_kb:.1f} KB, Modified: {mtime})")

        if not entries:
            return "No local notes found in 'notes/' directory."

        return "### 📝 Local Notes Knowledge Base:\n" + "\n".join(entries)
    except Exception as e:
        return f"Failed to list notes: {e}"


@tool
def search_notes(query: str) -> str:
    """Search for keywords across all local Markdown notes.

    Args:
        query: Search keywords or phrase to look for inside note files.
    """
    try:
        target_dir = os.path.abspath(NOTES_DIR)
        if not os.path.exists(target_dir):
            return "No local notes found to search."

        clean_query = query.strip().lower()
        if not clean_query:
            return "Please provide a non-empty search query."

        matches = []
        for fname in sorted(os.listdir(target_dir)):
            if fname.lower().endswith((".md", ".txt")):
                fpath = os.path.join(target_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        with open(fpath, encoding="utf-8") as f:
                            lines = f.readlines()
                    except Exception:
                        continue

                    matching_snippets = []
                    for idx, line in enumerate(lines, 1):
                        if clean_query in line.lower():
                            snippet = line.strip()
                            if len(snippet) > 120:
                                snippet = snippet[:120] + "..."
                            matching_snippets.append(f"  └─ Line {idx}: {snippet}")
                            if len(matching_snippets) >= 3:
                                break

                    if matching_snippets:
                        matches.append(f"- 📄 **{fname}**:\n" + "\n".join(matching_snippets))

        if not matches:
            return f"No notes found matching '{query}'."

        return f"### 🔍 Search Results for '{query}':\n" + "\n\n".join(matches)
    except Exception as e:
        return f"Failed to search notes: {e}"
