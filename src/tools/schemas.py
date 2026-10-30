"""Pydantic V2 input schemas for Lily AI Agent tools.

Defines strict input validation schemas for the 3 consolidated tools:
- WebSearchInput: Tavily web search and URL content extraction.
- PythonInterpreterInput: Subprocess execution of Python code.
- WorkspaceNotesInput: Markdown notes knowledge base operations.
"""

from typing import Literal

from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    """Input schema for web search and webpage content extraction."""

    query: str = Field(
        ...,
        description=(
            "Search keywords (e.g. 'Bitcoin price today', 'Bangkok weather', 'Python changelog') "
            "OR a full HTTP/HTTPS URL to directly extract full webpage text."
        ),
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of search results to return (default: 3, max: 10).",
    )
    topic: Literal["general", "news"] = Field(
        default="general",
        description="Search topic: 'general' for facts/general info, or 'news' for recent news.",
    )


class PythonInterpreterInput(BaseModel):
    """Input schema for isolated Python code execution."""

    code: str = Field(
        ...,
        description=(
            "The complete, executable Python code to run. "
            "Always use print(...) to output values you want returned. "
            "You have direct access to files in the workspace (CSV, PDF, Excel) "
            "and pre-installed libraries (pandas, openpyxl, pypdf, PIL, json, math, os)."
        ),
    )


class WorkspaceNotesInput(BaseModel):
    """Input schema for local notes management operations."""

    action: Literal["read", "write", "append", "list", "search"] = Field(
        ...,
        description=(
            "Action to perform on notes: "
            "'read' to view note content, "
            "'write' to create/overwrite a note, "
            "'append' to add content to an existing note, "
            "'list' to list all notes, "
            "'search' to find notes containing keywords."
        ),
    )
    title: str = Field(
        default="",
        description=(
            "Title or filename of the note without extension (e.g. 'meeting_notes', 'todo'). "
            "Required for read, write, append."
        ),
    )
    content: str = Field(
        default="",
        description="Content to write or append to the note. Required for write and append.",
    )
    query: str = Field(
        default="",
        description="Keyword to search across notes. Required for search action.",
    )
