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
        default=4,
        ge=1,
        le=10,
        description="Number of search results to return (default: 4, max: 10).",
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
    """Input schema for local markdown notes and knowledge base management."""

    action: Literal["read", "write", "append", "list", "search"] = Field(
        ...,
        description=(
            "Action to perform: "
            "'read' to read a note, "
            "'write' to create or overwrite a note, "
            "'append' to append text to an existing note, "
            "'list' to list all note filenames in the workspace, "
            "'search' to search for text inside all notes."
        ),
    )
    title: str | None = Field(
        default=None,
        description="Title or filename of the note (required for 'read', 'write', 'append').",
    )
    content: str | None = Field(
        default=None,
        description="Text content to save or append (required for 'write' and 'append').",
    )
    query: str | None = Field(
        default=None,
        description="Keyword to search across notes (required for 'search' action).",
    )
