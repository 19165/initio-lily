"""Comprehensive unit and contract test suite for Lily AI Agent Super Tools.

Coverage:
1. LLM Tool Calling Contract: Pydantic V2 schema validation and JSON schema definitions.
2. External Resource Contract (Tavily API): Full payloads, schema drift, network retries, and error handling.
3. Subprocess & OS Environment Contract: UTF-8 execution, stderr separation, timeouts, and workspace file access.
4. Local Filesystem Contract (Workspace Notes): CRUD operations, directory jail security, and unicode persistence.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.tools import ALL_TOOLS, python_interpreter, web_search, workspace_notes
from src.tools.schemas import (
    PythonInterpreterInput,
    WebSearchInput,
    WorkspaceNotesInput,
)

# ---------------------------------------------------------------------------
# 1. LLM Tool Calling Contract Tests (Pydantic V2 JSON Schemas)
# ---------------------------------------------------------------------------


class TestLLMToolCallingContracts:
    """Verifies that all Super Tools adhere to the strict LLM tool calling schema contract."""

    def test_all_tools_count_and_types(self):
        """ALL_TOOLS must contain exactly 3 Super Tools."""
        assert len(ALL_TOOLS) == 3
        tool_names = {t.name for t in ALL_TOOLS}
        assert tool_names == {"web_search", "python_interpreter", "workspace_notes"}

    def test_web_search_schema_contract(self):
        schema = WebSearchInput.model_json_schema()
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "max_results" in schema["properties"]
        assert "topic" in schema["properties"]
        assert schema["required"] == ["query"]

        # Valid input
        valid = WebSearchInput(query="LangGraph tutorial", max_results=5, topic="general")
        assert valid.query == "LangGraph tutorial"
        assert valid.max_results == 5

        # Invalid: max_results exceeds limit of 10
        with pytest.raises(ValidationError):
            WebSearchInput(query="test", max_results=99)

    def test_python_interpreter_schema_contract(self):
        schema = PythonInterpreterInput.model_json_schema()
        assert schema["type"] == "object"
        assert "code" in schema["properties"]
        assert schema["required"] == ["code"]

        # Valid input
        valid = PythonInterpreterInput(code="print(42)")
        assert valid.code == "print(42)"

        # Invalid: missing code
        with pytest.raises(ValidationError):
            PythonInterpreterInput()

    def test_workspace_notes_schema_contract(self):
        schema = WorkspaceNotesInput.model_json_schema()
        assert schema["type"] == "object"
        assert "action" in schema["properties"]
        assert set(schema["properties"]["action"]["enum"]) == {
            "read",
            "write",
            "append",
            "list",
            "search",
        }
        assert schema["required"] == ["action"]

        # Invalid: unknown action
        with pytest.raises(ValidationError):
            WorkspaceNotesInput(action="delete")


# ---------------------------------------------------------------------------
# 2. External Resource Contract: Tavily Search & Extract API
# ---------------------------------------------------------------------------


class TestTavilyExternalResourceContract:
    """Verifies contracts with Tavily external web service, handling payloads and schema drift."""

    def test_missing_api_key_contract(self):
        """When Tavily client is None (no API key), return a graceful user-friendly message."""
        with patch("src.tools.search.tavily_client", None):
            res = web_search.invoke({"query": "What is AI?"})
            assert "Web search is unavailable" in res
            assert "TAVILY_API_KEY" in res

    def test_tavily_search_full_contract_payload(self):
        """Verify handling of a standard well-formed Tavily Search API response."""
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {
            "answer": "LangGraph is a library for building stateful agent workflows.",
            "results": [
                {
                    "title": "LangGraph Overview",
                    "url": "https://langchain-ai.github.io/langgraph/",
                    "content": "Build cyclical agent graphs with checkpointers.",
                },
                {
                    "title": "GitHub Repo",
                    "url": "https://github.com/langchain-ai/langgraph",
                    "content": "Source code and release notes.",
                },
            ],
        }

        with patch("src.tools.search.tavily_client", mock_tavily):
            res = web_search.invoke({"query": "What is LangGraph?", "max_results": 2})

            # Check outbound contract call
            mock_tavily.search.assert_called_once_with(
                query="What is LangGraph?",
                search_depth="basic",
                topic="general",
                max_results=2,
                include_answer=True,
                include_raw_content=False,
            )

            # Check inbound contract parsing
            assert "💡 Quick Summary: LangGraph is a library" in res
            assert "🔗 [LangGraph Overview](https://langchain-ai.github.io/langgraph/)" in res
            assert "Build cyclical agent graphs" in res

    def test_tavily_search_schema_drift_missing_fields(self):
        """Verify resilience against third-party API schema drift (missing answer, empty list)."""
        mock_tavily = MagicMock()
        # API returns without 'answer' and items without 'content'
        mock_tavily.search.return_value = {
            "results": [{"title": "Only Title", "url": "https://example.com"}]
        }

        with patch("src.tools.search.tavily_client", mock_tavily):
            res = web_search.invoke({"query": "edge case query"})
            assert "🔗 [Only Title](https://example.com)" in res

    def test_tavily_extract_url_contract(self):
        """Verify contract for URL-triggered full webpage content extraction."""
        mock_tavily = MagicMock()
        mock_tavily.extract.return_value = {
            "results": [
                {
                    "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                    "raw_content": "Artificial intelligence is intelligence demonstrated by machines.",
                }
            ]
        }

        with patch("src.tools.search.tavily_client", mock_tavily):
            res = web_search.invoke({"query": "https://en.wikipedia.org/wiki/Artificial_intelligence"})

            # Must invoke extract() instead of search()
            mock_tavily.extract.assert_called_once_with(
                urls=["https://en.wikipedia.org/wiki/Artificial_intelligence"]
            )
            assert "📄 Content from https://en.wikipedia.org/wiki/Artificial_intelligence:" in res
            assert "intelligence demonstrated by machines" in res

    def test_tavily_network_error_contract(self):
        """Verify that external API network exceptions are caught and reported cleanly."""
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = TimeoutError("Connection to Tavily timed out")

        with patch("src.tools.search.tavily_client", mock_tavily):
            res = web_search.invoke({"query": "network failure test"})
            assert "failed" in res.lower() or "error" in res.lower()


# ---------------------------------------------------------------------------
# 3. Subprocess & OS Environment Contract: Python Interpreter
# ---------------------------------------------------------------------------


class TestPythonSubprocessOSContract:
    """Verifies contracts with OS subprocess execution, pipes, UTF-8, and workspace access."""

    def test_valid_code_stdout_contract(self):
        """Contract: stdout is captured, trimmed, and returned."""
        res = python_interpreter.invoke({"code": "print(21 * 2)"})
        assert res.strip() == "42"

    def test_utf8_thai_and_emoji_contract(self):
        """Contract: UTF-8 Thai language and Emoji are properly transported through pipes."""
        res = python_interpreter.invoke({"code": "print('สวัสดีค่ะ Master ลิลลี่มาแล้ว 🌸✨')"})
        assert "สวัสดีค่ะ Master ลิลลี่มาแล้ว 🌸✨" in res

    def test_stderr_separation_contract(self):
        """Contract: Runtime exceptions produce an informative STDERR section."""
        res = python_interpreter.invoke({"code": "raise ValueError('Custom test error')"})
        assert "--- STDERR ---" in res
        assert "ValueError: Custom test error" in res

    def test_empty_stdout_advisory_contract(self):
        """Contract: Code that runs without calling print() receives a helpful notice."""
        res = python_interpreter.invoke({"code": "x = 100\ny = 200"})
        assert "Did you forget to use print(...) to display the result?" in res

    def test_timeout_enforcement_contract(self):
        """Contract: Subprocess infinite loops are terminated at TOOL_EXEC_TIMEOUT_SEC."""
        # Patch timeout to 1.0 second for lightning fast deterministic test execution
        with patch("src.tools.interpreter.TOOL_EXEC_TIMEOUT_SEC", 1.0):
            res = python_interpreter.invoke({"code": "import time\nwhile True: time.sleep(0.1)"})
            assert "timed out after 1.0 seconds" in res

    def test_workspace_file_access_contract(self):
        """Contract: Python subprocess executes with cwd set to project workspace root."""
        code = (
            "import os\n"
            "assert os.path.exists('pyproject.toml'), 'pyproject.toml not found in cwd'\n"
            "print('Workspace access verified')"
        )
        res = python_interpreter.invoke({"code": code})
        assert "Workspace access verified" in res


# ---------------------------------------------------------------------------
# 4. Local Filesystem Contract: Workspace Notes Knowledge Base
# ---------------------------------------------------------------------------


class TestWorkspaceNotesFilesystemContract:
    """Verifies local disk persistence contract, CRUD actions, and directory jail security."""

    @pytest.fixture(autouse=True)
    def setup_temp_notes_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir
            with patch("src.tools.notes.NOTES_DIR", temp_dir):
                yield

    def test_write_and_read_contract(self):
        """Contract: Notes are written to disk in UTF-8 and read back intact."""
        write_res = workspace_notes.invoke({
            "action": "write",
            "title": "meeting_notes",
            "content": "Agenda: 1. Refactoring 2. Testing 🌸",
        })
        assert "Successfully saved note 'meeting_notes.md'" in write_res

        read_res = workspace_notes.invoke({"action": "read", "title": "meeting_notes"})
        assert "Agenda: 1. Refactoring 2. Testing 🌸" in read_res

    def test_append_contract(self):
        """Contract: Appending to a note adds content separated by double newlines."""
        workspace_notes.invoke({"action": "write", "title": "log", "content": "Entry 1"})
        workspace_notes.invoke({"action": "append", "title": "log", "content": "Entry 2"})

        read_res = workspace_notes.invoke({"action": "read", "title": "log"})
        assert "Entry 1\n\nEntry 2" in read_res

    def test_list_and_search_contract(self):
        """Contract: List returns all note files, and search locates matching keywords."""
        workspace_notes.invoke({"action": "write", "title": "alpha", "content": "LangGraph is great"})
        workspace_notes.invoke({"action": "write", "title": "beta", "content": "PostgreSQL database"})

        # List
        list_res = workspace_notes.invoke({"action": "list"})
        assert "alpha.md" in list_res
        assert "beta.md" in list_res

        # Search hit
        search_res = workspace_notes.invoke({"action": "search", "query": "LangGraph"})
        assert "alpha.md" in search_res
        assert "LangGraph is great" in search_res

        # Search miss
        miss_res = workspace_notes.invoke({"action": "search", "query": "nonexistent_token_xyz"})
        assert "No notes found matching 'nonexistent_token_xyz'" in miss_res

    def test_path_traversal_jail_security_contract(self):
        """Security Contract: Path traversal attempts (e.g. '../../etc/passwd') are strictly sanitized."""
        # _get_safe_note_path takes basename of title, neutralizing directory traversal
        res = workspace_notes.invoke({
            "action": "write",
            "title": "../../escaped_note",
            "content": "malicious content",
        })
        assert "Successfully saved note 'escaped_note.md'" in res

        # Verify the file was created INSIDE the jailed notes directory, not outside
        assert os.path.exists(os.path.join(self.temp_dir, "escaped_note.md"))
        assert not os.path.exists(os.path.join(self.temp_dir, "..", "..", "escaped_note.md"))

    def test_missing_required_arguments_contract(self):
        """Contract: Missing action-specific arguments return clear error descriptions."""
        res_read_no_title = workspace_notes.invoke({"action": "read"})
        assert "Error: 'title' is required" in res_read_no_title

        res_write_no_content = workspace_notes.invoke({"action": "write", "title": "test"})
        assert "Error: 'content' is required" in res_write_no_content

        res_search_no_query = workspace_notes.invoke({"action": "search"})
        assert "Error: 'query' is required" in res_search_no_query
