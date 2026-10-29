"""Isolated Python code interpreter tool for Lily AI Assistant."""

import contextlib
import os
import subprocess
import sys
import tempfile

from langchain_core.tools import tool

from src.config import TOOL_EXEC_MAX_CHARS, TOOL_EXEC_TIMEOUT_SEC
from src.tools.schemas import PythonInterpreterInput


@tool(args_schema=PythonInterpreterInput)
def python_interpreter(code: str) -> str:
    """Execute Python code in an isolated subprocess and return stdout/stderr.

    Features:
    - Runs in the workspace root directory with direct access to project files (CSV, PDF, Excel, TXT).
    - Supports pre-installed libraries: pandas, openpyxl, pypdf, PIL, json, math, os.
    - Enforces a 15-second execution timeout to prevent infinite loops.
    - Always use print(...) in your code to output results.
    """
    clean_code = code.strip()
    if not clean_code:
        return "Error: No Python code provided to execute."

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(clean_code)
            temp_path = f.name

        workspace_root = os.getcwd()
        sub_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TOOL_EXEC_TIMEOUT_SEC,
            cwd=workspace_root,
            env=sub_env,
            check=False,
        )

        output_parts: list[str] = []
        if result.stdout:
            output_parts.append(result.stdout.strip())
        if result.stderr:
            output_parts.append(f"--- STDERR ---\n{result.stderr.strip()}")

        full_output = "\n\n".join(output_parts).strip()

        if not full_output:
            return (
                "Python code executed successfully (exit code 0), but produced no output. "
                "Did you forget to use print(...) to display the result?"
            )

        if len(full_output) > TOOL_EXEC_MAX_CHARS:
            full_output = (
                full_output[:TOOL_EXEC_MAX_CHARS]
                + "\n...[TRUNCATED TO PREVENT CONTEXT OVERFLOW]"
            )

        return full_output

    except subprocess.TimeoutExpired:
        return (
            f"Execution timed out after {TOOL_EXEC_TIMEOUT_SEC} seconds. "
            "Please optimize your code or check for infinite loops."
        )
    except Exception as e:
        return f"Python execution error: {e!s}"
    finally:
        if temp_path and os.path.exists(temp_path):
            with contextlib.suppress(OSError):
                os.remove(temp_path)


# Backwards-compatibility alias for legacy callers during incremental refactor
execute_python = python_interpreter
