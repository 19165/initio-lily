## Relevant Files

- `src/tools/schemas.py` - Pydantic models and input validation schemas for the 3 consolidated tools.
- `src/tools/search.py` - Reliable Tavily web search with network retries and output formatting.
- `src/tools/benchmark.py` - Sandboxed Python code interpreter tool with timeouts.
- `src/tools/notes.py` - Unified local Markdown notes and knowledge base tool.
- `src/tools/__init__.py` - Central tool exports exposing only the 3 reliable tools (`ALL_TOOLS`).
- `tests/test_tools.py` - Unit tests for consolidated tools and schema validations.
- `src/db.py` - LangGraph checkpointer initialization with SQLite/PostgreSQL fail-safe fallbacks.
- `tests/test_db.py` - Unit tests for DB checkpointer, session storage, and fallback mechanisms.
- `src/agent.py` - LangGraph StateGraph execution loop, node definitions, and streaming telemetry.
- `tests/test_agent.py` - Unit tests for LangGraph transitions, context trimming, and event streaming.
- `src/gaia_evaluator.py` - GAIA benchmark and execution harness integrating with LangGraph traces.
- `tests/test_gaia_evaluator.py` - Unit tests for evaluation harness and trace generation.
- `main.py` - Rich CLI terminal interface updated for LangGraph event streaming and new tool commands.
- `agent.py` - Legacy root file to be removed.

### Notes

- Target test framework: `pytest` with `pytest-mock`.
- Run tests using `uv run pytest`.
- Lint and type checks using `uv run ruff check .`.
- As sub-tasks are completed, check them off by changing `- [ ]` to `- [x]`.

## Instructions for Completing Tasks

**IMPORTANT:** As you complete each task, you must check it off in this markdown file by changing `- [ ]` to `- [x]`. This helps track progress and ensures you don't skip any steps.

Example:
- `- [ ] 1.1 Read file` → `- [x] 1.1 Read file` (after completing)

Update the file after completing each sub-task, not just after completing an entire parent task.

## Tasks

- [x] 0.0 Create feature branch
  - [x] 0.1 Create and checkout a new branch for this feature (`git checkout -b feature/langgraph-reliable-architecture`)
- [x] 1.0 Consolidate Core Tools into 3 Reliable Super Tools (src/tools)
  - [x] 1.1 Create `src/tools/schemas.py` defining strict Pydantic V2 input schemas (`WebSearchInput`, `PythonInterpreterInput`, `WorkspaceNotesInput` with 5 actions: read, write, append, list, search)
  - [x] 1.2 Refactor `src/tools/search.py` to use `WebSearchInput` supporting dual-mode (Tavily Search for keywords & Tavily Extract for URLs) with `@retry_on_network_error`
  - [x] 1.3 Create `src/tools/interpreter.py` executing Python with `cwd=workspace_root` (allowing project file access), 15s timeout, and stdout/stderr capture
  - [x] 1.4 Refactor `src/tools/notes.py` into a unified `workspace_notes` tool supporting 5 actions (`read`, `write`, `append`, `list`, `search`) with strict `notes/` directory jail
  - [x] 1.5 Delete legacy tool files (`todoist.py`, `filter.py`, `system.py`, `memory.py`, `benchmark.py`) and export only `ALL_TOOLS = [web_search, python_interpreter, workspace_notes]` in `src/tools/__init__.py`
  - [x] 1.6 Update unit tests in `tests/test_tools.py` (retiring `test_tool_filter.py`) covering 3 Super Tools, external resource contract tests (Tavily API payloads, LLM JSON schemas, Subprocess UTF-8), and defensive edge cases
- [x] 2.0 Implement Dual-Layer Memory & Resilient Checkpointer (src/db)
  - [x] 2.1 Implement `get_checkpointer()` in `src/db.py` utilizing `PostgresSaver` with fail-fast diagnostics in production, and `MemorySaver` for unit testing
  - [x] 2.2 Verify persistent Long-term Memory (`user_facts` table CRUD) in `src/db.py` for cross-session Master profile facts
  - [x] 2.3 Provide thread/session helpers (`get_thread_config`, `reset_thread`) for LangGraph state management and thread isolation
  - [x] 2.4 Update unit tests in `tests/test_db.py` covering checkpointer lifecycle, thread configuration, and user facts CRUD operations
- [ ] 3.0 Build LangGraph Agent Engine & Event Streaming (src/agent)
  - [ ] 3.1 Define `AgentState` TypedDict, dynamic system prompt builder (injecting real-time timestamp & user facts), and safe context trimmer
  - [ ] 3.2 Build and compile LangGraph `StateGraph` (`agent` node -> `tools_condition` -> `ToolNode`) with checkpointer and recursion limits
  - [ ] 3.3 Implement `chat_stream(user_input)` generator adapting LangGraph stream events into standardized telemetry events (`token`, `tool_start`, `tool_end`, `usage`, `error`)
  - [ ] 3.4 Update unit tests in `tests/test_agent.py` for LangGraph transitions, context trimming, and event streaming
- [ ] 4.0 Modernize Harness & GAIA Benchmark Evaluator (src/gaia_evaluator)
  - [ ] 4.1 Update `GAIABenchmarkAgent` with unique thread isolation per evaluation run and align prompts to the 3 Super Tools
  - [ ] 4.2 Build step-by-step trajectory recorder tracking tool call arguments, outputs, execution duration, and token metrics
  - [ ] 4.3 Enhance exact-match final answer sanitizer to eliminate formatting artifacts and Markdown conversational noise
  - [ ] 4.4 Update unit tests in `tests/test_gaia_evaluator.py` covering mock evaluation runs, trajectory generation, and file resolver
- [ ] 5.0 Presentation Layer Integration & Legacy Cleanup (main.py & root files)
  - [ ] 5.1 Remove obsolete legacy files (`agent.py`, `src/tools/filter.py`, `tests/test_tool_filter.py`) and clean up empty directories
  - [ ] 5.2 Update `main.py` Rich CLI interface with updated slash commands and LangGraph stream event handlers
  - [ ] 5.3 Verify `app.py` Gradio interface compatibility with the updated LangGraph agent
  - [ ] 5.4 Run full test suite (`uv run pytest`) and linting (`uv run ruff check .`) to verify 100% green build
