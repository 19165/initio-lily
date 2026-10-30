## Relevant Files

- `src/tools/search.py` - Core search tool where relevance chunking, noise filtering, and query truncation limits will be implemented.
- `src/gaia_evaluator.py` - Evaluator harness where universal stopping criteria, direct path resolution, and self-awareness prompt directives will be enhanced.
- `src/agent.py` - Agent loop orchestrator where runtime query deduplication and graceful turn budget tracking will be added.
- `tests/test_tools.py` - Unit and contract tests for search truncation and relevance filtering.
- `tests/test_gaia_evaluator.py` - Tests for stopping rules and direct path execution.
- `tests/test_agent.py` - Tests for runtime query deduplication and state protection.

### Notes

- Avoid hacky "force-guess" fallbacks. All optimizations must be grounded in sound engineering (noise reduction, deduplication, and early exit).
- Use `uv run pytest` to execute unit tests.
- Use `uv run ruff check .` for PEP 8 compliance.

## Instructions for Completing Tasks

**IMPORTANT:** As you complete each task, you must check it off in this markdown file by changing `- [ ]` to `- [x]`. This helps track progress and ensures you don't skip any steps.

Example:
- `- [ ] 1.1 Read file` → `- [x] 1.1 Read file` (after completing)

Update the file after completing each sub-task, not just after completing an entire parent task.

## Tasks

- [x] 0.0 Create feature branch
  - [x] 0.1 Create and checkout a new branch for this optimization (`git checkout -b feature/agent-performance-optimization`)
- [x] 1.0 Clean & Compact Search Tool (Tame Token Monster & Reduce Noise)
  - [x] 1.1 Introduce `TOOL_SEARCH_EXTRACT_MAX_CHARS = 3500` in `src/config.py` and enforce strict extraction truncation in `src/tools/search.py`
  - [x] 1.2 Tune `max_results` default to 3 in `web_search` and `WebSearchInput` schema (`src/tools/schemas.py`) to reduce lower-rank noise
  - [x] 1.3 Normalize and clean search snippets in `web_search` to eliminate boilerplate navigation, repetitive headers, and empty artifacts
  - [x] 1.4 Update unit and contract tests in `tests/test_tools.py` covering the 3,500-char truncation guard and 3-result Top-K default
- [x] 2.0 Implement Runtime Query Deduplication Guard in Agent Engine
  - [x] 2.1 Implement `calculate_jaccard_similarity(q1, q2)` helper in `src/utils.py` to detect near-duplicate queries (>0.75 similarity threshold)
  - [x] 2.2 Add per-turn query cache and interceptor hook in `src/agent.py` prior to executing external search tool calls
  - [x] 2.3 Construct synthetic feedback message (`[SYSTEM DEDUPLICATION NOTICE]`) commanding the model to halt repetition and pivot strategy
  - [x] 2.4 Update unit tests in `tests/test_agent.py` and `tests/test_utils.py` verifying exact & fuzzy deduplication interception without invoking live APIs
- [ ] 3.0 Universal Meta-Cognitive Prompting & Early-Exit Protocol in GAIA Evaluator
  - [ ] 3.1 Formulate and integrate Universal Information Satiation directive (halt immediately upon locating first candidate evidence) in `GAIA_SYSTEM_PROMPT` (`src/gaia_evaluator.py`)
  - [ ] 3.2 Add Zero-Overhead Direct Path directive in `src/gaia_evaluator.py` strictly forbidding exploratory `os.walk` / `os.listdir` loops when attached files are provided
  - [ ] 3.3 Implement Early-Exit directive instructing the agent to immediately conclude when encountering unresolvable environment limits (e.g. missing audio/video libraries)
  - [ ] 3.4 Update unit and integration tests in `tests/test_gaia_evaluator.py` verifying direct path prompt injection, stopping directives, and early termination behavior
- [ ] 4.0 Accurate Token Telemetry on Interrupted Steps
  - [ ] 4.1 Update `LilyAgent.chat_stream()` in `src/agent.py` to yield accumulated partial token usage metrics inside `except GraphRecursionError:` prior to error event
  - [ ] 4.2 Update `session_stats` accumulators in `LilyAgent` to capture interrupted turn consumption, ensuring session analytics reflect reality
  - [ ] 4.3 Update unit tests in `tests/test_agent.py` verifying that recursion-limited runs still emit non-zero `usage` events with accurate token counts
- [ ] 5.0 End-to-End Verification & Benchmark Regression Testing
  - [ ] 5.1 Run full regression test suite (`uv run pytest`) and linting check (`uv run ruff check .`) ensuring 100% green build
  - [ ] 5.2 Execute representative benchmark tasks and verify reduction of search steps and token usage in `logging.csv`
  - [ ] 5.3 Document optimizations, telemetry before/after comparison, and empirical findings in `walkthrough.md`
  - [ ] 5.4 Commit and push all changes following conventional commit standards to feature branch
