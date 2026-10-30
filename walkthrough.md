# Walkthrough: Agent Performance Optimization & Benchmark Stabilization

## 📌 Executive Summary

This optimization cycle targets the primary bottleneck observed during GAIA benchmark evaluations: **repetitive search loops, context window saturation from bloated search extracts, exploratory path delays, and zero-token telemetry loss upon recursion limit interruptions**.

All five engineering initiatives (0.0 to 5.0) have been implemented, verified through 86 unit and contract tests, and empirically confirmed via live test runs.

---

## 🛠️ Summary of Implemented Optimizations

### 1. Clean & Compact Search Tool (Task 1.0)
- **Problem**: Raw Tavily extract responses frequently bloated beyond 15,000–30,000 characters, consuming vast amounts of LLM context and inducing token exhaustion.
- **Solution**:
  - Enforced `TOOL_SEARCH_EXTRACT_MAX_CHARS = 3500` in `src/config.py` and `src/tools/search.py`.
  - Changed `max_results` default from 5 to 3 in `web_search` and `WebSearchInput` (`src/tools/schemas.py`).
  - Added text normalization in `format_tool_result()` to clean repetitive boilerplate navigation.

### 2. Runtime Query Deduplication Guard (Task 2.0)
- **Problem**: When encountering ambiguous questions, LLMs frequently enter perseveration loops by submitting near-identical search queries (e.g. adding quotes or reordering words).
- **Solution**:
  - Implemented `calculate_jaccard_similarity()` in `src/utils.py` with a 0.75 token overlap threshold.
  - Subclassed LangGraph's `ToolNode` with `DeduplicatingToolNode` in `src/agent.py`.
  - Near-duplicate queries are intercepted in 0.001s, returning a synthetic `[SYSTEM DEDUPLICATION NOTICE]` instructing the model to pivot to structured sources or conclude reasoning.

### 3. Meta-Cognitive Prompting & Early-Exit Protocol (Task 3.0)
- **Problem**: Agent continued exploratory loops even after locating candidate evidence, executed expensive `os.walk` scans instead of opening attached files directly, and hung indefinitely when missing system libraries (e.g., audio/video codecs).
- **Solution**:
  - **Universal Information Satiation**: Instructs the agent to halt tool calls immediately upon discovering candidate evidence.
  - **Zero-Overhead Direct Path**: Injects explicit directives forbidding `os.walk`/`os.listdir` loops when attached files are supplied.
  - **Early-Exit Protocol**: Directs the agent to gracefully conclude when encountering environment boundaries rather than attempting infinite retry loops.

### 4. Accurate Token Telemetry on Interrupted Steps (Task 4.0)
- **Problem**: When an agent run hit `GraphRecursionError`, the stream exited before emitting a `"usage"` event, causing session stats and benchmark logs to record 0 tokens for failed turns.
- **Solution**:
  - Updated `LilyAgent.chat_stream()` in `src/agent.py` to calculate and emit accumulated turn tokens (`turn_in_tokens`, `turn_out_tokens`, `turn_total_tokens`) inside the `except GraphRecursionError:` block before the error notification.
  - Updated `self.session_stats` accumulators so that session analytics accurately reflect real consumption.

---

## 📊 Telemetry & Empirical Verification

### Empirical Run: Mercedes Sosa Benchmark Task (`8e867cd7-cff9-4e6c-867a-ff5ddc2550be`)

| Metric / Behavior | v1.0 (Baseline) | Current (Optimized) | Impact |
| :--- | :--- | :--- | :--- |
| **Search Extract Size** | 10,000+ chars/result | $\le$ 3,500 chars | ~65% reduction in search token volume |
| **Top-K Search Results** | 5 results | 3 results | Eliminates low-rank noise |
| **Query Loop Interception** | None (executed duplicate live API calls) | **Intercepted at Step 4 (0.001s)** with `[SYSTEM DEDUPLICATION NOTICE]` | Saves external API call latency & cost |
| **Model Adaptation** | Kept repeating searches until step limit | **Pivoted to Wikipedia query** (`site:en.wikipedia.org ...`) at Step 5 | Proves meta-cognitive responsiveness |
| **Interrupted Run Telemetry** | 0 tokens recorded | **18,291 tokens accurately captured** | Fixes benchmark data integrity |

---

## 🧪 Test Suite Results

```text
============================= 86 passed in 3.13s ==============================
- tests/test_agent.py: 14 passed
- tests/test_db.py: 12 passed
- tests/test_gaia_evaluator.py: 14 passed
- tests/test_notes.py: 7 passed
- tests/test_tools.py: 20 passed
- tests/test_utils.py: 19 passed
```

Linter status:
```text
uv run ruff check .
All checks passed!
```
