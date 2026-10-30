"""Modernized GAIA Benchmark Evaluator and Harness.

Decoupled, isolated, exact-match benchmark evaluator tailored for the
Hugging Face Agents Course Unit 4 scoring leaderboard.
"""

import os
import re
import sys
import time
import uuid
from typing import Any

# Reconfigure stdout/stderr encoding for Windows console emojis
stdout_reconf = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconf):
    stdout_reconf(encoding="utf-8")
stderr_reconf = getattr(sys.stderr, "reconfigure", None)
if callable(stderr_reconf):
    stderr_reconf(encoding="utf-8")

# Ensure the root project directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import LilyAgent  # noqa: E402
from src.config import AGENT_MAX_STEPS  # noqa: E402
from src.utils import clean_markdown_text  # noqa: E402

GAIA_SYSTEM_PROMPT = """You are an expert autonomous problem-solving agent taking the GAIA benchmark test.
Your goal is to solve the task accurately and output ONLY the exact final answer.

CRITICAL BENCHMARK RULES:
1. Output format: You MUST output ONLY the final answer without ANY conversational filler, pleasantries, explanations, or punctuation wrappers.
   - If the answer is 42, output: 42
   - If the answer is John Doe, output: John Doe
   - Never write 'The answer is 42' or 'Final Answer: 42'.
2. Tool usage guidelines:
   - Use `python_interpreter` for:
     * Inspecting, reading, and parsing local files (Excel .xlsx, CSV, PDF, JSON, images, text). You can use libraries like pandas, openpyxl, pypdf, PIL, sqlite3 in your code.
     * Complex mathematical calculations, data transformations, and algorithm execution.
   - Use `web_search` for:
     * Looking up real-world facts, current information, and extracting web page content via URLs.
   - Use `workspace_notes` if you need a persistent scratchpad for multi-step reasoning.
3. Think step-by-step and verify your calculations before giving the final answer."""


def sanitize_gaia_answer(text: str) -> str:
    """Sanitize LLM output for strict exact-match benchmark grading.

    Eliminates LaTeX formatting, conversational prefixes ('Final Answer:'),
    markdown bold/code markers, enclosing quotes, and extraneous trailing periods.
    """
    if not text:
        return ""

    cleaned = clean_markdown_text(text.strip())

    for _ in range(3):
        # 1. Strip common prefixes like 'Final Answer:', 'The answer is:', etc.
        prefix_pattern = (
            r"^(?:final answer|the answer is|the result is)[:\s]*|^(?:answer|result):\s*"
        )
        cleaned = re.sub(prefix_pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # 2. Strip trailing period if not an ellipsis (e.g. '42.' -> '42')
        if cleaned.endswith(".") and not cleaned.endswith("..."):
            cleaned = cleaned[:-1].strip()

        # 3. Strip enclosing markdown bold/italic/code formatting
        if cleaned.startswith("**") and cleaned.endswith("**") and len(cleaned) >= 4:
            cleaned = cleaned[2:-2].strip()
        elif (
            (cleaned.startswith("*") and cleaned.endswith("*") and len(cleaned) >= 2)
            or (cleaned.startswith("`") and cleaned.endswith("`") and len(cleaned) >= 2)
        ):
            cleaned = cleaned[1:-1].strip()

        # 4. Strip enclosing quotes ("..." or '...')
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1].strip()

    return cleaned


class GAIABenchmarkAgent:
    """Decoupled, isolated GAIA benchmark evaluation runner compatible with app.py."""

    def __init__(
        self,
        max_steps: int = AGENT_MAX_STEPS,
        verbose: bool = True,
        model: Any | None = None,
        use_memory_for_testing: bool = False,
    ):
        self.max_steps = max_steps
        self.verbose = verbose
        self.model = model
        self.use_memory_for_testing = use_memory_for_testing
        self.last_trace = self._create_empty_trace()

    def _create_empty_trace(self) -> dict[str, Any]:
        """Create a clean empty trace dictionary matching app.py contract."""
        return {
            "tools_used": [],
            "steps": 0,
            "duration": 0.0,
            "tokens": 0,
            "in_tokens": 0,
            "out_tokens": 0,
            "speed": 0.0,
            "trajectory": [],
        }

    def _resolve_file(
        self, file_name: str | None = None, file_path: str | None = None
    ) -> str | None:
        """Resolve file path against direct path or local GAIA directories."""
        if file_path and os.path.exists(file_path):
            return file_path
        if file_name:
            candidates = [
                os.path.abspath(file_name),
                os.path.abspath(os.path.join("GAIA", "2023", "validation", file_name)),
                os.path.abspath(os.path.join("GAIA", "2023", "test", file_name)),
                os.path.abspath(os.path.join("GAIA", file_name)),
            ]
            for c in candidates:
                if os.path.exists(c):
                    return c
        return None

    def __call__(
        self,
        question: str,
        file_name: str | None = None,
        file_path: str | None = None,
        task_id: str | None = None,
    ) -> str:
        """Run isolated evaluation on a single question and return sanitized answer."""
        start_time = time.perf_counter()
        self.last_trace = self._create_empty_trace()

        prompt = ""
        resolved_file = self._resolve_file(file_name=file_name, file_path=file_path)

        if resolved_file:
            prompt += (
                f"Attached File Context: You have access to a local file at '{resolved_file}'. "
                "You can inspect, read, and process it using the `python_interpreter` tool.\n\n"
            )
        elif file_name:
            prompt += (
                f"Attached File Context: The question references file '{file_name}', "
                "but it could not be located locally.\n\n"
            )

        prompt += f"Question: {question}\n\n"
        prompt += (
            "CRITICAL INSTRUCTION: You are taking a strict exact-match benchmark test (GAIA). "
            "You MUST output ONLY the final answer without ANY conversational filler, pleasantries, or explanations. "
            "For example, if the answer is 42, just output '42'. If the answer is 'John Doe', output 'John Doe'. "
            "Do not write 'The answer is 42' or 'Here is the result'."
        )

        # Isolated ephemeral session per benchmark run
        session_id = f"gaia_{task_id or uuid.uuid4().hex[:8]}"

        # Instantiate decoupled agent with ephemeral memory and cold benchmark prompt
        agent = LilyAgent(
            session_id=session_id,
            max_steps=self.max_steps,
            model=self.model,
            use_memory_for_testing=True,  # Always ephemeral MemorySaver for benchmark tasks
            system_prompt=GAIA_SYSTEM_PROMPT,
            enable_user_facts=False,
        )

        raw_answer = ""
        tools_called: list[str] = []
        max_step_reached = 0
        trajectory: list[dict[str, Any]] = []
        active_tool_calls: dict[int, dict[str, Any]] = {}

        for event_type, data in agent.chat_stream(prompt):
            if event_type == "token":
                raw_answer += data

            elif event_type == "tool_start":
                raw_answer = ""
                func_name = data.get("name", "")
                func_args = data.get("args", {})
                step_num = data.get("step", 1)
                max_step_reached = max(max_step_reached, step_num)

                active_tool_calls[step_num] = {
                    "tool": func_name,
                    "args": func_args,
                    "start": time.perf_counter(),
                }

                if func_name not in tools_called:
                    tools_called.append(func_name)

                if self.verbose:
                    print(f"    🔍 [Step {step_num}] Calling Tool: {func_name}")
                    if func_name == "python_interpreter" and "code" in func_args:
                        code_snippet = (
                            func_args["code"].strip().replace("\n", "\n        ")
                        )
                        print(f"        └─ Code:\n        {code_snippet}")
                    else:
                        print(f"        └─ Args: {func_args}")

            elif event_type == "tool_end":
                func_name = data.get("name", "")
                duration = data.get("duration", 0.0)
                char_count = data.get("char_count", 0)
                step_num = data.get("step", 1)
                res_text = data.get("result", "")

                info = active_tool_calls.pop(step_num, {})
                trajectory.append({
                    "step": step_num,
                    "tool": func_name,
                    "args": info.get("args", {}),
                    "result_snippet": res_text[:200] if len(res_text) > 200 else res_text,
                    "duration": round(duration, 3),
                })

                if self.verbose:
                    print(
                        f"    ✔ Finished {func_name} in {duration:.2f}s ({char_count:,} chars)"
                    )

            elif event_type == "usage":
                total_tok = data.get("total_tokens", 0)
                in_tok = data.get("input_tokens", 0)
                out_tok = data.get("output_tokens", 0)
                speed = data.get("tokens_per_sec", 0.0)
                self.last_trace.update({
                    "tokens": total_tok,
                    "in_tokens": in_tok,
                    "out_tokens": out_tok,
                    "speed": speed,
                })
                if self.verbose:
                    print(
                        f"    ⚡ {total_tok:,} tokens ({in_tok:,} in / {out_tok:,} out) • {speed:.1f} tok/s"
                    )

            elif event_type == "error" and self.verbose:
                print(f"    ❌ Error during step: {data}")

        total_duration = time.perf_counter() - start_time
        clean_ans = sanitize_gaia_answer(raw_answer)

        self.last_trace.update({
            "tools_used": tools_called,
            "steps": max_step_reached if max_step_reached > 0 else 1,
            "duration": total_duration,
            "trajectory": trajectory,
        })

        return clean_ans
