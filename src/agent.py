import json
import re
import time
from collections.abc import Generator
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama

from src.config import (
    AGENT_MAX_STEPS,
    BASE_SYSTEM_PROMPT,
    DEFAULT_SESSION_ID,
    MEMORY_WINDOW_SIZE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from src.db import (
    ensure_session,
    get_all_facts,
    get_recent_messages,
    init_db,
    save_message,
)
from src.tools import ALL_TOOLS, filter_tools_by_intent
from src.utils import format_tool_result


class LilyAgent:
    def __init__(
        self,
        session_id: str = DEFAULT_SESSION_ID,
        max_steps: int = AGENT_MAX_STEPS,
        window_size: int = MEMORY_WINDOW_SIZE,
    ):
        self.session_id = session_id
        self.max_steps = max_steps
        self.window_size = window_size

        # 1. Initialize Database & Session
        try:
            init_db()
            ensure_session(self.session_id)
            self.db_available = True
        except Exception as e:
            print(f"[Info] PostgreSQL not connected ({e}). Running in ephemeral mode.")
            self.db_available = False

        # 2. Setup LLM & Tools Catalog
        self.llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        self.tools = ALL_TOOLS
        self.tools_by_name = {t.name: t for t in self.tools}

        # 3. Session Statistics & Telemetry Tracker
        self.session_stats = {
            "total_turns": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_eval_duration_sec": 0.0,
        }

        # 4. Build Initial Context (System Prompt + History Restore)
        self.messages: list[BaseMessage] = []
        self._refresh_context()

    @staticmethod
    def _parse_fallback_tool_calls(text: str) -> list[dict[str, Any]]:
        """Extract and convert raw pseudo tool syntax (e.g. 'call:search_web{query:...}') into structured tool calls."""
        if not text or not isinstance(text, str):
            return []

        calls: list[dict[str, Any]] = []

        # Match 'call:tool_name{args}' or 'call:tool_name(args)'
        call_pattern = re.compile(r"call:([a-zA-Z0-9_]+)\s*[\{\(]([^}\)]+)[\}\)]")
        for match in call_pattern.finditer(text):
            tool_name = match.group(1).strip()
            args_raw = match.group(2).strip()
            args_dict: dict[str, Any] = {}

            # 1. Try direct JSON parsing
            try:
                candidate = "{" + args_raw + "}" if not args_raw.startswith("{") else args_raw
                parsed_json = json.loads(candidate)
                if isinstance(parsed_json, dict):
                    args_dict = parsed_json
            except Exception:
                pass

            # 2. Try unquoted key:value splitting
            if not args_dict and ":" in args_raw:
                parts = args_raw.split(":", 1)
                k = parts[0].strip().strip('"\' ')
                v = parts[1].strip().strip('"\' ')
                args_dict = {k: v}
            elif not args_dict and args_raw:
                args_dict = {"query": args_raw.strip('"\' ')}

            calls.append({
                "name": tool_name,
                "args": args_dict,
                "id": f"fallback_{tool_name}_{len(calls) + 1}",
            })

        return calls

    def _build_system_prompt(self) -> str:
        """Construct system prompt dynamically with active user facts from memory."""
        prompt = BASE_SYSTEM_PROMPT
        if self.db_available:
            try:
                facts = get_all_facts()
                if facts:
                    fact_lines = [f"- [{f['category']}] {f['key']}: {f['value']}" for f in facts]
                    prompt += (
                        "\n\n[Known Facts About the User]:\n"
                        + "\n".join(fact_lines)
                        + "\n(Use these facts naturally in conversation. Update them using `save_fact` when the user provides new preferences/info)."
                    )
            except Exception as e:
                print(f"[Warning] Failed to load user facts for prompt: {e}")
        return prompt

    def _refresh_context(self) -> None:
        """Reload dynamic system prompt and restore sliding window history from DB."""
        system_prompt = self._build_system_prompt()
        self.messages = [SystemMessage(content=system_prompt)]

        if self.db_available:
            try:
                recent_records = get_recent_messages(self.session_id, limit=self.window_size)
                for rec in recent_records:
                    role = rec["role"]
                    content = rec["content"]
                    raw_tc = rec["tool_calls"]
                    if isinstance(raw_tc, list):
                        tool_calls = raw_tc
                    elif isinstance(raw_tc, str):
                        tool_calls = json.loads(raw_tc)
                    else:
                        tool_calls = []
                    tool_call_id = rec.get("tool_call_id")

                    if role == "human":
                        self.messages.append(HumanMessage(content=content))
                    elif role == "ai":
                        self.messages.append(AIMessage(content=content, tool_calls=tool_calls))
                    elif role == "tool":
                        self.messages.append(ToolMessage(content=content, tool_call_id=tool_call_id or ""))
            except Exception as e:
                print(f"[Warning] Failed to restore session history: {e}")

    def reset_context(self) -> None:
        """Reset the working context messages to just the base system prompt (stateless turn)."""
        self.messages = [SystemMessage(content=self._build_system_prompt())]

    def chat_stream(self, user_input: str) -> Generator[tuple[str, Any], None, None]:
        """Stream autonomous ReAct loop execution yielding (event_type, data).

        Events:
            - ("token", text_chunk)
            - ("tool_start", {"name": ..., "args": ..., "step": ...})
            - ("tool_end", {"name": ..., "result": ..., "duration": ..., "step": ...})
            - ("usage", {"input_tokens": ..., "output_tokens": ..., "total_tokens": ..., "tokens_per_sec": ...})
            - ("error", error_message)
        """
        turn_start_time = time.perf_counter()
        turn_in_tokens = 0
        turn_out_tokens = 0
        turn_eval_duration_sec = 0.0

        # 1. Update system prompt with fresh facts
        self.messages[0] = SystemMessage(content=self._build_system_prompt())

        # 2. Append & Persist User Message
        user_msg = HumanMessage(content=user_input)
        self.messages.append(user_msg)
        if self.db_available:
            try:
                save_message(self.session_id, role="human", content=user_input)
            except Exception as e:
                print(f"[Warning] Failed to persist human message: {e}")

        # 3. Dynamic Tool Selection (Select 4-6 tools relevant to intent)
        active_tools = filter_tools_by_intent(user_input, self.tools)
        llm_with_tools = self.llm.bind_tools(active_tools)

        # 4. Autonomous Streaming Loop
        for step_idx in range(self.max_steps):
            full_response: Any = None
            has_tool_calls = False

            try:
                for chunk in llm_with_tools.stream(self.messages):
                    full_response = chunk if full_response is None else full_response + chunk

                    # If this chunk produces text and is not pseudo tool-call syntax, yield token
                    if chunk.content and full_response and not getattr(full_response, "tool_calls", None):
                        accumulated = str(full_response.content).strip()
                        if not accumulated.startswith("call:") and not accumulated.startswith("<tool_call>"):
                            yield ("token", chunk.content)

            except Exception as e:
                yield ("error", f"LLM stream error: {e!s}")
                return

            if full_response is None:
                yield ("error", "Received empty response from model.")
                return

            # Check for Native or Fallback Tool Calls
            tool_calls = list(getattr(full_response, "tool_calls", None) or [])
            if not tool_calls and full_response.content:
                fallback_calls = self._parse_fallback_tool_calls(str(full_response.content))
                if fallback_calls:
                    tool_calls = fallback_calls

            # Extract Token Usage & Performance Metadata from this step
            usage = getattr(full_response, "usage_metadata", None) or {}
            resp_meta = getattr(full_response, "response_metadata", None) or {}

            step_in_tokens = usage.get("input_tokens") or resp_meta.get("prompt_eval_count") or 0
            step_out_tokens = usage.get("output_tokens") or resp_meta.get("eval_count") or 0
            step_eval_duration = (resp_meta.get("eval_duration") or 0) / 1e9

            turn_in_tokens += step_in_tokens
            turn_out_tokens += step_out_tokens
            turn_eval_duration_sec += step_eval_duration

            # Append assistant response to messages & DB
            ai_content = "" if tool_calls and not getattr(full_response, "tool_calls", None) else str(full_response.content)
            ai_msg = AIMessage(content=ai_content, tool_calls=tool_calls)
            self.messages.append(ai_msg)

            if self.db_available:
                try:
                    save_message(
                        self.session_id,
                        role="ai",
                        content=ai_content,
                        tool_calls=tool_calls if tool_calls else None,
                    )
                except Exception as e:
                    print(f"[Warning] Failed to persist AI message: {e}")

            # Execute Tool Calls
            if tool_calls:
                has_tool_calls = True
                for call in tool_calls:
                    tool_name = call["name"]
                    tool_args = call.get("args", {})
                    call_id = call.get("id", f"call_{tool_name}_{step_idx}")

                    yield (
                        "tool_start",
                        {
                            "name": tool_name,
                            "args": tool_args,
                            "step": step_idx + 1,
                            "max_steps": self.max_steps,
                        },
                    )

                    t_start = time.perf_counter()
                    tool = self.tools_by_name.get(tool_name)
                    if tool:
                        try:
                            raw_result = tool.invoke(tool_args)
                            clean_result = format_tool_result(raw_result)
                        except Exception as te:
                            clean_result = f"Error executing {tool_name}: {te!s}"
                    else:
                        clean_result = f"Error: Tool '{tool_name}' not found."
                    duration = time.perf_counter() - t_start

                    yield (
                        "tool_end",
                        {
                            "name": tool_name,
                            "result": clean_result,
                            "duration": duration,
                            "step": step_idx + 1,
                            "char_count": len(clean_result),
                        },
                    )

                    tool_msg = ToolMessage(content=clean_result, tool_call_id=call_id)
                    self.messages.append(tool_msg)

                    if self.db_available:
                        try:
                            save_message(
                                self.session_id,
                                role="tool",
                                content=clean_result,
                                tool_call_id=call_id,
                            )
                        except Exception as e:
                            print(f"[Warning] Failed to persist tool message: {e}")

            # If no tool calls was made in this turn, we are finished!
            if not has_tool_calls:
                break

        # Calculate Turn Metrics & Update Session Accumulator
        turn_duration = time.perf_counter() - turn_start_time
        turn_total_tokens = turn_in_tokens + turn_out_tokens
        tok_per_sec = (
            turn_out_tokens / turn_eval_duration_sec
            if turn_eval_duration_sec > 0
            else 0.0
        )

        self.session_stats["total_turns"] += 1
        self.session_stats["total_input_tokens"] += turn_in_tokens
        self.session_stats["total_output_tokens"] += turn_out_tokens
        self.session_stats["total_tokens"] += turn_total_tokens
        self.session_stats["total_eval_duration_sec"] += turn_eval_duration_sec

        # Emit Token Usage Telemetry Event
        yield (
            "usage",
            {
                "input_tokens": turn_in_tokens,
                "output_tokens": turn_out_tokens,
                "total_tokens": turn_total_tokens,
                "tokens_per_sec": tok_per_sec,
                "duration_sec": turn_duration,
                "eval_duration_sec": turn_eval_duration_sec,
            },
        )

        self._trim_working_context()

    def get_session_stats(self) -> dict[str, Any]:
        """Return aggregated session statistics including token usage and average speed."""
        stats = self.session_stats.copy()
        total_eval_dur = stats.get("total_eval_duration_sec", 0.0)
        total_out = stats.get("total_output_tokens", 0)
        stats["avg_tokens_per_sec"] = (
            total_out / total_eval_dur if total_eval_dur > 0 else 0.0
        )
        stats["active_session_id"] = self.session_id
        stats["working_context_messages"] = len(self.messages)
        return stats

    def chat(self, user_input: str) -> str:
        """Synchronous chat fallback."""
        tokens = []
        for event_type, data in self.chat_stream(user_input):
            if event_type == "token":
                tokens.append(data)
            elif event_type == "error":
                return f"Error: {data}"
        return "".join(tokens)

    def _trim_working_context(self) -> None:
        """Maintain sliding window size in active memory (keeping SystemMessage at index 0)."""
        if len(self.messages) > (self.window_size + 1):
            self.messages = [self.messages[0]] + self.messages[-self.window_size :]
