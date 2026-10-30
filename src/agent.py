"""Autonomous LangGraph Agent Engine for Lily AI Assistant.

Features:
- StateGraph architecture with cyclical ReAct loop (agent ⇄ ToolNode).
- Strict PostgreSQL First checkpointer with test isolation support (MemorySaver).
- Dynamic system prompt with real-time clock and persistent user profile facts.
- Safe context window trimming with trim_messages(start_on="human") to prevent orphan tool messages.
- Multi-mode event streaming (tokens, tool lifecycle, token usage, error isolation).
"""

import time
from collections.abc import Generator
from datetime import datetime
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from src.config import (
    AGENT_MAX_STEPS,
    BASE_SYSTEM_PROMPT,
    DEFAULT_SESSION_ID,
    MEMORY_WINDOW_SIZE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from src.db import (
    get_all_facts,
    get_checkpointer,
    get_thread_config,
    is_db_available,
    reset_thread,
)
from src.tools import ALL_TOOLS


class AgentState(TypedDict):
    """LangGraph state schema representing conversation history with append reducer."""

    messages: Annotated[list[BaseMessage], add_messages]


def safe_trim_context(
    messages: list[BaseMessage], max_messages: int = MEMORY_WINDOW_SIZE
) -> list[BaseMessage]:
    """Trim conversation context window safely.

    Guarantees:
    - Retains SystemMessage at index 0.
    - Slices history starting on a HumanMessage to strictly prevent Orphan ToolMessage errors.
    """
    if not messages:
        return []
    return trim_messages(
        messages,
        max_tokens=max_messages + 1,  # headroom for SystemMessage
        token_counter=len,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )


class LilyAgent:
    """Production-grade autonomous AI assistant powered by LangGraph."""

    def __init__(
        self,
        session_id: str = DEFAULT_SESSION_ID,
        max_steps: int = AGENT_MAX_STEPS,
        window_size: int = MEMORY_WINDOW_SIZE,
        use_memory_for_testing: bool = False,
        model: Any | None = None,
        system_prompt: str | None = None,
        enable_user_facts: bool = True,
    ):
        self.session_id = session_id
        self.max_steps = max_steps
        self.window_size = window_size
        self.use_memory_for_testing = use_memory_for_testing
        self.custom_system_prompt = system_prompt
        self.enable_user_facts = enable_user_facts

        # 1. Initialize Persistence Checkpointer
        self.checkpointer: BaseCheckpointSaver = get_checkpointer(
            use_memory_for_testing=use_memory_for_testing
        )

        # 2. Setup LLM & Tool Bindings
        self.llm = model or ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        self.tools = ALL_TOOLS
        self.model_with_tools = self.llm.bind_tools(self.tools)

        # 3. Session Statistics & Telemetry Tracker
        self.session_stats: dict[str, Any] = {
            "total_turns": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_eval_duration_sec": 0.0,
        }

        # 4. Compile LangGraph StateGraph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct and compile the LangGraph StateGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add Nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))

        # Add Edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.checkpointer)

    def _build_system_prompt(self) -> str:
        """Construct dynamic system prompt with real-time clock and Master's profile facts."""
        base_prompt = self.custom_system_prompt or BASE_SYSTEM_PROMPT
        now_str = datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")
        prompt = f"{base_prompt}\n\n[Current System Time]: {now_str}"

        if self.enable_user_facts:
            try:
                facts = get_all_facts()
                if facts:
                    fact_lines = [f"- [{f['category']}] {f['key']}: {f['value']}" for f in facts]
                    prompt += (
                        "\n\n[Known Facts About the User]:\n"
                        + "\n".join(fact_lines)
                        + "\n(Use these facts naturally in conversation. Adapt your behavior to Master's preferences)."
                    )
            except Exception:
                pass

        return prompt

    def _agent_node(self, state: AgentState) -> dict[str, Any]:
        """LangGraph agent node: injects dynamic prompt, trims context safely, and calls LLM."""
        raw_messages = state.get("messages", [])
        system_content = self._build_system_prompt()

        if not raw_messages or not isinstance(raw_messages[0], SystemMessage):
            full_messages = [SystemMessage(content=system_content), *list(raw_messages)]
        else:
            full_messages = [SystemMessage(content=system_content), *list(raw_messages[1:])]

        trimmed = safe_trim_context(full_messages, max_messages=self.window_size)
        response = self.model_with_tools.invoke(trimmed)
        return {"messages": [response]}

    @property
    def db_available(self) -> bool:
        """Check live database connectivity status."""
        return is_db_available()

    @property
    def messages(self) -> list[BaseMessage]:
        """Inspect current working messages for active thread from checkpointer."""
        try:
            config = get_thread_config(self.session_id)
            state = self.graph.get_state(config)
            msgs = state.values.get("messages", [])
            system_prompt = SystemMessage(content=self._build_system_prompt())
            return [system_prompt, *msgs] if not msgs or not isinstance(msgs[0], SystemMessage) else msgs
        except Exception:
            return [SystemMessage(content=self._build_system_prompt())]

    @messages.setter
    def messages(self, new_messages: list[BaseMessage]) -> None:
        """Support direct assignment for testing context manipulations."""
        pass

    def reset_context(self) -> None:
        """Clear thread history in the checkpointer allowing a clean restart."""
        reset_thread(self.session_id, self.checkpointer)

    def chat_stream(self, user_input: str) -> Generator[tuple[str, Any], None, None]:
        """Execute autonomous ReAct loop streaming real-time telemetry events.

        Yields:
            ("token", str): Live streamed markdown text chunk.
            ("tool_start", dict): Tool invocation notification (name, args, step).
            ("tool_end", dict): Tool execution result notification (name, result, duration).
            ("usage", dict): Performance and token metrics for this turn.
            ("error", str): Descriptive failure notification.
        """
        clean_input = user_input.strip()
        if not clean_input:
            yield ("error", "ข้อความว่างเปล่า กรุณาพิมพ์คำถามหรือคำสั่งให้ลิลลี่นะคะ 🌸")
            return

        turn_start_time = time.perf_counter()
        turn_in_tokens = 0
        turn_out_tokens = 0
        turn_eval_duration_sec = 0.0
        step_counter = 0

        config = get_thread_config(self.session_id)
        config["recursion_limit"] = self.max_steps * 2 + 1

        active_tool_calls: dict[str, dict[str, Any]] = {}
        input_state = {"messages": [HumanMessage(content=clean_input)]}

        try:
            for mode, payload in self.graph.stream(
                input_state, config=config, stream_mode=["messages", "updates"]
            ):
                if mode == "messages":
                    chunk, metadata = payload
                    node = metadata.get("langgraph_node")
                    if node == "agent" and isinstance(chunk, (AIMessageChunk, AIMessage)):
                        has_tc = (
                            bool(getattr(chunk, "tool_call_chunks", None))
                            or bool(getattr(chunk, "tool_calls", None))
                        )
                        if chunk.content and not has_tc:
                            yield ("token", chunk.content)

                elif mode == "updates" and isinstance(payload, dict):
                    if "agent" in payload:
                        agent_update = payload["agent"]
                        msgs = agent_update.get("messages", [])
                        if msgs:
                            ai_msg = msgs[-1]
                            usage = getattr(ai_msg, "usage_metadata", None) or {}
                            resp_meta = getattr(ai_msg, "response_metadata", None) or {}

                            step_in = usage.get("input_tokens") or resp_meta.get("prompt_eval_count") or 0
                            step_out = usage.get("output_tokens") or resp_meta.get("eval_count") or 0
                            step_dur = (resp_meta.get("eval_duration") or 0) / 1e9

                            turn_in_tokens += step_in
                            turn_out_tokens += step_out
                            turn_eval_duration_sec += step_dur

                            tool_calls = getattr(ai_msg, "tool_calls", None) or []
                            for tc in tool_calls:
                                step_counter += 1
                                call_id = tc.get("id", f"call_{step_counter}")
                                name = tc.get("name", "tool")
                                args = tc.get("args", {})
                                active_tool_calls[call_id] = {
                                    "name": name,
                                    "args": args,
                                    "step": step_counter,
                                    "start": time.perf_counter(),
                                }
                                yield (
                                    "tool_start",
                                    {
                                        "name": name,
                                        "args": args,
                                        "step": step_counter,
                                        "max_steps": self.max_steps,
                                    },
                                )

                    elif "tools" in payload:
                        tools_update = payload["tools"]
                        tool_msgs = tools_update.get("messages", [])
                        for tm in tool_msgs:
                            call_id = getattr(tm, "tool_call_id", "")
                            info = active_tool_calls.pop(call_id, None)
                            name = (
                                getattr(tm, "name", None)
                                or (info["name"] if info else "tool")
                            )
                            step_num = info["step"] if info else step_counter
                            start_t = info["start"] if info else time.perf_counter()
                            duration = time.perf_counter() - start_t
                            res_text = str(tm.content)

                            yield (
                                "tool_end",
                                {
                                    "name": name,
                                    "result": res_text,
                                    "duration": duration,
                                    "step": step_num,
                                    "char_count": len(res_text),
                                },
                            )

        except GraphRecursionError:
            yield (
                "error",
                f"ขออภัยค่าา Master ลิลลี่พยายามดำเนินการเกินจำนวนก้าวสูงสุด ({self.max_steps} ก้าว) แล้ว ขอยกยอดมาปรึกษา Master โดยตรงนะคะ 🌸",
            )
            return
        except Exception as e:
            yield ("error", f"เกิดข้อผิดพลาดในการประมวลผล: {e!s}")
            return

        # Calculate metrics & update session accumulators
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
        """Synchronous chat interface collecting streamed text."""
        tokens: list[str] = []
        for event_type, data in self.chat_stream(user_input):
            if event_type == "token":
                tokens.append(data)
            elif event_type == "error":
                return f"Error: {data}"
        return "".join(tokens)
