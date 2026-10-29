"""Comprehensive test suite for LilyAgent and LangGraph ReAct engine.

Tests verify:
- Context trimming contracts (start_on='human', SystemMessage at index 0).
- StateGraph transitions (START -> agent -> END, ReAct tool cycles).
- Recursion limit safety breaker.
- Telemetry event emissions (token, tool_start, tool_end, usage, error).
- Isolated execution using use_memory_for_testing=True.
"""

from unittest.mock import patch

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.agent import LilyAgent, safe_trim_context


class MockChatModel:
    """Deterministic mock chat model for LangGraph workflow unit testing."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.call_count = 0

    def bind_tools(self, tools: list[object]) -> "MockChatModel":
        return self

    def invoke(self, messages: list[BaseMessage], **kwargs: object) -> AIMessage:
        res = self.responses[self.call_count]
        if self.call_count < len(self.responses) - 1:
            self.call_count += 1
        return res


class TestLilyAgentStateAndTrimming:
    """Unit tests for agent state, context trimming, and configuration."""

    def test_init_defaults(self):
        agent = LilyAgent(
            session_id="test_init",
            max_steps=4,
            window_size=6,
            use_memory_for_testing=True,
        )
        assert agent.session_id == "test_init"
        assert agent.max_steps == 4
        assert agent.window_size == 6
        assert agent.use_memory_for_testing is True
        assert agent.checkpointer is not None
        assert len(agent.messages) >= 1
        assert isinstance(agent.messages[0], SystemMessage)

    def test_safe_trim_context_preserves_system_and_starts_on_human(self):
        msgs: list[BaseMessage] = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="query 1"),
            AIMessage(
                content="",
                tool_calls=[{"name": "t", "args": {}, "id": "1"}],
            ),
            ToolMessage(content="tool output", tool_call_id="1"),
            HumanMessage(content="query 2"),
            AIMessage(content="final answer"),
        ]

        # Slicing with max_messages=2 should keep system + start on query 2 + final answer
        trimmed = safe_trim_context(msgs, max_messages=2)
        assert len(trimmed) == 3
        assert isinstance(trimmed[0], SystemMessage)
        assert trimmed[0].content == "system prompt"
        assert isinstance(trimmed[1], HumanMessage)
        assert trimmed[1].content == "query 2"
        assert isinstance(trimmed[2], AIMessage)
        assert trimmed[2].content == "final answer"

    def test_safe_trim_context_empty(self):
        assert safe_trim_context([]) == []

    def test_build_system_prompt_with_facts_and_time(self):
        agent = LilyAgent(session_id="test_prompt", use_memory_for_testing=True)
        with patch("src.agent.get_all_facts") as mock_facts:
            mock_facts.return_value = [
                {"category": "preference", "key": "coffee", "value": "latte"}
            ]
            prompt = agent._build_system_prompt()
            assert "[Current System Time]:" in prompt
            assert "[Known Facts About the User]:" in prompt
            assert "- [preference] coffee: latte" in prompt

    def test_reset_context(self):
        agent = LilyAgent(session_id="test_reset", use_memory_for_testing=True)
        mock_model = MockChatModel([AIMessage(content="Hello Master! 🌸")])
        agent.model_with_tools = mock_model

        agent.chat("Hi")
        assert len(agent.messages) > 1

        agent.reset_context()
        assert len(agent.messages) == 1
        assert isinstance(agent.messages[0], SystemMessage)

    def test_get_session_stats(self):
        agent = LilyAgent(session_id="test_stats", use_memory_for_testing=True)
        stats = agent.get_session_stats()

        assert stats["active_session_id"] == "test_stats"
        assert "total_turns" in stats
        assert "total_input_tokens" in stats
        assert "total_output_tokens" in stats
        assert "total_tokens" in stats
        assert "avg_tokens_per_sec" in stats

    def test_empty_input_returns_error(self):
        agent = LilyAgent(session_id="test_empty", use_memory_for_testing=True)
        events = list(agent.chat_stream("   "))
        assert len(events) == 1
        assert events[0][0] == "error"
        assert "ข้อความว่างเปล่า" in events[0][1]


class TestLangGraphAgentTransitions:
    """Integration tests for LangGraph state machine execution and tool cycles."""

    def test_single_turn_direct_answer(self):
        mock_model = MockChatModel([
            AIMessage(
                content="สวัสดีค่าา Master ลิลลี่พร้อมช่วยแล้วค่ะ 🌸",
                usage_metadata={
                    "input_tokens": 15,
                    "output_tokens": 10,
                    "total_tokens": 25,
                },
            )
        ])
        agent = LilyAgent(
            session_id="test_single_turn",
            model=mock_model,
            use_memory_for_testing=True,
        )

        events = list(agent.chat_stream("สวัสดี"))
        event_dict = {e[0]: e[1] for e in events}

        assert "token" in event_dict
        assert "สวัสดีค่าา Master" in event_dict["token"]
        assert "usage" in event_dict
        assert event_dict["usage"]["input_tokens"] == 15
        assert event_dict["usage"]["output_tokens"] == 10
        assert event_dict["usage"]["total_tokens"] == 25

    def test_react_tool_cycle(self):
        mock_model = MockChatModel([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "workspace_notes",
                    "args": {"action": "list"},
                    "id": "call_react_1",
                }],
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            ),
            AIMessage(
                content="พบโน้ตเรียบร้อยแล้วค่าา 🌸",
                usage_metadata={
                    "input_tokens": 25,
                    "output_tokens": 8,
                    "total_tokens": 33,
                },
            ),
        ])
        agent = LilyAgent(
            session_id="test_react_cycle",
            model=mock_model,
            use_memory_for_testing=True,
        )

        events = list(agent.chat_stream("ลิลลี่ดูโน้ตให้หน่อย"))
        event_types = [e[0] for e in events]

        assert "tool_start" in event_types
        assert "tool_end" in event_types
        assert "token" in event_types
        assert "usage" in event_types

        tool_start = [e[1] for e in events if e[0] == "tool_start"][0]
        assert tool_start["name"] == "workspace_notes"
        assert tool_start["args"] == {"action": "list"}

        tool_end = [e[1] for e in events if e[0] == "tool_end"][0]
        assert tool_end["name"] == "workspace_notes"
        assert "result" in tool_end
        assert tool_end["step"] == 1

        # Check token usage accumulated across both steps
        usage = [e[1] for e in events if e[0] == "usage"][0]
        assert usage["input_tokens"] == 35  # 10 + 25
        assert usage["output_tokens"] == 13  # 5 + 8
        assert usage["total_tokens"] == 48

    def test_recursion_limit_breaker(self):
        # Infinite tool calling model
        mock_model = MockChatModel([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "workspace_notes",
                    "args": {"action": "list"},
                    "id": "call_loop",
                }],
            )
        ])
        agent = LilyAgent(
            session_id="test_recursion",
            max_steps=1,
            model=mock_model,
            use_memory_for_testing=True,
        )

        events = list(agent.chat_stream("Infinite loop query"))
        event_types = [e[0] for e in events]

        assert "error" in event_types
        error_msg = [e[1] for e in events if e[0] == "error"][0]
        assert "เกินจำนวนก้าวสูงสุด" in error_msg


class TestAgentStreamingTelemetry:
    """Tests for chat synchronous wrappers and telemetry details."""

    def test_chat_sync_success(self):
        mock_model = MockChatModel([AIMessage(content="ทำงานเสร็จเรียบร้อยแล้วค่ะ 🌸")])
        agent = LilyAgent(
            session_id="test_sync",
            model=mock_model,
            use_memory_for_testing=True,
        )
        result = agent.chat("เสร็จหรือยัง")
        assert "ทำงานเสร็จเรียบร้อยแล้วค่ะ" in result

    def test_chat_sync_error(self):
        agent = LilyAgent(session_id="test_sync_err", use_memory_for_testing=True)
        result = agent.chat("   ")
        assert result.startswith("Error:")

    def test_db_available_property(self):
        agent = LilyAgent(session_id="test_db_avail", use_memory_for_testing=True)
        with patch("src.agent.is_db_available", return_value=True):
            assert agent.db_available is True
        with patch("src.agent.is_db_available", return_value=False):
            assert agent.db_available is False
