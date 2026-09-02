import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.agent import LilyAgent


class TestLilyAgentContextAndStats:
    def test_init_defaults(self):
        agent = LilyAgent(session_id="test_init_session", max_steps=3, window_size=5)
        assert agent.session_id == "test_init_session"
        assert agent.max_steps == 3
        assert agent.window_size == 5
        assert len(agent.messages) >= 1
        assert isinstance(agent.messages[0], SystemMessage)

    def test_trim_working_context(self):
        agent = LilyAgent(session_id="test_trim_session", window_size=4)
        agent.messages = [SystemMessage(content="System prompt")]

        # Add 6 human messages (exceeding window_size of 4)
        for i in range(6):
            agent.messages.append(HumanMessage(content=f"Message {i}"))

        assert len(agent.messages) == 7
        agent._trim_working_context()

        # Should be trimmed to 1 (System) + 4 (recent messages) = 5
        assert len(agent.messages) == 5
        assert isinstance(agent.messages[0], SystemMessage)
        assert agent.messages[0].content == "System prompt"
        # Most recent messages should be Message 2, 3, 4, 5
        assert agent.messages[1].content == "Message 2"
        assert agent.messages[-1].content == "Message 5"

    def test_reset_context(self):
        agent = LilyAgent(session_id="test_reset_session")
        agent.messages.append(HumanMessage(content="Hello"))
        agent.messages.append(AIMessage(content="Hi there!"))
        assert len(agent.messages) > 1

        agent.reset_context()
        assert len(agent.messages) == 1
        assert isinstance(agent.messages[0], SystemMessage)

    def test_get_session_stats(self):
        agent = LilyAgent(session_id="test_stats_session")
        stats = agent.get_session_stats()

        assert "total_turns" in stats
        assert "total_input_tokens" in stats
        assert "total_output_tokens" in stats
        assert "total_tokens" in stats
        assert "avg_tokens_per_sec" in stats
        assert stats["active_session_id"] == "test_stats_session"
        assert stats["working_context_messages"] == len(agent.messages)

    def test_build_system_prompt_with_facts(self):
        agent = LilyAgent(session_id="test_prompt_session")
        with patch("src.agent.get_all_facts") as mock_get_facts:
            mock_get_facts.return_value = [
                {"category": "user_pref", "key": "color", "value": "pink"}
            ]
            agent.db_available = True
            prompt = agent._build_system_prompt()
            assert "[Known Facts About the User]:" in prompt
            assert "[user_pref] color: pink" in prompt
