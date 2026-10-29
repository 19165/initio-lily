"""Comprehensive unit and contract test suite for Database Persistence & LangGraph Checkpointer.

Coverage:
1. Two-Stage Database Health Check (Socket probe + SQL ping handshake).
2. LangGraph Checkpointer Factory (PostgresSaver in prod, MemorySaver in tests).
3. Thread and Session Configuration Helpers (get_thread_config, reset_thread).
4. Long-term User Facts CRUD Operations (Live & Mocked fallback).
"""

from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.db import (
    delete_fact,
    ensure_session,
    get_all_facts,
    get_checkpointer,
    get_recent_messages,
    get_thread_config,
    init_db,
    is_db_available,
    reset_thread,
    save_message,
    upsert_fact,
)


class TestDatabaseHealthCheck:
    """Verifies two-stage health check and fail-fast diagnostics."""

    def test_is_db_available_returns_bool(self):
        result = is_db_available()
        assert isinstance(result, bool)

    def test_is_db_available_handles_socket_failure(self):
        with patch("socket.create_connection", side_effect=ConnectionRefusedError("Port closed")):
            assert is_db_available() is False

    def test_is_db_available_handles_sql_ping_failure(self):
        # Socket opens, but connection pool ping fails (e.g. wrong credentials / auth failure)
        mock_pool = MagicMock()
        mock_pool.connection.side_effect = TimeoutError("Connection pool timed out")
        with patch("socket.create_connection"), patch("src.db.get_pool", return_value=mock_pool):
            assert is_db_available() is False

    def test_init_db_raises_when_offline(self):
        with patch("src.db.is_db_available", return_value=False):
            with pytest.raises(ConnectionError) as exc_info:
                init_db()
            assert "PostgreSQL is not reachable" in str(exc_info.value)
            assert "docker compose up -d" in str(exc_info.value)


class TestCheckpointerLifecycle:
    """Verifies LangGraph checkpointer initialization and test isolation."""

    def test_get_checkpointer_testing_mode_returns_memory_saver(self):
        """Testing mode must return MemorySaver for fast, isolated execution."""
        saver = get_checkpointer(use_memory_for_testing=True)
        assert isinstance(saver, MemorySaver)

    def test_get_checkpointer_production_mode_raises_when_offline(self):
        """Production mode must fail-fast with clear instructions when database is offline."""
        with patch("src.db.is_db_available", return_value=False):
            with pytest.raises(ConnectionError) as exc_info:
                get_checkpointer(use_memory_for_testing=False)
            assert "PostgreSQL is not reachable" in str(exc_info.value)
            assert "docker compose up -d" in str(exc_info.value)

    def test_memory_saver_checkpoint_storage_contract(self):
        """Verify checkpointer write and read cycle contract."""
        saver = get_checkpointer(use_memory_for_testing=True)
        config = get_thread_config("test_thread_01")
        checkpoint = {
            "v": 1,
            "ts": "2026-10-29T00:00:00Z",
            "id": "chk_01",
            "channel_values": {"messages": ["hello from lily 🌸"]},
            "channel_versions": {"messages": 1},
            "versions_seen": {},
            "pending_sends": [],
        }
        # Write checkpoint
        saver.put(config, checkpoint, metadata={"step": 1}, new_versions={"messages": 1})

        # Retrieve checkpoint
        saved = saver.get(config)
        assert saved is not None
        assert saved["channel_values"]["messages"] == ["hello from lily 🌸"]


class TestThreadStateHelpers:
    """Verifies thread configuration and state reset helpers."""

    def test_get_thread_config_structure(self):
        config = get_thread_config("safe_session_99")
        assert config == {"configurable": {"thread_id": "safe_session_99", "checkpoint_ns": ""}}

    def test_get_thread_config_strips_whitespace(self):
        config = get_thread_config("   padded_session_id   ")
        assert config["configurable"]["thread_id"] == "padded_session_id"

    def test_reset_thread_with_memory_saver(self):
        saver = get_checkpointer(use_memory_for_testing=True)
        # Should not raise exception
        reset_thread("thread_to_reset", checkpointer=saver)


@pytest.mark.skipif(not is_db_available(), reason="PostgreSQL is not running locally")
class TestLiveDatabaseOperations:
    """Verifies live PostgreSQL operations when Docker container is running."""

    TEST_SESSION_ID = "pytest_test_session_lily"
    TEST_FACT_KEY = "pytest_favorite_snack"

    def test_session_lifecycle(self):
        ensure_session(self.TEST_SESSION_ID, title="Pytest Temporary Chat")
        save_message(
            self.TEST_SESSION_ID,
            role="human",
            content="Hello from pytest!",
            tool_calls=None,
        )
        save_message(
            self.TEST_SESSION_ID,
            role="ai",
            content="Hello back!",
            tool_calls=None,
        )

        messages = get_recent_messages(self.TEST_SESSION_ID, limit=5)
        assert len(messages) >= 2
        assert messages[-2]["role"] == "human"
        assert messages[-2]["content"] == "Hello from pytest!"
        assert messages[-1]["role"] == "ai"
        assert messages[-1]["content"] == "Hello back!"

    def test_facts_crud(self):
        # 1. Upsert
        upsert_fact(self.TEST_FACT_KEY, "Strawberry Cake", category="snacks")

        # 2. Retrieve
        all_facts = get_all_facts(category="snacks")
        matching = [f for f in all_facts if f["key"] == self.TEST_FACT_KEY]
        assert len(matching) == 1
        assert matching[0]["value"] == "Strawberry Cake"

        # 3. Delete
        deleted = delete_fact(self.TEST_FACT_KEY)
        assert deleted is True

        # 4. Verify deleted
        after_delete = get_all_facts(category="snacks")
        assert not any(f["key"] == self.TEST_FACT_KEY for f in after_delete)
