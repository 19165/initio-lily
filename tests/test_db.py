import pytest
from unittest.mock import patch, MagicMock
from src.db import is_db_available, init_db, ensure_session, save_message, get_recent_messages, upsert_fact, get_all_facts, delete_fact


class TestDatabaseAvailability:
    def test_is_db_available_returns_bool(self):
        result = is_db_available()
        assert isinstance(result, bool)

    def test_init_db_raises_when_offline(self):
        with patch("src.db.is_db_available", return_value=False):
            with pytest.raises(ConnectionError) as exc_info:
                init_db()
            assert "PostgreSQL is not reachable" in str(exc_info.value)


@pytest.mark.skipif(not is_db_available(), reason="PostgreSQL is not running locally")
class TestDatabaseOperations:
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
