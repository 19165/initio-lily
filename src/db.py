"""Database persistence and LangGraph checkpointer module for Lily AI Assistant.

Features:
- Strict PostgreSQL First persistence using PostgresSaver with connection pooling.
- Fast two-stage health checks (socket probe + SQL ping) with fail-fast diagnostics.
- Testing isolation mode using MemorySaver for ultra-fast, offline unit tests.
- Long-term memory store for user facts (user_facts table).
- Thread configuration and state reset helpers for LangGraph workflows.
"""

import json
import socket
from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.config import (
    DATABASE_URL,
    POSTGRES_HOST,
    POSTGRES_PORT,
)

_pool: ConnectionPool | None = None
_postgres_saver: PostgresSaver | None = None


# ---------------------------------------------------------------------------
# 1. Connection Pool & Health Check
# ---------------------------------------------------------------------------


def get_pool() -> ConnectionPool:
    """Get or create the global psycopg connection pool."""
    global _pool
    if _pool is None or _pool.closed:
        _pool = ConnectionPool(
            conninfo=DATABASE_URL,
            min_size=1,
            max_size=10,
            open=True,
            kwargs={"row_factory": dict_row},
        )
    return _pool


def is_db_available(timeout: float = 0.5) -> bool:
    """Two-stage health check: fast socket probe followed by quick SQL ping.

    Returns:
        True only if the PostgreSQL port is open AND authentication/SQL ping succeeds.
    """
    # Stage 1: Quick socket probe (0.2s)
    try:
        with socket.create_connection((POSTGRES_HOST, POSTGRES_PORT), timeout=0.2):
            pass
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False

    # Stage 2: Quick SQL query ping (timeout limit)
    try:
        pool = get_pool()
        with pool.connection(timeout=timeout) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1;")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 2. LangGraph Checkpointer Factory & Helpers
# ---------------------------------------------------------------------------


def get_checkpointer(use_memory_for_testing: bool = False) -> BaseCheckpointSaver:
    """Retrieve the LangGraph checkpointer for conversation state persistence.

    Args:
        use_memory_for_testing: If True, returns an ephemeral MemorySaver for offline unit tests.

    Returns:
        BaseCheckpointSaver: PostgresSaver in production or MemorySaver in test mode.

    Raises:
        ConnectionError: In production if PostgreSQL is unreachable.
    """
    if use_memory_for_testing:
        return MemorySaver()

    if not is_db_available():
        raise ConnectionError(
            f"PostgreSQL is not reachable at {POSTGRES_HOST}:{POSTGRES_PORT}. "
            "Please start the database container using: docker compose up -d"
        )

    global _postgres_saver
    if _postgres_saver is None:
        _postgres_saver = PostgresSaver(get_pool())
        _postgres_saver.setup()

    return _postgres_saver


def get_thread_config(thread_id: str) -> dict[str, Any]:
    """Format the standard LangGraph configuration dictionary for a given thread/session."""
    return {"configurable": {"thread_id": thread_id.strip(), "checkpoint_ns": ""}}


def reset_thread(thread_id: str, checkpointer: Any | None = None) -> None:
    """Delete all checkpoints and state history for a given thread to allow a clean restart."""
    cp = checkpointer or get_checkpointer(use_memory_for_testing=not is_db_available())
    if hasattr(cp, "delete_thread") and callable(cp.delete_thread):
        cp.delete_thread(thread_id.strip())


# ---------------------------------------------------------------------------
# 3. Database Initialization & Schema Migrations
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Initialize database tables, indexes, and LangGraph checkpointer migrations."""
    if not is_db_available():
        raise ConnectionError(
            f"PostgreSQL is not reachable at {POSTGRES_HOST}:{POSTGRES_PORT}. "
            "Please start the database container using: docker compose up -d"
        )

    schema_sql = """
    CREATE TABLE IF NOT EXISTS sessions (
        id VARCHAR(64) PRIMARY KEY,
        title VARCHAR(255) DEFAULT 'General Chat',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        metadata JSONB NULL
    );

    CREATE TABLE IF NOT EXISTS messages (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(64) NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role VARCHAR(32) NOT NULL,
        content TEXT NOT NULL,
        tool_calls JSONB NULL,
        tool_call_id VARCHAR(128) NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_messages_session_time ON messages(session_id, created_at);

    CREATE TABLE IF NOT EXISTS user_facts (
        id SERIAL PRIMARY KEY,
        key VARCHAR(128) UNIQUE NOT NULL,
        value TEXT NOT NULL,
        category VARCHAR(64) DEFAULT 'general',
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """
    with get_pool().connection(timeout=2.0) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()

    # Run LangGraph checkpointer table migrations
    saver = get_checkpointer(use_memory_for_testing=False)
    if hasattr(saver, "setup") and callable(saver.setup):
        saver.setup()


# ---------------------------------------------------------------------------
# 4. Long-Term Memory: User Profile & Facts Store
# ---------------------------------------------------------------------------


def upsert_fact(key: str, value: str, category: str = "general") -> None:
    """Insert or update a long-term user fact in the database."""
    sql = """
    INSERT INTO user_facts (key, value, category, updated_at)
    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
    ON CONFLICT (key) DO UPDATE
    SET value = EXCLUDED.value,
        category = EXCLUDED.category,
        updated_at = CURRENT_TIMESTAMP;
    """
    with get_pool().connection(timeout=2.0) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (key.strip().lower(), value.strip(), category.strip().lower()))
        conn.commit()


def get_all_facts(category: str | None = None) -> list[dict[str, Any]]:
    """Retrieve all long-term user facts, optionally filtered by category."""
    if category:
        sql = "SELECT key, value, category FROM user_facts WHERE category = %s ORDER BY key ASC;"
        params = (category.strip().lower(),)
    else:
        sql = "SELECT key, value, category FROM user_facts ORDER BY category ASC, key ASC;"
        params = ()

    with get_pool().connection(timeout=2.0) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cast(list[dict[str, Any]], cur.fetchall())


def delete_fact(key: str) -> bool:
    """Delete a user fact by key. Returns True if a record was removed."""
    sql = "DELETE FROM user_facts WHERE key = %s;"
    with get_pool().connection(timeout=2.0) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (key.strip().lower(),))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted


# ---------------------------------------------------------------------------
# 5. Backward Compatibility Wrappers (Sessions & Messages)
# ---------------------------------------------------------------------------


def ensure_session(session_id: str, title: str = "General Chat") -> None:
    """Ensure a session exists in the database."""
    sql = """
    INSERT INTO sessions (id, title)
    VALUES (%s, %s)
    ON CONFLICT (id) DO NOTHING;
    """
    with get_pool().connection(timeout=2.0) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id, title))
        conn.commit()


def touch_session(session_id: str) -> None:
    """Update the updated_at timestamp of a session."""
    sql = "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = %s;"
    with get_pool().connection(timeout=2.0) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id,))
        conn.commit()


def save_message(
    session_id: str,
    role: str,
    content: str,
    tool_calls: Any | None = None,
    tool_call_id: str | None = None,
) -> None:
    """Save a conversation message and touch session updated_at."""
    ensure_session(session_id)
    sql = """
    INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id)
    VALUES (%s, %s, %s, %s, %s);
    """
    tool_calls_json = json.dumps(tool_calls) if tool_calls else None
    with get_pool().connection(timeout=2.0) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (session_id, role, content, tool_calls_json, tool_call_id))
            cur.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = %s;", (session_id,))
        conn.commit()


def get_recent_messages(session_id: str, limit: int = 12) -> list[dict[str, Any]]:
    """Retrieve the most recent N messages for a session in chronological order."""
    sql = """
    SELECT role, content, tool_calls, tool_call_id
    FROM (
        SELECT id, role, content, tool_calls, tool_call_id
        FROM messages
        WHERE session_id = %s
        ORDER BY id DESC
        LIMIT %s
    ) sub
    ORDER BY id ASC;
    """
    with get_pool().connection(timeout=2.0) as conn, conn.cursor() as cur:
        cur.execute(sql, (session_id, limit))
        return cast(list[dict[str, Any]], cur.fetchall())
