import json
import socket
from typing import Any, cast

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.config import (
    DATABASE_URL,
    POSTGRES_HOST,
    POSTGRES_PORT,
)

_pool: ConnectionPool | None = None


def is_db_available() -> bool:
    """Instant socket check to see if PostgreSQL port is open."""
    try:
        with socket.create_connection((POSTGRES_HOST, POSTGRES_PORT), timeout=0.5):
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


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


def init_db() -> None:
    """Initialize database tables and indexes."""
    if not is_db_available():
        raise ConnectionError(f"PostgreSQL is not reachable at {POSTGRES_HOST}:{POSTGRES_PORT}")

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


# -------------------------------------------------------------
# Session Management
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# Message History Management (Sliding Window)
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# User Profile & Facts Management
# -------------------------------------------------------------
def upsert_fact(key: str, value: str, category: str = "general") -> None:
    """Insert or update a user fact."""
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
    """Retrieve all user facts, optionally filtered by category."""
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
    """Delete a user fact by key. Returns True if a row was deleted."""
    sql = "DELETE FROM user_facts WHERE key = %s;"
    with get_pool().connection(timeout=2.0) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (key.strip().lower(),))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted
