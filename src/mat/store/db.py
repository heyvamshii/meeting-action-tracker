"""SQLite connection handling and schema creation."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from mat.config import settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(
    db_path: str | Path | None = None, check_same_thread: bool = True
) -> sqlite3.Connection:
    """Open a connection with sane defaults and the schema applied.

    `:memory:` is passed through untouched so tests get a real database
    without touching the filesystem.

    `check_same_thread=False` is needed by Streamlit, which caches the
    connection across reruns that may land on different script-runner
    threads. Safe here because the app serialises its own writes; it would
    not be safe for concurrent writers.
    """
    path = str(db_path or settings.db_path)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path, check_same_thread=check_same_thread)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    # WAL keeps a Streamlit rerun from blocking on a writer mid-render.
    if path != ":memory:":
        connection.execute("PRAGMA journal_mode = WAL")

    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    migrate(connection)
    connection.commit()
    return connection


# Columns added after the first databases were created. `CREATE TABLE IF NOT
# EXISTS` cannot add a column to a table that already exists, so they are
# applied here instead. Phase 4 listed "no migrations" as a limitation; this
# is the smallest thing that removes it without pulling in a migration tool.
ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("action_items", "deleted", "INTEGER NOT NULL DEFAULT 0"),
    ("action_items", "origin", "TEXT NOT NULL DEFAULT 'model'"),
    ("decisions", "deleted", "INTEGER NOT NULL DEFAULT 0"),
    ("decisions", "origin", "TEXT NOT NULL DEFAULT 'model'"),
    ("open_questions", "deleted", "INTEGER NOT NULL DEFAULT 0"),
    ("open_questions", "origin", "TEXT NOT NULL DEFAULT 'model'"),
    ("risks", "deleted", "INTEGER NOT NULL DEFAULT 0"),
    ("risks", "origin", "TEXT NOT NULL DEFAULT 'model'"),
    ("corrections", "action", "TEXT NOT NULL DEFAULT 'edit'"),
)


def migrate(connection: sqlite3.Connection) -> list[str]:
    """Add any columns missing from an older database. Returns what it did."""
    applied = []

    for table, column, definition in ADDED_COLUMNS:
        existing = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if not existing or column in existing:
            continue
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        applied.append(f"{table}.{column}")

    return applied


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Commit on success, roll back on any failure.

    A meeting is saved as one unit: a half-written meeting whose action
    items landed but whose decisions did not is worse than no meeting.
    """
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()
