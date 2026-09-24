"""Reading and writing meetings.

Design rules, in priority order:

1. **Deterministic ids.** An item's id is derived from its meeting, kind,
   timestamp and normalised text. Re-extracting a meeting updates the same
   rows rather than duplicating them, so re-runs are safe and a prompt
   change can be diffed instead of guessed at.
2. **Human edits win.** A row with `edited = 1` is never overwritten by
   re-extraction. A tool that silently reverts someone's correction does
   not get corrected a second time.
3. **Nothing is deleted silently.** Corrections are appended to their own
   table; items that disappear from a re-run are left in place unless the
   caller explicitly replaces the meeting.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone

from mat.extract import MeetingExtract
from mat.extract.merge import SIMILARITY_THRESHOLD, normalize, similarity
from mat.ingest import Transcript

from .db import transaction

OPEN_STATUSES = ("open", "in_progress")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_item_id(meeting_id: str, kind: str, timestamp: str, text: str) -> str:
    """A stable id for an item, independent of extraction run.

    Normalised text is used rather than raw text so that a reworded but
    identical item keeps its id - and therefore keeps its human edits and
    its status - across re-extractions.
    """
    fingerprint = " ".join(sorted(normalize(text)))
    digest = hashlib.sha1(
        f"{meeting_id}|{kind}|{timestamp}|{fingerprint}".encode()
    ).hexdigest()
    return f"{kind[:3]}_{digest[:16]}"


@dataclass(frozen=True, slots=True)
class SaveReport:
    meeting_id: str
    inserted: int = 0
    updated: int = 0
    preserved: int = 0  # rows left alone because a human had edited them
    carried_links: int = 0

    def summary_line(self) -> str:
        return (
            f"{self.meeting_id}: {self.inserted} new, {self.updated} updated, "
            f"{self.preserved} preserved (edited by hand), "
            f"{self.carried_links} linked to earlier meetings"
        )


# --- writing ------------------------------------------------------------


def save_meeting(
    connection: sqlite3.Connection,
    extract: MeetingExtract,
    transcript: Transcript | None = None,
    model: str = "",
) -> SaveReport:
    """Persist one meeting and everything extracted from it."""
    counts = {"inserted": 0, "updated": 0, "preserved": 0}

    with transaction(connection):
        connection.execute(
            """
            INSERT INTO meetings
                (meeting_id, title, meeting_date, summary, source,
                 duration_secs, extracted_at, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(meeting_id) DO UPDATE SET
                title = excluded.title,
                meeting_date = excluded.meeting_date,
                summary = excluded.summary,
                extracted_at = excluded.extracted_at,
                model = excluded.model
            """,
            (
                extract.meeting_id,
                extract.title,
                extract.meeting_date.isoformat(),
                extract.summary,
                transcript.source if transcript else "text",
                transcript.duration if transcript else 0.0,
                _now(),
                model,
            ),
        )

        connection.executemany(
            "INSERT OR IGNORE INTO participants (meeting_id, label) VALUES (?, ?)",
            [(extract.meeting_id, label) for label in extract.participants],
        )

        if transcript is not None:
            connection.executemany(
                """
                INSERT INTO transcript_segments
                    (meeting_id, timestamp, start_secs, end_secs, speaker, text)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(meeting_id, timestamp) DO UPDATE SET
                    speaker = excluded.speaker, text = excluded.text
                """,
                [
                    (
                        extract.meeting_id,
                        segment.timestamp,
                        segment.start,
                        segment.end,
                        segment.speaker,
                        segment.text,
                    )
                    for segment in transcript
                ],
            )

        for item in extract.action_items:
            _upsert(
                connection,
                counts,
                table="action_items",
                item_id=make_item_id(extract.meeting_id, "action", item.timestamp, item.task),
                meeting_id=extract.meeting_id,
                columns={
                    "task": item.task,
                    "owner": item.owner,
                    "owner_source": item.owner_source.value,
                    "due_phrase": item.due_phrase,
                    "due_date": item.due_date.isoformat() if item.due_date else None,
                    "due_date_source": item.due_date_source.value,
                    "status": item.status.value,
                    "timestamp": item.timestamp,
                    "confidence": item.confidence,
                },
                # Status is a human's to own once set; re-extraction always
                # reports 'open' and would otherwise reopen finished work.
                protected=("status",),
            )

        for item in extract.decisions:
            _upsert(
                connection,
                counts,
                table="decisions",
                item_id=make_item_id(
                    extract.meeting_id, "decision", item.timestamp, item.decision
                ),
                meeting_id=extract.meeting_id,
                columns={
                    "decision": item.decision,
                    "made_by": item.made_by,
                    "agreed_by": ",".join(item.agreed_by),
                    "timestamp": item.timestamp,
                    "confidence": item.confidence,
                },
            )

        for item in extract.open_questions:
            _upsert(
                connection,
                counts,
                table="open_questions",
                item_id=make_item_id(
                    extract.meeting_id, "question", item.timestamp, item.question
                ),
                meeting_id=extract.meeting_id,
                columns={
                    "question": item.question,
                    "raised_by": item.raised_by,
                    "status": item.status.value,
                    "timestamp": item.timestamp,
                    "confidence": item.confidence,
                },
                protected=("status",),
            )

        for item in extract.risks:
            _upsert(
                connection,
                counts,
                table="risks",
                item_id=make_item_id(extract.meeting_id, "risk", item.timestamp, item.risk),
                meeting_id=extract.meeting_id,
                columns={
                    "risk": item.risk,
                    "severity": item.severity.value,
                    "timestamp": item.timestamp,
                    "confidence": item.confidence,
                },
            )

    carried = link_carried_items(connection, extract.meeting_id)
    return SaveReport(meeting_id=extract.meeting_id, carried_links=carried, **counts)


def _upsert(
    connection: sqlite3.Connection,
    counts: dict[str, int],
    table: str,
    item_id: str,
    meeting_id: str,
    columns: dict[str, object],
    protected: tuple[str, ...] = (),
) -> None:
    """Insert, or update only the fields a human has not claimed."""
    existing = connection.execute(
        f"SELECT edited FROM {table} WHERE item_id = ?", (item_id,)  # noqa: S608 - fixed names
    ).fetchone()

    now = _now()

    if existing is None:
        names = ["item_id", "meeting_id", *columns, "created_at", "updated_at"]
        placeholders = ", ".join("?" for _ in names)
        connection.execute(
            f"INSERT INTO {table} ({', '.join(names)}) VALUES ({placeholders})",  # noqa: S608
            [item_id, meeting_id, *columns.values(), now, now],
        )
        counts["inserted"] += 1
        return

    if existing["edited"]:
        counts["preserved"] += 1
        return

    updatable = {k: v for k, v in columns.items() if k not in protected}
    assignments = ", ".join(f"{name} = ?" for name in updatable)
    connection.execute(
        f"UPDATE {table} SET {assignments}, updated_at = ? WHERE item_id = ?",  # noqa: S608
        [*updatable.values(), now, item_id],
    )
    counts["updated"] += 1


# --- carry-forward ------------------------------------------------------


def link_carried_items(connection: sqlite3.Connection, meeting_id: str) -> int:
    """Link this meeting's items to matching open items from earlier ones.

    The same commitment restated week after week ("still waiting on those
    keys") should read as one ageing item, not five unrelated ones. The
    link is recorded rather than the rows merged, so each meeting's record
    of what was actually said stays intact.
    """
    meeting = get_meeting(connection, meeting_id)
    if meeting is None:
        return 0

    earlier = connection.execute(
        """
        SELECT a.item_id, a.task, a.owner
        FROM action_items a
        JOIN meetings m ON m.meeting_id = a.meeting_id
        WHERE m.meeting_date < ?
          AND a.status IN (?, ?)
          AND a.deleted = 0
        ORDER BY m.meeting_date DESC
        """,
        (meeting["meeting_date"], *OPEN_STATUSES),
    ).fetchall()

    if not earlier:
        return 0

    current = connection.execute(
        "SELECT item_id, task, owner FROM action_items "
        "WHERE meeting_id = ? AND carried_from IS NULL AND deleted = 0",
        (meeting_id,),
    ).fetchall()

    links = 0
    with transaction(connection):
        for item in current:
            for candidate in earlier:
                if candidate["owner"].casefold() != item["owner"].casefold():
                    continue
                if similarity(candidate["task"], item["task"]) < SIMILARITY_THRESHOLD:
                    continue
                connection.execute(
                    "UPDATE action_items SET carried_from = ? WHERE item_id = ?",
                    (candidate["item_id"], item["item_id"]),
                )
                links += 1
                break

    return links


def open_items_before(connection: sqlite3.Connection, before: date) -> list[sqlite3.Row]:
    """Unresolved commitments from meetings earlier than `before`.

    This is what makes the tool a tracker rather than a note taker: last
    week's promise reappears at the top of this week's board.
    """
    return connection.execute(
        """
        SELECT a.*, m.meeting_date, m.title
        FROM action_items a
        JOIN meetings m ON m.meeting_id = a.meeting_id
        WHERE m.meeting_date < ?
          AND a.status IN (?, ?)
          AND a.deleted = 0
        ORDER BY a.due_date IS NULL, a.due_date, m.meeting_date
        """,
        (before.isoformat(), *OPEN_STATUSES),
    ).fetchall()


def overdue_items(connection: sqlite3.Connection, as_of: date) -> list[sqlite3.Row]:
    """Open items whose due date has passed."""
    return connection.execute(
        """
        SELECT a.*, m.meeting_date, m.title
        FROM action_items a
        JOIN meetings m ON m.meeting_id = a.meeting_id
        WHERE a.status IN (?, ?)
          AND a.deleted = 0
          AND a.due_date IS NOT NULL
          AND a.due_date < ?
        ORDER BY a.due_date
        """,
        (*OPEN_STATUSES, as_of.isoformat()),
    ).fetchall()


# --- reading ------------------------------------------------------------


def get_meeting(connection: sqlite3.Connection, meeting_id: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM meetings WHERE meeting_id = ?", (meeting_id,)
    ).fetchone()


def list_meetings(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT m.*,
               (SELECT COUNT(*) FROM action_items a
                 WHERE a.meeting_id = m.meeting_id AND a.deleted = 0)
                   AS action_count,
               (SELECT COUNT(*) FROM action_items a
                 WHERE a.meeting_id = m.meeting_id AND a.deleted = 0
                   AND a.status IN ('open', 'in_progress'))
                   AS open_count
        FROM meetings m
        ORDER BY m.meeting_date DESC
        """
    ).fetchall()


def action_items(
    connection: sqlite3.Connection,
    meeting_id: str | None = None,
    owner: str | None = None,
    status: str | None = None,
) -> list[sqlite3.Row]:
    clauses, params = ["a.deleted = 0"], []
    if meeting_id:
        clauses.append("a.meeting_id = ?")
        params.append(meeting_id)
    if owner:
        clauses.append("a.owner = ?")
        params.append(owner)
    if status:
        clauses.append("a.status = ?")
        params.append(status)

    where = f"WHERE {' AND '.join(clauses)}"
    return connection.execute(
        f"""
        SELECT a.*, m.meeting_date, m.title
        FROM action_items a
        JOIN meetings m ON m.meeting_id = a.meeting_id
        {where}
        ORDER BY m.meeting_date DESC, a.timestamp
        """,  # noqa: S608 - `where` is built from a fixed set of clauses
        params,
    ).fetchall()


def meeting_items(connection: sqlite3.Connection, meeting_id: str) -> dict[str, list]:
    """Everything belonging to one meeting, ready for the UI."""
    return {
        table: connection.execute(
            f"SELECT * FROM {table} WHERE meeting_id = ? AND deleted = 0 "  # noqa: S608
            "ORDER BY timestamp",
            (meeting_id,),
        ).fetchall()
        for table in ("action_items", "decisions", "open_questions", "risks")
    }


def transcript_segments(connection: sqlite3.Connection, meeting_id: str) -> list[sqlite3.Row]:
    return connection.execute(
        "SELECT * FROM transcript_segments WHERE meeting_id = ? ORDER BY start_secs",
        (meeting_id,),
    ).fetchall()


def owner_workload(connection: sqlite3.Connection) -> list[dict]:
    """Commitments made versus closed, per person.

    Grouped on first name rather than the raw label. Extraction writes
    "Meera" from one line and "Meera (Lead)" from another, and grouping on
    the exact string split one person into two rows in the chart - the
    "owner matching is by exact string" limitation, showing up where a
    reader would actually be misled by it. The longest label is kept for
    display, since it carries the role.
    """
    rows = connection.execute(
        """
        SELECT owner,
               COUNT(*) AS total,
               SUM(status = 'done') AS done,
               SUM(status IN ('open', 'in_progress')) AS outstanding
        FROM action_items
        WHERE deleted = 0
        GROUP BY owner
        """
    ).fetchall()

    merged: dict[str, dict] = {}
    for row in rows:
        key = _owner_key(row["owner"])
        entry = merged.setdefault(
            key, {"owner": row["owner"], "total": 0, "done": 0, "outstanding": 0}
        )
        if len(row["owner"]) > len(entry["owner"]):
            entry["owner"] = row["owner"]
        entry["total"] += row["total"]
        entry["done"] += row["done"] or 0
        entry["outstanding"] += row["outstanding"] or 0

    return sorted(
        merged.values(), key=lambda e: (e["outstanding"], e["total"]), reverse=True
    )


def _owner_key(label: str) -> str:
    """"Meera" and "Meera (Lead)" are the same person."""
    bare = label.split("(")[0].strip().casefold()
    return bare.split()[0] if bare else label.casefold()


# --- editing ------------------------------------------------------------


def update_item(
    connection: sqlite3.Connection,
    table: str,
    item_id: str,
    field: str,
    value: object,
) -> bool:
    """Apply a human edit, log it, and mark the row as theirs from now on."""
    if table not in ITEM_TABLES:
        raise ValueError(f"unknown table {table!r}")

    row = connection.execute(
        f"SELECT * FROM {table} WHERE item_id = ?", (item_id,)  # noqa: S608 - checked above
    ).fetchone()
    if row is None:
        return False
    if field not in row.keys():
        raise ValueError(f"unknown field {field!r} on {table}")

    with transaction(connection):
        connection.execute(
            f"UPDATE {table} SET {field} = ?, edited = 1, updated_at = ? "  # noqa: S608
            "WHERE item_id = ?",
            (value, _now(), item_id),
        )
        _log(connection, table, item_id, "edit", field, row[field], value)

    return True


ITEM_TABLES = {
    "action_items": ("action", "task"),
    "decisions": ("decision", "decision"),
    "open_questions": ("question", "question"),
    "risks": ("risk", "risk"),
}


def add_item(
    connection: sqlite3.Connection,
    table: str,
    meeting_id: str,
    columns: dict[str, object],
) -> str:
    """Record an item a person added because the model missed it.

    Marked `origin = 'human'` and `edited = 1`, so re-extraction never
    touches it. These rows are the measured misses: together with deleted
    rows (the false positives) they give a recall and precision signal from
    real use, independent of the hand-labelled corpus.
    """
    if table not in ITEM_TABLES:
        raise ValueError(f"unknown table {table!r}")

    kind, text_column = ITEM_TABLES[table]
    text = str(columns.get(text_column, "")).strip()
    if not text:
        raise ValueError(f"{text_column} is required")

    timestamp = str(columns.get("timestamp", "00:00:00"))
    item_id = make_item_id(meeting_id, f"{kind}-human", timestamp, text)
    now = _now()

    payload = {
        **columns,
        "timestamp": timestamp,
        # A person's own entry is certain by definition; the confidence
        # field describes the model's belief, not theirs.
        "confidence": 1.0,
        "edited": 1,
        "origin": "human",
    }

    names = ["item_id", "meeting_id", *payload, "created_at", "updated_at"]
    placeholders = ", ".join("?" for _ in names)

    with transaction(connection):
        connection.execute(
            f"INSERT OR REPLACE INTO {table} ({', '.join(names)}) "  # noqa: S608
            f"VALUES ({placeholders})",
            [item_id, meeting_id, *payload.values(), now, now],
        )
        _log(connection, table, item_id, "add", text_column, None, text)

    return item_id


def delete_item(connection: sqlite3.Connection, table: str, item_id: str) -> bool:
    """Hide a wrongly extracted item without losing the evidence.

    A rejected item is a measured false positive - the most useful feedback
    the system can get - so it is flagged, never removed.
    """
    return _set_deleted(connection, table, item_id, deleted=True)


def restore_item(connection: sqlite3.Connection, table: str, item_id: str) -> bool:
    return _set_deleted(connection, table, item_id, deleted=False)


def _set_deleted(
    connection: sqlite3.Connection, table: str, item_id: str, deleted: bool
) -> bool:
    if table not in ITEM_TABLES:
        raise ValueError(f"unknown table {table!r}")

    _, text_column = ITEM_TABLES[table]
    row = connection.execute(
        f"SELECT * FROM {table} WHERE item_id = ?", (item_id,)  # noqa: S608 - checked above
    ).fetchone()
    if row is None:
        return False

    with transaction(connection):
        connection.execute(
            f"UPDATE {table} SET deleted = ?, updated_at = ? WHERE item_id = ?",  # noqa: S608
            (1 if deleted else 0, _now(), item_id),
        )
        _log(
            connection,
            table,
            item_id,
            "delete" if deleted else "restore",
            text_column,
            row[text_column],
            None,
        )
    return True


def deleted_items(connection: sqlite3.Connection, table: str, meeting_id: str) -> list:
    if table not in ITEM_TABLES:
        raise ValueError(f"unknown table {table!r}")
    return connection.execute(
        f"SELECT * FROM {table} WHERE meeting_id = ? AND deleted = 1 "  # noqa: S608
        "ORDER BY timestamp",
        (meeting_id,),
    ).fetchall()


def _log(
    connection: sqlite3.Connection,
    table: str,
    item_id: str,
    action: str,
    field: str,
    old_value: object,
    new_value: object,
) -> None:
    connection.execute(
        """
        INSERT INTO corrections
            (item_kind, item_id, action, field, old_value, new_value, corrected_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            table,
            item_id,
            action,
            field,
            None if old_value is None else str(old_value),
            None if new_value is None else str(new_value),
            _now(),
        ),
    )


def correction_stats(connection: sqlite3.Connection) -> dict[str, int]:
    """Counts by action, plus what they imply about the model.

    `deletes` are false positives a person found; `adds` are misses. Both
    are lower bounds - they only count what someone bothered to correct -
    so they are reported as observed counts, never as a precision figure
    dressed up as measured accuracy. The hand-labelled corpus in Phase 7
    remains the real measurement.
    """
    rows = connection.execute(
        "SELECT action, COUNT(*) AS count FROM corrections GROUP BY action"
    ).fetchall()
    counts = {row["action"]: row["count"] for row in rows}

    model_items = sum(
        connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE origin = 'model'"  # noqa: S608
        ).fetchone()[0]
        for table in ITEM_TABLES
    )

    return {
        "edits": counts.get("edit", 0),
        "adds": counts.get("add", 0),
        "deletes": counts.get("delete", 0),
        "restores": counts.get("restore", 0),
        "model_items": model_items,
    }


def corrections(connection: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    return connection.execute(
        "SELECT * FROM corrections ORDER BY correction_id DESC LIMIT ?", (limit,)
    ).fetchall()


def delete_meeting(connection: sqlite3.Connection, meeting_id: str) -> bool:
    """Remove a meeting and everything cascading from it."""
    with transaction(connection):
        cursor = connection.execute(
            "DELETE FROM meetings WHERE meeting_id = ?", (meeting_id,)
        )
    return cursor.rowcount > 0
