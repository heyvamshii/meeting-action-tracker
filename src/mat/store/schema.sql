-- Meeting Action Tracker schema.
--
-- Two ideas shape this design:
--
-- 1. Item ids are DETERMINISTIC, derived from (meeting, kind, timestamp,
--    normalised text). Re-extracting a meeting therefore updates the same
--    rows instead of duplicating them, which makes re-runs safe and makes
--    "did the new prompt change anything?" answerable with a diff.
--
-- 2. Human edits win. Every item carries `edited`; once a person has
--    touched a row, re-extraction leaves their version alone. A tool that
--    silently reverts a correction will not be trusted twice.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meetings (
    meeting_id    TEXT PRIMARY KEY,
    title         TEXT NOT NULL DEFAULT '',
    meeting_date  TEXT NOT NULL,              -- ISO date
    summary       TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL DEFAULT 'text',
    duration_secs REAL NOT NULL DEFAULT 0,
    extracted_at  TEXT NOT NULL,
    model         TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS participants (
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    label      TEXT NOT NULL,
    PRIMARY KEY (meeting_id, label)
);

-- The transcript is stored so the UI can jump from an item to its source
-- line, and so a citation can still be verified after the .txt is gone.
CREATE TABLE IF NOT EXISTS transcript_segments (
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    timestamp  TEXT NOT NULL,
    start_secs REAL NOT NULL,
    end_secs   REAL NOT NULL,
    speaker    TEXT NOT NULL,
    text       TEXT NOT NULL,
    PRIMARY KEY (meeting_id, timestamp)
);

CREATE TABLE IF NOT EXISTS action_items (
    item_id         TEXT PRIMARY KEY,
    meeting_id      TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    task            TEXT NOT NULL,
    owner           TEXT NOT NULL DEFAULT 'UNASSIGNED',
    owner_source    TEXT NOT NULL DEFAULT 'none',
    due_phrase      TEXT,
    due_date        TEXT,                     -- ISO date, nullable
    due_date_source TEXT NOT NULL DEFAULT 'none',
    status          TEXT NOT NULL DEFAULT 'open',
    timestamp       TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.5,
    edited          INTEGER NOT NULL DEFAULT 0,
    -- Soft delete. A rejected item is the single most valuable correction
    -- signal there is - it is a measured false positive - so it is hidden,
    -- never removed.
    deleted         INTEGER NOT NULL DEFAULT 0,
    -- 'model' or 'human'. Items a person added are the misses.
    origin          TEXT NOT NULL DEFAULT 'model',
    -- Set when this item restates a commitment from an earlier meeting.
    carried_from    TEXT REFERENCES action_items(item_id) ON DELETE SET NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    item_id    TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    decision   TEXT NOT NULL,
    made_by    TEXT NOT NULL DEFAULT 'UNASSIGNED',
    agreed_by  TEXT NOT NULL DEFAULT '',      -- comma separated
    timestamp  TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    edited     INTEGER NOT NULL DEFAULT 0,
    deleted    INTEGER NOT NULL DEFAULT 0,
    origin     TEXT NOT NULL DEFAULT 'model',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS open_questions (
    item_id    TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    question   TEXT NOT NULL,
    raised_by  TEXT NOT NULL DEFAULT 'UNASSIGNED',
    status     TEXT NOT NULL DEFAULT 'open',
    timestamp  TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    edited     INTEGER NOT NULL DEFAULT 0,
    deleted    INTEGER NOT NULL DEFAULT 0,
    origin     TEXT NOT NULL DEFAULT 'model',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS risks (
    item_id    TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    risk       TEXT NOT NULL,
    severity   TEXT NOT NULL DEFAULT 'low',
    timestamp  TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    edited     INTEGER NOT NULL DEFAULT 0,
    deleted    INTEGER NOT NULL DEFAULT 0,
    origin     TEXT NOT NULL DEFAULT 'model',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Every human correction, kept. These become few-shot examples in Phase 6
-- and a second accuracy signal alongside the hand labels in Phase 7.
CREATE TABLE IF NOT EXISTS corrections (
    correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_kind     TEXT NOT NULL,
    item_id       TEXT NOT NULL,
    -- edit | add | delete | restore
    action        TEXT NOT NULL DEFAULT 'edit',
    field         TEXT NOT NULL,
    old_value     TEXT,
    new_value     TEXT,
    corrected_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_actions_meeting ON action_items(meeting_id);
CREATE INDEX IF NOT EXISTS idx_actions_status  ON action_items(status);
CREATE INDEX IF NOT EXISTS idx_actions_owner   ON action_items(owner);
CREATE INDEX IF NOT EXISTS idx_actions_due     ON action_items(due_date);
CREATE INDEX IF NOT EXISTS idx_decisions_meeting ON decisions(meeting_id);
CREATE INDEX IF NOT EXISTS idx_questions_meeting ON open_questions(meeting_id);
CREATE INDEX IF NOT EXISTS idx_risks_meeting     ON risks(meeting_id);
CREATE INDEX IF NOT EXISTS idx_corrections_item  ON corrections(item_kind, item_id);
