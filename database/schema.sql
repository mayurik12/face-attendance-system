PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_code TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL CHECK (length(trim(name)) > 0),
    email         TEXT,
    department    TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- One row per captured sample. 128 float32 values stored as a 512-byte BLOB.
CREATE TABLE IF NOT EXISTS face_embeddings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL CHECK (length(embedding) > 0),
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS attendance (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    timestamp            TEXT NOT NULL,            -- local time, 'YYYY-MM-DD HH:MM:SS'
    status               TEXT NOT NULL DEFAULT 'PRESENT',
    recognition_distance REAL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_user  ON face_embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_attendance_user_ts ON attendance(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_attendance_ts    ON attendance(timestamp);
