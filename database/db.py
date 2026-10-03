"""SQLite access layer.

A short-lived connection is opened per operation, which makes the class safe to
use from both the GUI thread and the recognition worker thread.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from config.settings import EMBEDDING_DIM
from database.models import User
from utils.errors import DatabaseError, DuplicateUserError, ValidationError
from utils.logger import get_logger

log = get_logger(__name__)
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def _like(term: str) -> str:
    """Escape LIKE wildcards so user input is matched literally."""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class Database:
    def __init__(self, path: Path | str, schema_path: Path = SCHEMA_PATH) -> None:
        self.path = Path(path)
        self._schema_path = schema_path

    # ---- connection / setup -------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(self.path, timeout=10)
        except sqlite3.Error as exc:
            log.error("Cannot open database %s: %s", self.path, exc)
            raise DatabaseError("Could not open the database file.") from exc
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            log.error("Database error: %s", exc)
            raise DatabaseError("A database operation failed.") from exc
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create the database file and tables if they do not exist."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            schema = self._schema_path.read_text(encoding="utf-8")
        except OSError as exc:
            log.error("Cannot prepare database: %s", exc)
            raise DatabaseError("Could not read the database schema or create the data folder.") from exc
        with self._connect() as conn:
            conn.executescript(schema)
        log.info("Database initialised at %s", self.path)

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _user(row: sqlite3.Row) -> User:
        return User(row["id"], row["employee_code"], row["name"], row["email"], row["department"], row["created_at"])

    @staticmethod
    def _to_blob(embedding: Sequence[float] | np.ndarray) -> bytes:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (EMBEDDING_DIM,) or not np.all(np.isfinite(vector)):
            raise ValidationError(f"Embedding must be {EMBEDDING_DIM} finite numbers.")
        return vector.tobytes()

    # ---- users --------------------------------------------------------------
    def add_user_with_embeddings(
        self,
        employee_code: str,
        name: str,
        email: str | None,
        department: str | None,
        embeddings: Sequence[np.ndarray],
    ) -> User:
        """Insert user + all embeddings atomically (all-or-nothing)."""
        if not embeddings:
            raise ValidationError("At least one embedding is required.")
        blobs = [self._to_blob(e) for e in embeddings]
        with self._connect() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO users (employee_code, name, email, department) VALUES (?, ?, ?, ?)",
                    (employee_code, name, email, department),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateUserError(f"ID '{employee_code}' is already registered.") from exc
            user_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO face_embeddings (user_id, embedding) VALUES (?, ?)",
                [(user_id, blob) for blob in blobs],
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        log.info("User registered: id=%s samples=%d", user_id, len(blobs))
        return self._user(row)

    def get_user_by_code(self, employee_code: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE employee_code = ?", (employee_code,)).fetchone()
        return self._user(row) if row else None

    def list_users(self) -> list[User]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY name COLLATE NOCASE").fetchall()
        return [self._user(r) for r in rows]

    def count_embeddings(self, user_id: int) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM face_embeddings WHERE user_id = ?", (user_id,)).fetchone()[0]

    def delete_user(self, user_id: int) -> bool:
        """Full erasure: user, embeddings and attendance history (ON DELETE CASCADE)."""
        with self._connect() as conn:
            deleted = conn.execute("DELETE FROM users WHERE id = ?", (user_id,)).rowcount > 0
        if deleted:
            log.info("User deleted: id=%s", user_id)
        return deleted

    def delete_biometrics(self, user_id: int) -> int:
        """Remove only a user's face embeddings; the user and attendance history stay."""
        with self._connect() as conn:
            count = conn.execute("DELETE FROM face_embeddings WHERE user_id = ?", (user_id,)).rowcount
        log.info("Biometric data deleted: user_id=%s samples=%d", user_id, count)
        return count

    # ---- embeddings ---------------------------------------------------------
    def load_all_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (user_ids[N], embeddings[N, 128]) for the in-memory matcher cache."""
        with self._connect() as conn:
            rows = conn.execute("SELECT user_id, embedding FROM face_embeddings ORDER BY id").fetchall()
        if not rows:
            return np.empty(0, dtype=np.int64), np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        ids = np.array([r["user_id"] for r in rows], dtype=np.int64)
        matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        return ids, matrix

    # ---- attendance ---------------------------------------------------------
    def add_attendance(self, user_id: int, when: datetime, status: str = "PRESENT",
                       distance: float | None = None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO attendance (user_id, timestamp, status, recognition_distance) VALUES (?, ?, ?, ?)",
                (user_id, when.strftime(TS_FORMAT), status, distance),
            )
            return int(cur.lastrowid)

    def last_attendance_time(self, user_id: int) -> datetime | None:
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(timestamp) AS ts FROM attendance WHERE user_id = ?", (user_id,)).fetchone()
        return datetime.strptime(row["ts"], TS_FORMAT) if row and row["ts"] else None

    def query_attendance(
        self,
        name: str = "",
        employee_code: str = "",
        on_date: date | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[dict]:
        """Filtered attendance history, newest first."""
        sql = [
            "SELECT a.id, a.user_id, u.employee_code, u.name, date(a.timestamp) AS date,",
            "       time(a.timestamp) AS time, a.status, a.recognition_distance",
            "FROM attendance a JOIN users u ON u.id = a.user_id WHERE 1 = 1",
        ]
        params: list = []
        if name.strip():
            sql.append("AND u.name LIKE ? ESCAPE '\\'")
            params.append(_like(name.strip()))
        if employee_code.strip():
            sql.append("AND u.employee_code LIKE ? ESCAPE '\\'")
            params.append(_like(employee_code.strip()))
        if on_date:
            sql.append("AND date(a.timestamp) = ?")
            params.append(on_date.isoformat())
        if date_from:
            sql.append("AND date(a.timestamp) >= ?")
            params.append(date_from.isoformat())
        if date_to:
            sql.append("AND date(a.timestamp) <= ?")
            params.append(date_to.isoformat())
        sql.append("ORDER BY a.timestamp DESC, a.id DESC")
        with self._connect() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        return [dict(r) for r in rows]
