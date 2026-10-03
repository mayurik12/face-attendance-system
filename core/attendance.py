"""Attendance rules: mark once per cooldown window per person."""
from __future__ import annotations

import threading
from datetime import datetime
from typing import Callable

from database.db import Database
from utils.logger import get_logger

log = get_logger(__name__)


class AttendanceManager:
    def __init__(self, db: Database, cooldown_seconds: int, clock: Callable[[], datetime] = datetime.now) -> None:
        self._db = db
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._last_seen: dict[int, datetime] = {}  # cache: avoids a DB query on every frame
        self._lock = threading.Lock()

    def mark(self, user_id: int, distance: float | None = None) -> int | None:
        """Insert an attendance row unless the person was already marked within the cooldown.

        Returns the new attendance id, or None if skipped by the cooldown rule.
        """
        with self._lock:
            now = self._clock()
            last = self._last_seen.get(user_id) or self._db.last_attendance_time(user_id)
            if last is not None and (now - last).total_seconds() < self._cooldown:
                self._last_seen[user_id] = last
                return None
            record_id = self._db.add_attendance(user_id, now, "PRESENT", distance)
            self._last_seen[user_id] = now
        log.info("Attendance marked: user_id=%s record=%s", user_id, record_id)
        return record_id
