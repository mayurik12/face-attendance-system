"""Real-time recognition: frame -> detect -> encode -> match -> attendance."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np

from config.settings import Settings
from core.attendance import AttendanceManager
from core.camera import Camera
from core.face_detector import FaceDetector
from core.face_encoder import FaceEncoder
from core.face_matcher import FaceMatcher
from database.db import Database
from database.models import User
from utils.errors import AppError, CameraError, InvalidImageError
from utils.helpers import Box, draw_face_box
from utils.logger import get_logger

log = get_logger(__name__)
GREEN, RED = (60, 160, 60), (50, 50, 200)


@dataclass(frozen=True)
class FaceResult:
    box: Box
    user: User | None
    distance: float | None
    confidence: float
    attendance_marked: bool

    @property
    def is_known(self) -> bool:
        return self.user is not None

    @property
    def label(self) -> str:
        if not self.user:
            return "Unknown"
        return f"{self.user.name} ({self.user.employee_code}) d={self.distance:.2f} conf={self.confidence:.0%}"


class RecognitionService:
    def __init__(self, db: Database, detector: FaceDetector, encoder: FaceEncoder,
                 matcher: FaceMatcher, attendance: AttendanceManager) -> None:
        self._db, self._detector, self._encoder = db, detector, encoder
        self._matcher, self._attendance = matcher, attendance
        self._users: dict[int, User] = {}

    def reload_embeddings(self) -> int:
        """Load users + embeddings from SQLite into memory (once, not per frame)."""
        ids, matrix = self._db.load_all_embeddings()
        self._matcher.load(ids, matrix)
        self._users = {u.id: u for u in self._db.list_users()}
        log.info("Recognition cache loaded: %d samples, %d users", len(ids), len(self._users))
        return len(ids)

    def process_frame(self, frame: np.ndarray) -> list[FaceResult]:
        results: list[FaceResult] = []
        for box in self._detector.detect(frame):
            try:
                embedding = self._encoder.encode(frame, box)
            except InvalidImageError as exc:
                log.debug("Skipping face: %s", exc)
                continue
            match = self._matcher.match(embedding)
            user = self._users.get(match.user_id) if match.is_known and match.user_id is not None else None
            marked = False
            if user is not None:
                marked = self._attendance.mark(user.id, match.distance) is not None
            results.append(FaceResult(box, user, match.distance, match.confidence, marked))
        return results


class RecognitionRunner:
    """Capture + recognise on a background thread; the GUI only polls the latest frame."""

    def __init__(self, camera: Camera, service: RecognitionService, settings: Settings) -> None:
        self._camera, self._service, self._settings = camera, service, settings
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._status = "Camera Connected"
        self.events: "queue.Queue[str]" = queue.Queue()
        self.error: str | None = None

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def start(self) -> None:
        self._camera.open()  # raises CameraError for the GUI to display
        self._thread = threading.Thread(target=self._loop, name="recognition", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._camera.release()

    def _loop(self) -> None:
        index, results = 0, []
        try:
            while not self._stop.is_set():
                frame = self._camera.read()
                index += 1
                if index % self._settings.process_every_n_frames == 0:
                    try:
                        results = self._service.process_frame(frame)
                    except AppError as exc:
                        log.warning("Frame skipped: %s", exc)
                        results = []
                    self._update_status(results)
                view = frame.copy()
                for r in results:
                    draw_face_box(view, r.box, r.label, GREEN if r.is_known else RED)
                with self._lock:
                    self._latest = view
        except CameraError as exc:
            self.error = str(exc)
            self._status = "Camera Not Available"
        except Exception:  # last line of defence: log, report, never crash the GUI
            log.exception("Unexpected error in recognition loop")
            self.error = "Unexpected error in recognition. See logs/app.log."
            self._status = "Recognition stopped"

    def _update_status(self, results: list[FaceResult]) -> None:
        if not results:
            self._status = "Recognizing..."
            return
        for r in results:
            if r.attendance_marked and r.user:
                self._status = f"Attendance Marked: {r.user.name}"
                self.events.put(f"{r.user.name} ({r.user.employee_code}) - attendance marked")
                return
        known = [r for r in results if r.user]
        self._status = f"Face Detected: {known[0].user.name}" if known else "Unknown Face"
