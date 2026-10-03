"""Dependency wiring: builds every long-lived object once."""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import Settings
from core.attendance import AttendanceManager
from core.face_detector import FaceDetector
from core.face_encoder import FaceEncoder
from core.face_matcher import FaceMatcher
from database.db import Database
from services.recognition_service import RecognitionService
from services.registration_service import RegistrationService


@dataclass
class AppContext:
    """All long-lived objects, built once and shared with the GUI."""
    settings: Settings
    db: Database
    detector: FaceDetector
    encoder: FaceEncoder
    matcher: FaceMatcher
    attendance: AttendanceManager
    registration: RegistrationService
    recognition: RecognitionService


def build_context(settings: Settings) -> AppContext:
    db = Database(settings.db_path)
    db.initialize()
    detector = FaceDetector(scale=settings.frame_scale)
    encoder = FaceEncoder()
    matcher = FaceMatcher(settings.recognition_threshold)
    attendance = AttendanceManager(db, settings.attendance_cooldown_seconds)
    return AppContext(
        settings, db, detector, encoder, matcher, attendance,
        RegistrationService(db, detector, encoder, settings),
        RecognitionService(db, detector, encoder, matcher, attendance),
    )
