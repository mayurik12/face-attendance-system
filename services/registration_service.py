"""Registration workflow (no GUI code here)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.settings import Settings
from core.face_detector import FaceDetector, assess_quality
from core.face_encoder import FaceEncoder
from database.db import Database
from database.models import User
from utils.errors import AppError, DuplicateUserError, RegistrationError
from utils.helpers import Box
from utils.logger import get_logger
from utils.validators import validate_department, validate_email, validate_name, validate_user_code

log = get_logger(__name__)


@dataclass(frozen=True)
class UserDetails:
    employee_code: str
    name: str
    email: str | None
    department: str | None


@dataclass(frozen=True)
class SampleResult:
    accepted: bool
    message: str
    box: Box | None = None


class RegistrationSession:
    """Collects embeddings from webcam frames; each frame is checked before use."""

    def __init__(self, detector: FaceDetector, encoder: FaceEncoder, settings: Settings) -> None:
        self._detector, self._encoder, self._settings = detector, encoder, settings
        self.embeddings: list[np.ndarray] = []

    @property
    def required(self) -> int:
        return self._settings.registration_samples

    @property
    def is_complete(self) -> bool:
        return len(self.embeddings) >= self.required

    def process_frame(self, frame: np.ndarray) -> SampleResult:
        """Try to turn one frame into a sample. Never raises for expected problems."""
        try:
            box = self._detector.detect_single(frame)  # exactly one face
            quality = assess_quality(frame, box, self._settings.min_face_size, self._settings.min_sharpness)
            if not quality.ok:
                return SampleResult(False, quality.reason, box)
            self.embeddings.append(self._encoder.encode(frame, box))
        except AppError as exc:
            return SampleResult(False, str(exc))
        return SampleResult(True, f"Sample {len(self.embeddings)}/{self.required} captured", box)


class RegistrationService:
    def __init__(self, db: Database, detector: FaceDetector, encoder: FaceEncoder, settings: Settings) -> None:
        self._db, self._detector, self._encoder, self._settings = db, detector, encoder, settings

    def validate_details(self, employee_code: str, name: str, email: str = "", department: str = "") -> UserDetails:
        """Validate input and make sure the ID is not already registered."""
        details = UserDetails(
            validate_user_code(employee_code), validate_name(name),
            validate_email(email), validate_department(department),
        )
        if self._db.get_user_by_code(details.employee_code):
            raise DuplicateUserError(f"ID '{details.employee_code}' is already registered.")
        return details

    def new_session(self) -> RegistrationSession:
        return RegistrationSession(self._detector, self._encoder, self._settings)

    def complete(self, details: UserDetails, session: RegistrationSession) -> User:
        """Persist the user only if enough good samples were collected."""
        if len(session.embeddings) < session.required:
            raise RegistrationError(
                f"Only {len(session.embeddings)} of {session.required} samples captured. Registration cancelled."
            )
        return self._db.add_user_with_embeddings(
            details.employee_code, details.name, details.email, details.department, session.embeddings
        )
