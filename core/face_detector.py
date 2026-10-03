"""Face detection (localisation) with HOG + linear SVM via dlib.

Pipeline for one frame:  BGR frame -> resize -> BGR->RGB -> detector -> boxes ->
scale boxes back to full-resolution coordinates.
Detecting on a downscaled copy is much faster (cost grows with pixel count).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from utils.errors import ModelUnavailableError, MultipleFacesError, NoFaceError
from utils.helpers import Box, bgr_to_rgb, ensure_valid_frame
from utils.logger import get_logger

log = get_logger(__name__)
DetectorBackend = Callable[[np.ndarray], list[Box]]  # RGB image -> boxes


def _default_backend() -> DetectorBackend:
    try:
        import face_recognition
    except ImportError as exc:
        raise ModelUnavailableError(
            "The 'face_recognition' library (dlib) is not installed. See README -> Installation."
        ) from exc
    return lambda rgb: face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model="hog")


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    reason: str
    face_size: int
    sharpness: float


class FaceDetector:
    def __init__(self, scale: float = 0.25, backend: DetectorBackend | None = None) -> None:
        self.scale = scale
        self._backend = backend

    def detect(self, frame: np.ndarray) -> list[Box]:
        """Return face boxes (top, right, bottom, left) in full-frame pixels."""
        ensure_valid_frame(frame)
        if self._backend is None:
            self._backend = _default_backend()
        height, width = frame.shape[:2]
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        boxes = self._backend(bgr_to_rgb(small))
        inv = 1.0 / self.scale
        scaled: list[Box] = []
        for top, right, bottom, left in boxes:
            scaled.append((
                max(0, int(top * inv)), min(width - 1, int(right * inv)),
                min(height - 1, int(bottom * inv)), max(0, int(left * inv)),
            ))
        return scaled

    def detect_single(self, frame: np.ndarray) -> Box:
        """Return the only face in the frame, or raise NoFaceError / MultipleFacesError."""
        boxes = self.detect(frame)
        if not boxes:
            raise NoFaceError("No face detected. Face the camera in good lighting.")
        if len(boxes) > 1:
            raise MultipleFacesError(f"{len(boxes)} faces detected. Keep only one face in the frame.")
        return boxes[0]


def assess_quality(frame: np.ndarray, box: Box, min_face_size: int, min_sharpness: float) -> QualityReport:
    """Reject faces that are too small or blurry (variance of the Laplacian)."""
    top, right, bottom, left = box
    size = min(right - left, bottom - top)
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return QualityReport(False, "Face region is empty.", 0, 0.0)
    sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if size < min_face_size:
        return QualityReport(False, "Face too small - move closer to the camera.", size, sharpness)
    if sharpness < min_sharpness:
        return QualityReport(False, "Image too blurry - hold still and improve lighting.", size, sharpness)
    return QualityReport(True, "OK", size, sharpness)
