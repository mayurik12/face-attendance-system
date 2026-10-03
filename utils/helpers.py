"""Small reusable helpers (image checks, dates, drawing)."""
from __future__ import annotations

from datetime import date, datetime

import cv2
import numpy as np

from utils.errors import InvalidImageError, ValidationError

Box = tuple[int, int, int, int]  # (top, right, bottom, left) - same order as dlib


def ensure_valid_frame(frame: object) -> np.ndarray:
    """Return frame if it is a non-empty uint8 HxWx3 array, else raise InvalidImageError."""
    if not isinstance(frame, np.ndarray):
        raise InvalidImageError("Frame is not an image array.")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise InvalidImageError("Frame must be an 8-bit, 3-channel (BGR) image.")
    if frame.shape[0] < 10 or frame.shape[1] < 10:
        raise InvalidImageError("Frame is too small to process.")
    return frame


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """OpenCV delivers BGR; dlib/face_recognition expect RGB."""
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def parse_date(text: str | None, field: str = "date") -> date | None:
    """Parse YYYY-MM-DD. Empty -> None."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    try:
        return datetime.strptime(cleaned, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValidationError(f"{field} must be in YYYY-MM-DD format.") from exc


def draw_face_box(frame: np.ndarray, box: Box, label: str, color: tuple[int, int, int]) -> None:
    """Draw a labelled rectangle in place."""
    top, right, bottom, left = box
    scale = max(0.55, frame.shape[1] / 900)  # keep text readable on large frames
    thick = max(1, round(scale * 1.5))
    cv2.rectangle(frame, (left, top), (right, bottom), color, thick + 1)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = int(8 * scale)
    y0 = max(0, top - h - 2 * pad)
    x0 = max(0, min(left, frame.shape[1] - (w + 2 * pad)))  # keep the label inside the frame
    cv2.rectangle(frame, (x0, y0), (x0 + w + 2 * pad, y0 + h + 2 * pad), color, cv2.FILLED)
    cv2.putText(frame, label, (x0 + pad, y0 + h + pad), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
